from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import (
    IterationBudget,
    ResearchPreferences,
    SearchAction,
    SearchExecutionResult,
    SupervisorDecision,
)
from deep_research.observability import span
from deep_research.providers.search import ProviderRegistry


def _resolve_preferred_source_groups(
    preferences: ResearchPreferences | None,
) -> list[SourceGroup]:
    if preferences is None:
        return []
    groups = list(preferences.preferred_source_groups)
    comparison_targets = list(getattr(preferences, "comparison_targets", []) or [])
    if comparison_targets and SourceGroup.REPOS not in groups:
        groups.append(SourceGroup.REPOS)
    if (
        getattr(preferences, "planning_mode", None) is not None
        and preferences.planning_mode.value in {"comparison", "decision_support"}
        and SourceGroup.WEB not in groups
    ):
        groups.append(SourceGroup.WEB)
    return groups


def _resolve_preferred_providers(
    preferences: ResearchPreferences | None,
) -> list[str]:
    if preferences is None:
        return []
    preferred = list(preferences.preferred_providers)
    preferred_groups = set(_resolve_preferred_source_groups(preferences))
    if SourceGroup.WEB in preferred_groups or SourceGroup.REPOS in preferred_groups:
        for provider_name in ("exa", "brave"):
            if provider_name not in preferred:
                preferred.append(provider_name)
    if SourceGroup.PAPERS in preferred_groups:
        for provider_name in ("semantic_scholar", "arxiv"):
            if provider_name not in preferred:
                preferred.append(provider_name)
    return preferred


def _default_action_for_seed_query(query: str) -> SearchAction:
    return SearchAction(
        query=query,
        rationale="Bootstrap named entities and concrete systems before detailed planning.",
        preferred_source_kinds=[SourceKind.WEB],
        max_results=5,
    )


@checkpoint(type="tool_call")
def execute_searches(
    decision: SupervisorDecision,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> SearchExecutionResult:
    with span("execute_searches", action_count=len(decision.search_actions)):
        registry = ProviderRegistry(config)
        raw_results = []
        estimated_cost_usd = 0.0
        seen: set[tuple[object, ...]] = set()
        deduped_actions: list[SearchAction] = []

        excluded_providers = preferences.excluded_providers if preferences else []
        excluded_source_groups = (
            preferences.excluded_source_groups if preferences else []
        )
        preferred_source_groups = _resolve_preferred_source_groups(preferences)
        preferred_providers = _resolve_preferred_providers(preferences)

        for action in decision.search_actions:
            identity = (
                action.query.casefold(),
                tuple(action.preferred_providers),
                tuple(action.preferred_source_kinds),
                action.recency_days,
                action.max_results,
            )
            if identity in seen:
                continue
            seen.add(identity)
            deduped_actions.append(action)

        for action in deduped_actions[: config.max_tool_calls_per_cycle]:
            providers = registry.providers_for(
                action,
                excluded_providers=excluded_providers,
                excluded_source_groups=excluded_source_groups,
                preferred_source_groups=preferred_source_groups,
                preferred_providers=preferred_providers,
            )
            for provider in providers:
                raw_results.extend(
                    provider.search(
                        [action.query],
                        max_results_per_query=action.max_results
                        or config.max_results_per_query,
                        recency_days=action.recency_days,
                    )
                )
                estimated_cost_usd += provider.estimate_cost_usd(1)

        return SearchExecutionResult(
            raw_results=raw_results,
            budget=IterationBudget(estimated_cost_usd=round(estimated_cost_usd, 6)),
        )


@checkpoint(type="tool_call")
def seed_brief_entities(
    brief: str,
    config: ResearchConfig,
    comparison_targets: list[str] | None = None,
    max_queries: int = 5,
) -> dict[str, list[str]]:
    """Bootstrap concrete named entities from lightweight web-first searches.

    This checkpoint is additive and intentionally returns a plain mapping so the
    flow can adopt it before the typed model contract lands everywhere else.
    """
    registry = ProviderRegistry(config)
    comparison_targets = [target.strip() for target in comparison_targets or [] if target]
    seed_queries = comparison_targets or [brief.strip()]
    queries = seed_queries[:max_queries]
    discovered: dict[str, list[str]] = {
        "projects": [],
        "benchmarks": [],
        "products": [],
        "companies": [],
        "key_terms": [],
    }
    seen_titles: set[str] = set()

    with span("seed_brief_entities", query_count=len(queries)):
        for query in queries:
            action = _default_action_for_seed_query(query)
            providers = registry.providers_for(
                action,
                preferred_source_groups=[SourceGroup.WEB, SourceGroup.REPOS],
                preferred_providers=["exa", "brave"],
            )
            if not providers:
                continue
            provider = providers[0]
            for result in provider.search(
                [query],
                max_results_per_query=action.max_results or config.max_results_per_query,
            ):
                for item in (result.payload.get("results") or []):
                    title = str(item.get("title") or "").strip()
                    if not title:
                        continue
                    folded = title.casefold()
                    if folded in seen_titles:
                        continue
                    seen_titles.add(folded)
                    if "benchmark" in folded or "bench" in folded:
                        bucket = "benchmarks"
                    elif any(token in folded for token in ("github", "repo", "repository")):
                        bucket = "projects"
                    else:
                        bucket = "key_terms"
                    discovered[bucket].append(title)

        return {
            key: values[:10]
            for key, values in discovered.items()
            if values
        }
