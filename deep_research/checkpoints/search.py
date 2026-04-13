from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.enums import SourceGroup
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
    """Return the effective preferred source groups derived from *preferences*.

    Appends REPOS when comparison targets are present and WEB for comparison or
    decision-support planning modes, preserving the original list order.
    """
    if preferences is None:
        return []
    groups = list(preferences.preferred_source_groups)
    if preferences.comparison_targets and SourceGroup.REPOS not in groups:
        groups.append(SourceGroup.REPOS)
    if (
        preferences.planning_mode.value in {"comparison", "decision_support"}
        and SourceGroup.WEB not in groups
    ):
        groups.append(SourceGroup.WEB)
    return groups


def _resolve_preferred_providers(
    preferences: ResearchPreferences | None,
) -> list[str]:
    """Return the effective preferred provider names derived from *preferences*.

    Supplements the explicit list with web providers (exa, brave) for WEB/REPOS
    groups and academic providers (semantic_scholar, arxiv) for the PAPERS group.
    """
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


@checkpoint(type="tool_call")
def execute_searches(
    decision: SupervisorDecision,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> SearchExecutionResult:
    """Fan out search actions across applicable providers and return raw results.

    Deduplicates actions by (query, providers, kinds, recency, max_results) before
    dispatching, and caps total dispatched actions to `config.max_tool_calls_per_cycle`.
    """
    with span("execute_searches", action_count=len(decision.search_actions)):
        registry = ProviderRegistry(config)
        raw_results = []
        estimated_cost_usd = 0.0

        excluded_providers = preferences.excluded_providers if preferences else []
        excluded_source_groups = (
            preferences.excluded_source_groups if preferences else []
        )
        preferred_source_groups = _resolve_preferred_source_groups(preferences)
        preferred_providers = _resolve_preferred_providers(preferences)

        seen: set[tuple[object, ...]] = set()
        deduped_actions: list[SearchAction] = []
        for action in decision.search_actions:
            identity = (
                action.query.casefold(),
                tuple(action.preferred_providers),
                tuple(action.preferred_source_kinds),
                action.recency_days,
                action.max_results,
            )
            if identity not in seen:
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
