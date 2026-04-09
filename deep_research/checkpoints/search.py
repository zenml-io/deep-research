from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.models import (
    IterationBudget,
    ResearchPreferences,
    SearchAction,
    SearchExecutionResult,
    SupervisorDecision,
)
from deep_research.providers.search import ProviderRegistry


@checkpoint(type="tool_call")
def execute_searches(
    decision: SupervisorDecision,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> SearchExecutionResult:
    registry = ProviderRegistry(config)
    raw_results = []
    estimated_cost_usd = 0.0
    seen: set[tuple[object, ...]] = set()
    deduped_actions: list[SearchAction] = []

    excluded_providers = preferences.excluded_providers if preferences else []
    excluded_source_groups = preferences.excluded_source_groups if preferences else []

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
