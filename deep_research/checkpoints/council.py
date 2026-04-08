from kitaru import checkpoint

from deep_research.checkpoints.supervisor import (
    execute_supervisor_turn,
)
from deep_research.config import ResearchConfig
from deep_research.flow.costing import merge_usage
from deep_research.models import (
    EvidenceLedger,
    IterationBudget,
    RawToolResult,
    ResearchPlan,
    SearchAction,
    SupervisorDecision,
    SupervisorCheckpointResult,
)


@checkpoint(type="llm_call")
def run_council_generator(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    model_name: str,
    config: ResearchConfig,
    uncovered_subtopics: list[str] | None = None,
) -> SupervisorCheckpointResult:
    """Checkpoint: run one council member's supervisor turn for parallel evidence gathering."""
    override_config = config.model_copy(update={"supervisor_model": model_name})
    return execute_supervisor_turn(
        plan,
        ledger,
        iteration,
        override_config,
        uncovered_subtopics,
    )
@checkpoint(type="tool_call")
def aggregate_council_results(
    grouped_results: list[SupervisorCheckpointResult],
) -> SupervisorCheckpointResult:
    """Merge raw results and budgets from all council members into one result."""
    merged: list[RawToolResult] = []
    budget = IterationBudget()
    merged_actions: list[SearchAction] = []
    seen_actions: set[tuple[object, ...]] = set()
    for group in grouped_results:
        merged.extend(group.raw_results)
        budget = merge_usage(budget, group.budget)
        for action in group.decision.search_actions:
            identity = (
                action.query.casefold(),
                tuple(action.preferred_providers),
                tuple(action.preferred_source_kinds),
                action.recency_days,
                action.max_results,
            )
            if identity in seen_actions:
                continue
            seen_actions.add(identity)
            merged_actions.append(action)
    return SupervisorCheckpointResult(
        decision=SupervisorDecision(
            rationale=f"Aggregated {len(grouped_results)} council decisions.",
            search_actions=merged_actions,
        ),
        raw_results=merged,
        budget=budget,
    )
