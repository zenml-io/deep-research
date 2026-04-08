from kitaru import checkpoint

from deep_research.checkpoints.supervisor import run_supervisor
from deep_research.config import ResearchConfig
from deep_research.flow.costing import merge_usage
from deep_research.models import (
    EvidenceLedger,
    IterationBudget,
    RawToolResult,
    ResearchPlan,
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
    return run_supervisor(plan, ledger, iteration, override_config, uncovered_subtopics)


def aggregate_council_results(
    grouped_results: list[SupervisorCheckpointResult],
) -> SupervisorCheckpointResult:
    """Merge raw results and budgets from all council members into one result."""
    merged: list[RawToolResult] = []
    budget = IterationBudget()
    for group in grouped_results:
        merged.extend(group.raw_results)
        budget = merge_usage(budget, group.budget)
    return SupervisorCheckpointResult(raw_results=merged, budget=budget)
