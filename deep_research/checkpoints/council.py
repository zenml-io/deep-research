from kitaru import checkpoint

from deep_research.checkpoints.supervisor import _execute_supervisor_turn
from deep_research.config import ResearchConfig
from deep_research.flow.costing import merge_usage
from deep_research.models import (
    EvidenceLedger,
    IterationBudget,
    RawToolResult,
    ResearchPlan,
    SupervisorCheckpointResult,
)

# Council fan-out happens in flow scope via checkpoint.submit(); this module only
# defines the generator checkpoint and pure aggregation helpers.


@checkpoint(type="llm_call")
def run_council_generator(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    model_name: str,
    config: ResearchConfig,
) -> SupervisorCheckpointResult:
    return _execute_supervisor_turn(
        plan,
        ledger,
        iteration,
        config,
        model_name=model_name,
    )


def aggregate_council_results(
    grouped_results: list[SupervisorCheckpointResult],
) -> SupervisorCheckpointResult:
    merged: list[RawToolResult] = []
    budget = IterationBudget()
    for group in grouped_results:
        merged.extend(group.raw_results)
        budget = merge_usage(budget, group.budget)
    return SupervisorCheckpointResult(raw_results=merged, budget=budget)
