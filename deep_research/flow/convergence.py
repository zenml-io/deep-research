from pydantic import BaseModel

from deep_research.enums import StopReason
from deep_research.models import CoverageScore, IterationRecord


class StopDecision(BaseModel):
    should_stop: bool
    reason: StopReason | None = None


def check_convergence(
    current: CoverageScore,
    history: list[IterationRecord],
    *,
    spent_usd: float,
    elapsed_seconds: int,
    max_iterations: int,
    epsilon: float,
    min_coverage: float,
    budget_limit_usd: float,
    time_limit_seconds: int,
) -> StopDecision:
    if spent_usd >= budget_limit_usd:
        return StopDecision(should_stop=True, reason=StopReason.BUDGET_EXHAUSTED)
    if elapsed_seconds >= time_limit_seconds:
        return StopDecision(should_stop=True, reason=StopReason.TIME_EXHAUSTED)
    if current.total >= min_coverage:
        return StopDecision(should_stop=True, reason=StopReason.CONVERGED)
    if history:
        previous = history[-1]
        coverage_gain = current.total - previous.coverage

        if coverage_gain <= 0:
            return StopDecision(should_stop=True, reason=StopReason.LOOP_STALL)
        if coverage_gain < epsilon:
            return StopDecision(should_stop=True, reason=StopReason.DIMINISHING_RETURNS)
    if len(history) + 1 >= max_iterations:
        return StopDecision(should_stop=True, reason=StopReason.MAX_ITERATIONS)
    return StopDecision(should_stop=False)
