from math import isclose

from pydantic import BaseModel

from deep_research.enums import StopReason
from deep_research.models import CoverageScore, EvidenceLedger, IterationRecord


class StopDecision(BaseModel):
    should_stop: bool
    reason: StopReason | None = None


def detect_source_diversity_warning(ledger: EvidenceLedger) -> str | None:
    selected = list(ledger.selected)
    if len(selected) < 2:
        return None
    provider_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    for candidate in selected:
        provider = getattr(candidate, "provider", None)
        source_kind = getattr(getattr(candidate, "source_kind", None), "value", None)
        if provider is not None:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        if source_kind is not None:
            source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1
    threshold = 0.8
    if provider_counts:
        provider, provider_count = max(provider_counts.items(), key=lambda item: item[1])
        if provider_count / len(selected) > threshold:
            return f"source diversity warning: {provider_count}/{len(selected)} selected sources came from provider '{provider}'"
    if source_kind_counts:
        source_kind, source_kind_count = max(
            source_kind_counts.items(), key=lambda item: item[1]
        )
        if source_kind_count / len(selected) > threshold:
            return f"source diversity warning: {source_kind_count}/{len(selected)} selected sources share source_kind '{source_kind}'"
    return None


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
    new_candidate_count: int | None = None,
) -> StopDecision:
    """Decide whether the research loop should stop based on budget, time, and coverage."""
    has_explicit_gap_state = "uncovered_subtopics" in current.model_fields_set
    has_remaining_gaps = has_explicit_gap_state and bool(current.uncovered_subtopics)
    has_unanswered_questions = bool(current.unanswered_questions)

    if spent_usd >= budget_limit_usd:
        return StopDecision(should_stop=True, reason=StopReason.BUDGET_EXHAUSTED)
    if elapsed_seconds >= time_limit_seconds:
        return StopDecision(should_stop=True, reason=StopReason.TIME_EXHAUSTED)
    if current.total >= min_coverage and not has_remaining_gaps and not has_unanswered_questions:
        return StopDecision(should_stop=True, reason=StopReason.CONVERGED)
    if history:
        previous = history[-1]
        coverage_gain = current.total - previous.coverage
        low_gain = coverage_gain < epsilon and not isclose(
            coverage_gain,
            epsilon,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        spent_ratio = spent_usd / budget_limit_usd if budget_limit_usd > 0 else 1.0
        elapsed_ratio = (
            elapsed_seconds / time_limit_seconds if time_limit_seconds > 0 else 1.0
        )
        resource_pressure = max(spent_ratio, elapsed_ratio)

        if coverage_gain <= 0:
            return StopDecision(should_stop=True, reason=StopReason.LOOP_STALL)
        if new_candidate_count is not None and new_candidate_count <= 0 and low_gain:
            return StopDecision(should_stop=True, reason=StopReason.LOOP_STALL)
        if low_gain and resource_pressure >= 0.5:
            return StopDecision(should_stop=True, reason=StopReason.DIMINISHING_RETURNS)
    if len(history) + 1 >= max_iterations:
        return StopDecision(should_stop=True, reason=StopReason.MAX_ITERATIONS)
    return StopDecision(should_stop=False)
