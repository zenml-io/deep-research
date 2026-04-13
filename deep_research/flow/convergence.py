from math import isclose

from pydantic import BaseModel, Field

from deep_research.enums import StopReason
from deep_research.models import CoverageScore, EvidenceLedger, IterationRecord


class StopDecision(BaseModel):
    should_stop: bool
    reason: StopReason | None = None
    continue_reason: str | None = None
    diagnostics: dict[str, int | float | str | bool] = Field(default_factory=dict)


class ConvergenceSignal(BaseModel):
    coverage_total: float = 0.0
    subtopic_coverage: float = 0.0
    unanswered_questions_count: int = 0
    considered_count: int = 0
    selected_count: int = 0
    source_group_count: int = 0
    new_candidate_count: int = 0
    marginal_info_gain: float = 0.0
    stall_count: int = 0
    spent_usd: float = 0.0
    active_elapsed_seconds: int = 0
    wall_elapsed_seconds: int = 0
    total_tokens: int = 0
    token_budget_remaining_ratio: float = 1.0


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
    signal: ConvergenceSignal | CoverageScore,
    history: list[IterationRecord],
    **kwargs,
) -> StopDecision:
    """Decide whether the research loop should stop using a composite signal."""
    if "uncovered_subtopics" in kwargs:
        raise TypeError(
            "check_convergence does not accept 'uncovered_subtopics' override; "
            "pass the state via CoverageScore.uncovered_subtopics instead"
        )
    if isinstance(signal, CoverageScore):
        current = signal
        spent_usd = kwargs["spent_usd"]
        elapsed_seconds = kwargs["elapsed_seconds"]
        max_iterations = kwargs["max_iterations"]
        epsilon = kwargs["epsilon"]
        min_coverage = kwargs["min_coverage"]
        budget_limit_usd = kwargs["budget_limit_usd"]
        time_limit_seconds = kwargs["time_limit_seconds"]
        new_candidate_count = kwargs.get("new_candidate_count", 0)
        has_explicit_gap_state = "uncovered_subtopics" in current.model_fields_set
        has_remaining_gaps = has_explicit_gap_state and bool(current.uncovered_subtopics)
        has_unanswered_questions = bool(current.unanswered_questions)

        if spent_usd >= budget_limit_usd:
            return StopDecision(should_stop=True, reason=StopReason.BUDGET_EXHAUSTED)
        if elapsed_seconds >= time_limit_seconds:
            return StopDecision(should_stop=True, reason=StopReason.TIME_EXHAUSTED)
        if (
            current.total >= min_coverage
            and not has_remaining_gaps
            and not has_unanswered_questions
        ):
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
            if low_gain and resource_pressure >= 0.5:
                return StopDecision(
                    should_stop=True, reason=StopReason.DIMINISHING_RETURNS
                )
            if "new_candidate_count" in kwargs and new_candidate_count <= 0 and low_gain:
                return StopDecision(should_stop=True, reason=StopReason.LOOP_STALL)
        if len(history) + 1 >= max_iterations:
            return StopDecision(should_stop=True, reason=StopReason.MAX_ITERATIONS)
        return StopDecision(should_stop=False)

    max_iterations = kwargs["max_iterations"]
    coverage_threshold = kwargs["coverage_threshold"]
    strong_coverage_threshold = kwargs["strong_coverage_threshold"]
    marginal_gain_threshold = kwargs["marginal_gain_threshold"]
    max_stall_count = kwargs["max_stall_count"]
    budget_limit_usd = kwargs["budget_limit_usd"]
    active_time_limit_seconds = kwargs["active_time_limit_seconds"]
    min_considered_floor = kwargs["min_considered_floor"]
    min_selected_for_stop = kwargs["min_selected_for_stop"]
    diagnostics: dict[str, int | float | str | bool] = {
        "coverage_total": round(signal.coverage_total, 6),
        "subtopic_coverage": round(signal.subtopic_coverage, 6),
        "unanswered_questions_count": signal.unanswered_questions_count,
        "considered_count": signal.considered_count,
        "selected_count": signal.selected_count,
        "source_group_count": signal.source_group_count,
        "new_candidate_count": signal.new_candidate_count,
        "marginal_info_gain": round(signal.marginal_info_gain, 6),
        "stall_count": signal.stall_count,
        "spent_usd": round(signal.spent_usd, 6),
        "active_elapsed_seconds": signal.active_elapsed_seconds,
        "wall_elapsed_seconds": signal.wall_elapsed_seconds,
        "total_tokens": signal.total_tokens,
        "token_budget_remaining_ratio": round(
            max(0.0, signal.token_budget_remaining_ratio), 6
        ),
    }
    if signal.spent_usd >= budget_limit_usd:
        return StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            diagnostics=diagnostics,
        )
    if signal.token_budget_remaining_ratio <= 0.0:
        return StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            diagnostics=diagnostics,
        )
    if signal.active_elapsed_seconds >= active_time_limit_seconds:
        return StopDecision(
            should_stop=True,
            reason=StopReason.TIME_EXHAUSTED,
            diagnostics=diagnostics,
        )

    coverage_ready = signal.subtopic_coverage >= coverage_threshold
    strong_coverage = signal.subtopic_coverage >= strong_coverage_threshold
    breadth_floor_met = signal.considered_count >= min_considered_floor
    enough_selected = signal.selected_count >= min_selected_for_stop
    enough_diversity = signal.source_group_count >= 3
    low_gain = signal.marginal_info_gain < marginal_gain_threshold and not isclose(
        signal.marginal_info_gain,
        marginal_gain_threshold,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    stalled = signal.stall_count >= max_stall_count
    has_open_gaps = signal.unanswered_questions_count > 0

    if coverage_ready and breadth_floor_met and not has_open_gaps and (low_gain or stalled):
        return StopDecision(
            should_stop=True,
            reason=StopReason.CONVERGED,
            diagnostics=diagnostics,
        )
    if strong_coverage and enough_selected and enough_diversity and (low_gain or stalled):
        return StopDecision(
            should_stop=True,
            reason=StopReason.CONVERGED,
            diagnostics=diagnostics,
        )
    if stalled and signal.new_candidate_count <= 0:
        return StopDecision(
            should_stop=True,
            reason=StopReason.LOOP_STALL,
            diagnostics=diagnostics,
        )
    if low_gain and coverage_ready:
        return StopDecision(
            should_stop=True,
            reason=StopReason.DIMINISHING_RETURNS,
            diagnostics=diagnostics,
        )
    if len(history) + 1 >= max_iterations:
        return StopDecision(
            should_stop=True,
            reason=StopReason.MAX_ITERATIONS,
            diagnostics=diagnostics,
        )

    if has_open_gaps:
        continue_reason = "remaining unanswered questions require another wave"
    elif not breadth_floor_met:
        continue_reason = (
            f"breadth floor not met ({signal.considered_count}/{min_considered_floor} considered)"
        )
    elif not enough_selected:
        continue_reason = (
            f"selection set still thin ({signal.selected_count}/{min_selected_for_stop})"
        )
    else:
        continue_reason = "continue exploring for additional corroboration and diversity"

    return StopDecision(
        should_stop=False,
        continue_reason=continue_reason,
        diagnostics=diagnostics,
    )
