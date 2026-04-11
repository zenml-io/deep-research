from deep_research.enums import StopReason
from deep_research.flow.convergence import check_convergence
from deep_research.models import CoverageScore, IterationRecord
import pytest


def test_check_convergence_stops_on_coverage_target() -> None:
    current = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.7,
        total=0.733,
        uncovered_subtopics=[],
    )
    history = [IterationRecord(iteration=0, new_candidate_count=4, coverage=0.68)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.CONVERGED


def test_check_convergence_preserves_legacy_total_threshold_when_gaps_omitted() -> None:
    current = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.7,
        total=0.733,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=4, coverage=0.68)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.CONVERGED


def test_check_convergence_stops_on_diminishing_returns() -> None:
    current = CoverageScore(
        subtopic_coverage=0.62,
        source_diversity=0.55,
        evidence_density=0.58,
        total=0.57,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=2, coverage=0.55)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.DIMINISHING_RETURNS


def test_check_convergence_stops_on_loop_stall() -> None:
    current = CoverageScore(
        subtopic_coverage=0.62,
        source_diversity=0.58,
        evidence_density=0.61,
        total=0.60,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=3, coverage=0.60)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.LOOP_STALL


def test_check_convergence_does_not_stall_from_stale_history_only() -> None:
    current = CoverageScore(
        subtopic_coverage=0.62,
        source_diversity=0.58,
        evidence_density=0.61,
        total=0.65,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=0, coverage=0.52)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_budget_exhaustion_outranks_convergence() -> None:
    current = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.7,
        total=0.75,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=4, coverage=0.70)]

    decision = check_convergence(
        current,
        history,
        spent_usd=1.0,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.BUDGET_EXHAUSTED


def test_check_convergence_stops_when_budget_hits_exact_limit() -> None:
    current = CoverageScore(
        subtopic_coverage=0.5,
        source_diversity=0.5,
        evidence_density=0.5,
        total=0.5,
    )

    decision = check_convergence(
        current,
        [],
        spent_usd=1.0,
        elapsed_seconds=30,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.BUDGET_EXHAUSTED


def test_check_convergence_stops_when_time_hits_exact_limit() -> None:
    current = CoverageScore(
        subtopic_coverage=0.5,
        source_diversity=0.5,
        evidence_density=0.5,
        total=0.5,
    )

    decision = check_convergence(
        current,
        [],
        spent_usd=0.01,
        elapsed_seconds=60,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.TIME_EXHAUSTED


def test_check_convergence_stops_at_max_iterations_boundary() -> None:
    current = CoverageScore(
        subtopic_coverage=0.5,
        source_diversity=0.5,
        evidence_density=0.5,
        total=0.5,
    )
    history = [
        IterationRecord(iteration=0, new_candidate_count=1, coverage=0.2),
        IterationRecord(iteration=1, new_candidate_count=1, coverage=0.3),
    ]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=3,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.MAX_ITERATIONS


def test_check_convergence_does_not_stop_before_max_iterations_boundary() -> None:
    current = CoverageScore(
        subtopic_coverage=0.5,
        source_diversity=0.5,
        evidence_density=0.5,
        total=0.5,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=1, coverage=0.2)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=30,
        max_iterations=3,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=60,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_stops_on_zero_novelty_stall() -> None:
    current = CoverageScore(
        subtopic_coverage=0.7,
        source_diversity=0.62,
        evidence_density=0.6,
        total=0.64,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=5, coverage=0.62)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.40,
        elapsed_seconds=180,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=600,
        new_candidate_count=0,
    )

    assert decision.should_stop is True
    assert decision.reason is StopReason.LOOP_STALL


def test_check_convergence_does_not_converge_on_full_subtopic_coverage_alone() -> None:
    current = CoverageScore(
        subtopic_coverage=1.0,
        source_diversity=0.45,
        evidence_density=0.5,
        total=0.65,
        uncovered_subtopics=[],
    )

    decision = check_convergence(
        current,
        [],
        spent_usd=0.35,
        elapsed_seconds=240,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=600,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_does_not_converge_with_uncovered_subtopics_remaining() -> (
    None
):
    current = CoverageScore(
        subtopic_coverage=0.95,
        source_diversity=0.72,
        evidence_density=0.74,
        total=0.72,
        uncovered_subtopics=["regulatory-risks"],
    )

    decision = check_convergence(
        current,
        [],
        spent_usd=0.20,
        elapsed_seconds=120,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=600,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_continues_on_low_gain_when_resources_remain() -> None:
    current = CoverageScore(
        subtopic_coverage=0.62,
        source_diversity=0.55,
        evidence_density=0.58,
        total=0.57,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=2, coverage=0.55)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.01,
        elapsed_seconds=5,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=600,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_does_not_stop_at_exact_epsilon_gain_boundary() -> None:
    current = CoverageScore(
        subtopic_coverage=0.62,
        source_diversity=0.55,
        evidence_density=0.58,
        total=0.6,
    )
    history = [IterationRecord(iteration=0, new_candidate_count=2, coverage=0.55)]

    decision = check_convergence(
        current,
        history,
        spent_usd=0.60,
        elapsed_seconds=400,
        max_iterations=5,
        epsilon=0.05,
        min_coverage=0.70,
        budget_limit_usd=1.0,
        time_limit_seconds=600,
    )

    assert decision.should_stop is False
    assert decision.reason is None


def test_check_convergence_rejects_uncovered_subtopics_override_argument() -> None:
    current = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.7,
        total=0.733,
        uncovered_subtopics=[],
    )

    with pytest.raises(TypeError, match="uncovered_subtopics"):
        check_convergence(
            current,
            [],
            spent_usd=0.01,
            elapsed_seconds=30,
            max_iterations=5,
            epsilon=0.05,
            min_coverage=0.70,
            budget_limit_usd=1.0,
            time_limit_seconds=60,
            uncovered_subtopics=(),
        )


def test_check_convergence_never_returns_supervisor_complete() -> None:
    """SUPERVISOR_COMPLETE is handled in the iteration loop, not by check_convergence."""
    all_reasons = set()
    # Collect all possible stop reasons from representative scenarios
    scenarios = [
        # Budget exhaustion
        dict(
            current=CoverageScore(
                subtopic_coverage=0.5,
                source_diversity=0.5,
                evidence_density=0.5,
                total=0.5,
            ),
            history=[],
            spent_usd=1.0,
            elapsed_seconds=10,
            max_iterations=5,
            epsilon=0.05,
            min_coverage=0.70,
            budget_limit_usd=1.0,
            time_limit_seconds=600,
        ),
        # Time exhaustion
        dict(
            current=CoverageScore(
                subtopic_coverage=0.5,
                source_diversity=0.5,
                evidence_density=0.5,
                total=0.5,
            ),
            history=[],
            spent_usd=0.01,
            elapsed_seconds=600,
            max_iterations=5,
            epsilon=0.05,
            min_coverage=0.70,
            budget_limit_usd=1.0,
            time_limit_seconds=600,
        ),
        # Converged
        dict(
            current=CoverageScore(
                subtopic_coverage=0.8,
                source_diversity=0.7,
                evidence_density=0.7,
                total=0.733,
            ),
            history=[IterationRecord(iteration=0, coverage=0.68)],
            spent_usd=0.01,
            elapsed_seconds=10,
            max_iterations=5,
            epsilon=0.05,
            min_coverage=0.70,
            budget_limit_usd=1.0,
            time_limit_seconds=600,
        ),
        # Loop stall
        dict(
            current=CoverageScore(
                subtopic_coverage=0.62,
                source_diversity=0.58,
                evidence_density=0.61,
                total=0.60,
            ),
            history=[IterationRecord(iteration=0, coverage=0.60)],
            spent_usd=0.01,
            elapsed_seconds=10,
            max_iterations=5,
            epsilon=0.05,
            min_coverage=0.70,
            budget_limit_usd=1.0,
            time_limit_seconds=600,
        ),
    ]
    for kwargs in scenarios:
        current = kwargs.pop("current")
        history = kwargs.pop("history")
        decision = check_convergence(current, history, **kwargs)
        if decision.reason is not None:
            all_reasons.add(decision.reason)

    assert StopReason.SUPERVISOR_COMPLETE not in all_reasons
