from deep_research.enums import StopReason
from deep_research.flow.convergence import check_convergence
from deep_research.models import CoverageScore, IterationRecord


def test_check_convergence_stops_on_coverage_target() -> None:
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

    assert decision.should_stop is True
    assert decision.reason is StopReason.LOOP_STALL
