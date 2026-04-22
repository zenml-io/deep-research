"""Tests for V2 convergence stop rules.

Pure Python — no Kitaru or PydanticAI imports required.
Each test covers a specific stop rule or interaction between rules.
"""

from __future__ import annotations

import pytest

from research.config.budget import BudgetConfig
from research.contracts.decisions import SupervisorDecision
from research.flows.convergence import StopDecision, StopReason, check_convergence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget(spent: float = 0.0, soft: float = 0.10) -> BudgetConfig:
    return BudgetConfig(soft_budget_usd=soft, spent_usd=spent)


def _supervisor(done: bool, rationale: str = "test") -> SupervisorDecision:
    return SupervisorDecision(done=done, rationale=rationale)


# ---------------------------------------------------------------------------
# 1. Budget exhausted
# ---------------------------------------------------------------------------


class TestBudgetExhausted:
    """Budget stop fires when spent_usd >= soft_budget_usd."""

    def test_budget_exceeded(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.15, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_at_exact_limit(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.10, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_not_exceeded(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.05, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False

    def test_budget_zero_spent(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.0, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=None,
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False


# ---------------------------------------------------------------------------
# 2. Time exhausted
# ---------------------------------------------------------------------------


class TestTimeExhausted:
    """Time stop fires when elapsed_seconds >= time_limit_seconds."""

    def test_time_exceeded(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=4000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_time_at_exact_limit(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=3600.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_time_not_exceeded(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=100.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False


# ---------------------------------------------------------------------------
# 3. Supervisor done
# ---------------------------------------------------------------------------


class TestSupervisorDone:
    """Supervisor stop fires when supervisor_decision.done is True."""

    def test_supervisor_signals_done(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.SUPERVISOR_DONE

    def test_supervisor_not_done(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False

    def test_supervisor_none(self) -> None:
        """No supervisor decision yet (e.g. before first iteration)."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=None,
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False


# ---------------------------------------------------------------------------
# 4. Max iterations
# ---------------------------------------------------------------------------


class TestMaxIterations:
    """Max iterations stop fires when iteration_index + 1 >= max_iterations."""

    def test_max_iterations_reached(self) -> None:
        # iteration_index=4 means 5 iterations (0..4) completed, max=5
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=4,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.MAX_ITERATIONS

    def test_max_iterations_exceeded(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=6,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.MAX_ITERATIONS

    def test_max_iterations_not_reached(self) -> None:
        # iteration_index=2 means 3 iterations done, max=5 → 2 more to go
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=2,
            max_iterations=5,
        )
        assert result.should_stop is False

    def test_single_iteration_allowed(self) -> None:
        """max_iterations=1 means stop after iteration 0."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=1,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Priority ordering — when multiple rules fire, highest-priority wins
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """When multiple stop conditions are satisfied simultaneously,
    the highest-priority reason is returned."""

    def test_budget_beats_time(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.20, soft=0.10),
            elapsed_seconds=5000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_beats_supervisor_done(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.20, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_beats_max_iterations(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.20, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=4,
            max_iterations=5,
        )
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_budget_beats_all(self) -> None:
        """Budget wins when all four rules fire simultaneously."""
        result = check_convergence(
            budget=_budget(spent=0.20, soft=0.10),
            elapsed_seconds=5000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=4,
            max_iterations=5,
        )
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_time_beats_supervisor_done(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=5000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_time_beats_max_iterations(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=5000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=4,
            max_iterations=5,
        )
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_supervisor_done_beats_max_iterations(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=4,
            max_iterations=5,
        )
        assert result.reason is StopReason.SUPERVISOR_DONE


# ---------------------------------------------------------------------------
# No stop — everything is within limits
# ---------------------------------------------------------------------------


class TestNoStop:
    """When no rule fires, should_stop is False and reason is None."""

    def test_all_within_limits(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False
        assert result.reason is None

    def test_no_supervisor_decision_yet(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.0),
            elapsed_seconds=0.0,
            time_limit_seconds=3600.0,
            supervisor_decision=None,
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False
        assert result.reason is None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """StopDecision.diagnostics carries useful context for logging."""

    def test_diagnostics_present_on_stop(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.20, soft=0.10),
            elapsed_seconds=42.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=2,
            max_iterations=5,
        )
        assert result.diagnostics["spent_usd"] == pytest.approx(0.20)
        assert result.diagnostics["soft_budget_usd"] == pytest.approx(0.10)
        assert result.diagnostics["elapsed_seconds"] == pytest.approx(42.0)
        assert result.diagnostics["time_limit_seconds"] == pytest.approx(3600.0)
        assert result.diagnostics["iteration_index"] == 2
        assert result.diagnostics["max_iterations"] == 5
        assert result.diagnostics["supervisor_done"] is False

    def test_diagnostics_present_on_continue(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is False
        assert "spent_usd" in result.diagnostics

    def test_diagnostics_supervisor_none(self) -> None:
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=None,
            iteration_index=0,
            max_iterations=5,
        )
        assert result.diagnostics["supervisor_done"] is False


# ---------------------------------------------------------------------------
# StopDecision model
# ---------------------------------------------------------------------------


class TestStopDecisionModel:
    """StopDecision Pydantic model behaves correctly."""

    def test_default_values(self) -> None:
        d = StopDecision(should_stop=False)
        assert d.reason is None
        assert d.diagnostics == {}

    def test_serialization_roundtrip(self) -> None:
        d = StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            diagnostics={"spent_usd": 0.15},
        )
        dumped = d.model_dump()
        assert dumped["reason"] == "budget_exhausted"
        restored = StopDecision.model_validate(dumped)
        assert restored.reason is StopReason.BUDGET_EXHAUSTED


# ---------------------------------------------------------------------------
# StopReason enum
# ---------------------------------------------------------------------------


class TestStopReasonEnum:
    """StopReason has exactly four members with expected string values."""

    def test_member_count(self) -> None:
        assert len(StopReason) == 4

    def test_values(self) -> None:
        assert StopReason.BUDGET_EXHAUSTED == "budget_exhausted"
        assert StopReason.TIME_EXHAUSTED == "time_exhausted"
        assert StopReason.SUPERVISOR_DONE == "supervisor_done"
        assert StopReason.MAX_ITERATIONS == "max_iterations"

    def test_is_str_enum(self) -> None:
        assert isinstance(StopReason.BUDGET_EXHAUSTED, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and unusual input conditions."""

    def test_zero_time_limit(self) -> None:
        """time_limit_seconds=0 means time is immediately exhausted
        (but budget is checked first if also exceeded)."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=0.0,
            time_limit_seconds=0.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_max_iterations_zero(self) -> None:
        """max_iterations=0 means iteration_index+1 >= 0 is always True."""
        result = check_convergence(
            budget=_budget(spent=0.0),
            elapsed_seconds=0.0,
            time_limit_seconds=3600.0,
            supervisor_decision=None,
            iteration_index=0,
            max_iterations=0,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.MAX_ITERATIONS

    def test_budget_stop_still_allows_deliverable(self) -> None:
        """Budget stop sets should_stop=True but the flow can still
        produce a deliverable — the stop only exits the iteration loop."""
        result = check_convergence(
            budget=_budget(spent=0.50, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=1,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.BUDGET_EXHAUSTED
        # The StopDecision doesn't have a "skip deliverable" flag —
        # it's purely about iteration loop exit.  The flow always
        # runs draft/critique/finalize after the loop.

    def test_supervisor_done_with_gaps_still_stops(self) -> None:
        """Even if the supervisor reports gaps, done=True means stop."""
        decision = SupervisorDecision(
            done=True,
            rationale="Enough coverage achieved",
            gaps=["minor gap 1", "minor gap 2"],
        )
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=decision,
            iteration_index=0,
            max_iterations=5,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.SUPERVISOR_DONE


# ---------------------------------------------------------------------------
# 5. respect_supervisor_done flag
# ---------------------------------------------------------------------------


class TestRespectSupervisorDone:
    """Tests for the ``respect_supervisor_done`` parameter.

    When False (exhaustive tier), supervisor done=True is ignored and the
    loop continues until budget, time, or max iterations stop it.
    """

    def test_supervisor_done_ignored_when_disabled(self) -> None:
        """Supervisor says done=True but respect_supervisor_done=False
        means the loop keeps going."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=20,
            respect_supervisor_done=False,
        )
        assert result.should_stop is False
        assert result.reason is None

    def test_budget_still_stops_when_supervisor_ignored(self) -> None:
        """Budget exhaustion beats everything, even with
        respect_supervisor_done=False."""
        result = check_convergence(
            budget=_budget(spent=0.50, soft=0.10),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=20,
            respect_supervisor_done=False,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.BUDGET_EXHAUSTED

    def test_time_still_stops_when_supervisor_ignored(self) -> None:
        """Time exhaustion still fires with respect_supervisor_done=False."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=4000.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=20,
            respect_supervisor_done=False,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.TIME_EXHAUSTED

    def test_max_iterations_still_stops_when_supervisor_ignored(self) -> None:
        """Max iterations fires even with respect_supervisor_done=False."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=19,
            max_iterations=20,
            respect_supervisor_done=False,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.MAX_ITERATIONS

    def test_default_respects_supervisor_done(self) -> None:
        """Default behavior (respect_supervisor_done=True) unchanged."""
        result = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=True),
            iteration_index=0,
            max_iterations=20,
        )
        assert result.should_stop is True
        assert result.reason is StopReason.SUPERVISOR_DONE

    def test_diagnostics_include_respect_flag(self) -> None:
        """Diagnostics dict includes the respect_supervisor_done value."""
        result_true = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=20,
            respect_supervisor_done=True,
        )
        assert result_true.diagnostics["respect_supervisor_done"] is True

        result_false = check_convergence(
            budget=_budget(spent=0.01),
            elapsed_seconds=10.0,
            time_limit_seconds=3600.0,
            supervisor_decision=_supervisor(done=False),
            iteration_index=0,
            max_iterations=20,
            respect_supervisor_done=False,
        )
        assert result_false.diagnostics["respect_supervisor_done"] is False
