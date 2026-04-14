"""Convergence stop rules for the V2 research iteration loop.

Checks four stop conditions in strict priority order:
1. Budget exhausted — soft budget reached or exceeded.
2. Time exhausted — wall-clock elapsed >= configured limit.
3. Supervisor done — supervisor has signalled completion.
4. Max iterations — iteration count >= configured ceiling.

The function is pure logic with no side effects.  Budget stops do NOT
prevent deliverable production — the flow exits the iteration loop but
still runs draft/critique/finalize/assemble.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from research.config.budget import BudgetConfig
from research.contracts.decisions import SupervisorDecision


class StopReason(StrEnum):
    """Why the iteration loop stopped."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    SUPERVISOR_DONE = "supervisor_done"
    MAX_ITERATIONS = "max_iterations"


class StopDecision(BaseModel):
    """Result of a convergence check.

    ``should_stop=False`` means the iteration loop should continue.
    When ``should_stop=True``, ``reason`` identifies the highest-priority
    rule that fired.  ``diagnostics`` carries numeric context for logging.
    """

    should_stop: bool
    reason: StopReason | None = None
    diagnostics: dict[str, float | int | str | bool] = {}


def check_convergence(
    *,
    budget: BudgetConfig,
    elapsed_seconds: float,
    time_limit_seconds: float,
    supervisor_decision: SupervisorDecision | None,
    iteration_index: int,
    max_iterations: int,
) -> StopDecision:
    """Check stop rules in priority order and return the first that fires.

    Parameters
    ----------
    budget:
        Current budget state (checked via ``is_exceeded()``).
    elapsed_seconds:
        Wall-clock seconds since the run started.
    time_limit_seconds:
        Maximum allowed wall-clock seconds for the iteration loop.
    supervisor_decision:
        The supervisor's latest decision, or ``None`` if no iteration
        has run yet.
    iteration_index:
        Zero-based index of the *completed* iteration (i.e. after
        iteration 0 finishes, this is 0).  The next iteration would
        be ``iteration_index + 1``.
    max_iterations:
        Maximum number of iterations allowed.

    Returns
    -------
    StopDecision
        ``should_stop=True`` with the winning ``reason`` if any rule
        fires, otherwise ``should_stop=False``.
    """
    diagnostics: dict[str, float | int | str | bool] = {
        "spent_usd": budget.spent_usd,
        "soft_budget_usd": budget.soft_budget_usd,
        "elapsed_seconds": elapsed_seconds,
        "time_limit_seconds": time_limit_seconds,
        "iteration_index": iteration_index,
        "max_iterations": max_iterations,
        "supervisor_done": (
            supervisor_decision.done if supervisor_decision is not None else False
        ),
    }

    # 1. Budget exhausted (highest priority)
    if budget.is_exceeded():
        return StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            diagnostics=diagnostics,
        )

    # 2. Time exhausted
    if elapsed_seconds >= time_limit_seconds:
        return StopDecision(
            should_stop=True,
            reason=StopReason.TIME_EXHAUSTED,
            diagnostics=diagnostics,
        )

    # 3. Supervisor done
    if supervisor_decision is not None and supervisor_decision.done:
        return StopDecision(
            should_stop=True,
            reason=StopReason.SUPERVISOR_DONE,
            diagnostics=diagnostics,
        )

    # 4. Max iterations
    #    iteration_index is zero-based and represents the last completed
    #    iteration.  The next iteration would be iteration_index + 1,
    #    so we stop when iteration_index + 1 >= max_iterations.
    if iteration_index + 1 >= max_iterations:
        return StopDecision(
            should_stop=True,
            reason=StopReason.MAX_ITERATIONS,
            diagnostics=diagnostics,
        )

    # Nothing triggered — keep iterating.
    return StopDecision(
        should_stop=False,
        reason=None,
        diagnostics=diagnostics,
    )
