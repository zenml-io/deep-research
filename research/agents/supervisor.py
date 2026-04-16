"""Supervisor agent — decides continue/stop and dispatches subagent tasks.

The supervisor is a pure decision-maker with **no tools**. It reads the
current research state (brief, plan, ledger summary, budget, iteration
index) and returns a ``SupervisorDecision``. This is a structural guard
against supervisor-as-executor creep.
"""

from __future__ import annotations

from research.agents._factory import BudgetAwareAgent, _build_agent
from research.contracts.decisions import SupervisorDecision


def build_supervisor_agent(model_name: str) -> BudgetAwareAgent:
    return _build_agent(
        model_name,
        name="supervisor",
        prompt_name="supervisor",
        output_type=SupervisorDecision,
    )
