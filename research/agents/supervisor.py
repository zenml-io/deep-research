"""Supervisor agent — decides continue/stop and dispatches subagent tasks."""

from __future__ import annotations

from pydantic_ai import Agent

from research.agents._wrap import wrap_agent
from research.contracts.decisions import SupervisorDecision
from research.prompts import get_prompt


def build_supervisor_agent(model_name: str):
    """Build the supervisor agent that orchestrates the research loop.

    The supervisor reads the current research state (brief, plan, ledger
    summary, budget, iteration index) and returns a ``SupervisorDecision``
    indicating whether to continue or stop, what gaps remain, and what
    subagent tasks to dispatch next.

    **No tools.** The supervisor is a pure decision-maker — it delegates
    all execution to subagents. This is a structural guard against
    supervisor-as-executor creep.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A Kitaru-wrapped PydanticAI agent with ``SupervisorDecision`` output type.
    """
    agent = Agent(
        model_name,
        output_type=SupervisorDecision,
        system_prompt=get_prompt("supervisor").text,
    )
    return wrap_agent(agent, name="supervisor")
