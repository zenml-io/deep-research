"""Supervisor checkpoint — runs the supervisor agent to produce a decision."""

import json

from kitaru import checkpoint

from research.agents.supervisor import build_supervisor_agent
from research.contracts.brief import ResearchBrief
from research.contracts.decisions import SupervisorDecision
from research.contracts.plan import ResearchPlan


@checkpoint(type="llm_call")
def run_supervisor(
    brief: ResearchBrief,
    plan: ResearchPlan,
    ledger_projection: str,
    remaining_budget_usd: float,
    iteration_index: int,
    model_name: str,
    breadth_first: bool = False,
) -> SupervisorDecision:
    """Checkpoint: evaluate research state and decide next actions.

    The supervisor receives the brief, plan, a windowed projection of the
    evidence ledger, budget information, and iteration index. It decides
    whether to continue or stop, identifies gaps, and dispatches subagent tasks.

    Args:
        brief: The normalized research brief.
        plan: The research plan.
        ledger_projection: Formatted string of the current evidence ledger window.
        remaining_budget_usd: Remaining budget in USD.
        iteration_index: Current iteration number (0-indexed).
        model_name: PydanticAI model string for the supervisor agent.
        breadth_first: When True, append a ``"mode": "breadth_first"`` hint to
            the prompt so the supervisor agent prioritises breadth over depth.

    Returns:
        A SupervisorDecision with done flag, gaps, and subagent tasks.
    """
    agent = build_supervisor_agent(model_name)
    prompt_data: dict = {
        "brief": brief.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "ledger_projection": ledger_projection,
        "remaining_budget_usd": remaining_budget_usd,
        "iteration_index": iteration_index,
    }
    if breadth_first:
        prompt_data["mode"] = "breadth_first"
    prompt = json.dumps(prompt_data, indent=2)
    return agent.run_sync(prompt).output
