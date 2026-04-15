"""Scope checkpoint — runs the scope agent to produce a ResearchBrief."""

from kitaru import checkpoint

from research.agents.scope import build_scope_agent
from research.contracts.brief import ResearchBrief


@checkpoint(type="llm_call")
def run_scope(raw_request: str, model_name: str) -> ResearchBrief:
    """Checkpoint: classify and normalize a user request into a ResearchBrief.

    Args:
        raw_request: The user's original research question/request.
        model_name: PydanticAI model string for the scope agent.

    Returns:
        A normalized ResearchBrief.
    """
    agent = build_scope_agent(model_name)
    return agent.run_sync(raw_request).output
