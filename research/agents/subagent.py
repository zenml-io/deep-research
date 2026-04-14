"""Subagent — executes search tasks and synthesizes findings."""

from __future__ import annotations

from pydantic_ai import Agent

from research.agents._wrap import wrap_agent
from research.contracts.decisions import SubagentFindings
from research.prompts import get_prompt


def build_subagent_agent(model_name: str, tools=None):
    """Build a subagent with tool access for research tasks.

    The subagent is the only agent slot bound to the inward tool surface.
    It receives search/fetch/code_exec tools and uses them to investigate
    specific subtopics assigned by the supervisor.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).
        tools: Optional list of tool functions/callables for the agent.
            Typically ``search``, ``fetch``, and optionally ``code_exec``
            from :class:`~research.providers.agent_tools.AgentToolSurface`.

    Returns:
        A Kitaru-wrapped PydanticAI agent with ``SubagentFindings`` output type.
    """
    kwargs: dict = {
        "output_type": SubagentFindings,
        "system_prompt": get_prompt("subagent").text,
    }
    if tools:
        kwargs["tools"] = tools

    agent = Agent(model_name, **kwargs)
    return wrap_agent(agent, name="subagent")
