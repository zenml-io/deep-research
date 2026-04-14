"""Scope agent — normalizes a raw user request into a ResearchBrief."""

from __future__ import annotations

from pydantic_ai import Agent

from research.agents._wrap import wrap_agent
from research.contracts import ResearchBrief
from research.prompts import get_prompt


def build_scope_agent(model_name: str):
    """Build the scoping agent that normalizes user requests into briefs.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A Kitaru-wrapped PydanticAI agent with ``ResearchBrief`` output type.
    """
    agent = Agent(
        model_name,
        output_type=ResearchBrief,
        system_prompt=get_prompt("scope").text,
    )
    return wrap_agent(agent, name="scope")
