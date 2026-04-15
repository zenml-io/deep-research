"""Planner agent — decomposes a ResearchBrief into a ResearchPlan."""

from __future__ import annotations

from pydantic_ai import Agent

from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
from research.contracts import ResearchPlan
from research.prompts import get_prompt


def build_planner_agent(model_name: str):
    """Build the planner agent that creates investigation plans from briefs.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A :class:`KitaruAgent` with ``ResearchPlan`` output type.
    """
    agent = Agent(
        model_name,
        output_type=ResearchPlan,
        system_prompt=get_prompt("planner").text,
    )
    return KitaruAgent(
        agent, name="planner", capture=CapturePolicy(tool_capture="full")
    )
