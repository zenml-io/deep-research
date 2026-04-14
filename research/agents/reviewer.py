"""Reviewer agent — critiques a draft report using a structured rubric."""

from __future__ import annotations

from pydantic_ai import Agent

from research.agents._wrap import wrap_agent
from research.contracts.reports import CritiqueReport
from research.prompts import get_prompt


def build_reviewer_agent(model_name: str):
    """Build the reviewer agent that critiques draft reports.

    The reviewer evaluates a draft report across three dimensions
    (source_reliability, completeness, grounding) and produces a
    ``CritiqueReport`` with scored dimensions, specific issues, and
    a ``require_more_research`` flag.

    **No tools.** The reviewer is a pure evaluation agent — it does not
    search for or modify evidence.

    On the deep tier, two reviewers from different providers produce
    independent critiques that the pipeline merges (union of issues,
    averaged scores). This prompt is designed for a single reviewer;
    merging is handled externally.

    Args:
        model_name: PydanticAI model string (e.g. ``"anthropic:claude-sonnet-4-20250514"``).

    Returns:
        A Kitaru-wrapped PydanticAI agent with ``CritiqueReport`` output type.
    """
    agent = Agent(
        model_name,
        output_type=CritiqueReport,
        system_prompt=get_prompt("reviewer").text,
    )
    return wrap_agent(agent, name="reviewer")
