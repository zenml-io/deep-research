"""Judge agent — compares generator outputs in council mode."""

from __future__ import annotations

from pydantic_ai import Agent

from research.agents._wrap import wrap_agent
from research.contracts.package import CouncilComparison
from research.prompts import get_prompt


def build_judge_agent(model_name: str):
    """Build the council judge agent that compares generator outputs.

    The judge evaluates two research reports (produced by different
    generators from the same evidence base) across five dimensions:
    grounding, coherence, completeness, accuracy, and clarity. It
    produces a ``CouncilComparison`` with a detailed textual comparison,
    per-generator scores, and a recommendation.

    **No tools.** The judge is a pure evaluation agent — it does not
    search for or modify evidence.

    The judge should run on a provider distinct from both generators
    to eliminate model-bias in the comparison.

    Args:
        model_name: PydanticAI model string (e.g. ``"openai:gpt-4o-mini"``).

    Returns:
        A Kitaru-wrapped PydanticAI agent with ``CouncilComparison`` output type.
    """
    agent = Agent(
        model_name,
        output_type=CouncilComparison,
        system_prompt=get_prompt("council_judge").text,
    )
    return wrap_agent(agent, name="council_judge")
