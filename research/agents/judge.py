"""Judge agent — compares generator outputs in council mode."""

from __future__ import annotations

from pydantic_ai import Agent

from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
from research.contracts.package import CouncilComparison
from research.prompts import get_prompt


def build_judge_agent(model_name: str, model_settings: dict | None = None):
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
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-3.1-pro-preview"``).
        model_settings: Optional provider-specific model settings dict
            (e.g. ``{"google_thinking_config": {"thinking_level": "high"}}``).

    Returns:
        A :class:`KitaruAgent` with ``CouncilComparison`` output type.
    """
    agent = Agent(
        model_name,
        output_type=CouncilComparison,
        system_prompt=get_prompt("council_judge").text,
        model_settings=model_settings,
    )
    return KitaruAgent(
        agent, name="council_judge", capture=CapturePolicy(tool_capture="full")
    )
