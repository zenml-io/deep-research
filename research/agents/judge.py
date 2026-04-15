"""Judge agent — compares generator outputs in council mode.

The judge evaluates two research reports (produced by different generators
from the same evidence base) across five dimensions: grounding, coherence,
completeness, accuracy, and clarity. It should run on a provider distinct
from both generators to eliminate model-bias in the comparison.
"""

from __future__ import annotations

from research.agents._factory import _build_agent
from research.contracts.package import CouncilComparison


def build_judge_agent(model_name: str, model_settings: dict | None = None):
    """Build the council judge agent that compares generator outputs."""
    return _build_agent(
        model_name,
        name="council_judge",
        prompt_name="council_judge",
        output_type=CouncilComparison,
        model_settings=model_settings,
    )
