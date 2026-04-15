"""Reviewer agent — critiques a draft report using a structured rubric.

On the deep tier, two reviewers from different providers produce independent
critiques that the pipeline merges (union of issues, averaged scores). This
prompt is designed for a single reviewer; merging is handled externally in
the critique checkpoint.
"""

from __future__ import annotations

from research.agents._factory import _build_agent
from research.contracts.reports import CritiqueReport


def build_reviewer_agent(model_name: str):
    """Build the reviewer agent that critiques draft reports."""
    return _build_agent(
        model_name,
        name="reviewer",
        prompt_name="reviewer",
        output_type=CritiqueReport,
    )
