"""Finalizer agent — revises a draft report in response to critique.

Uses ``output_type=str`` (plain text) for the same reason as the generator:
long-form markdown is a poor fit for tool-call schemas. The checkpoint layer
wraps the output in a ``FinalReport`` after extracting section headings.
"""

from __future__ import annotations

from research.agents._factory import _build_agent


def build_finalizer_agent(model_name: str):
    """Build the finalizer agent that produces the final report."""
    return _build_agent(
        model_name,
        name="finalizer",
        prompt_name="finalizer",
        output_type=str,
    )
