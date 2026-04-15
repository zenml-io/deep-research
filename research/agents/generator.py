"""Generator agent — synthesizes evidence into a draft report.

Uses ``output_type=str`` (plain text) rather than tool-based structured
output. Long-form markdown reports are a poor fit for tool-call schemas —
models frequently fail to wrap thousands of words inside a JSON ``content``
field. The checkpoint layer parses section headings from the raw markdown
to construct the ``DraftReport`` contract.
"""

from __future__ import annotations

from research.agents._factory import _build_agent


def build_generator_agent(model_name: str):
    """Build the generator agent that produces draft reports from evidence."""
    return _build_agent(
        model_name,
        name="generator",
        prompt_name="generator",
        output_type=str,
    )
