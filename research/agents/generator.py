"""Generator agent — synthesizes evidence into a draft report.

Uses ``output_type=str`` (plain text) rather than tool-based structured
output. Long-form markdown reports are a poor fit for tool-call schemas —
models frequently fail to wrap thousands of words inside a JSON ``content``
field. The checkpoint layer parses section headings from the raw markdown
to construct the ``DraftReport`` contract.
"""

from __future__ import annotations

from pydantic_ai import Agent

from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
from research.prompts import get_prompt


def build_generator_agent(model_name: str):
    """Build the generator agent that produces draft reports from evidence.

    The generator receives the evidence ledger, research plan, and brief,
    and produces a markdown report with inline ``[evidence_id]`` citations.

    Returns plain text (``output_type=str``) — the checkpoint layer wraps
    the output in a ``DraftReport`` after extracting section headings.

    **No tools.** The generator is a pure synthesis agent — it does not
    search for or fetch additional evidence.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A :class:`KitaruAgent` with ``str`` output type.
    """
    agent = Agent(
        model_name,
        output_type=str,
        system_prompt=get_prompt("generator").text,
    )
    return KitaruAgent(
        agent, name="generator", capture=CapturePolicy(tool_capture="full")
    )
