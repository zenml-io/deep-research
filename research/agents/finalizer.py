"""Finalizer agent — revises a draft report in response to critique.

Uses ``output_type=str`` (plain text) for the same reason as the generator:
long-form markdown is a poor fit for tool-call schemas. The checkpoint layer
wraps the output in a ``FinalReport`` after extracting section headings.
"""

from __future__ import annotations

from pydantic_ai import Agent

from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
from research.prompts import get_prompt


def build_finalizer_agent(model_name: str):
    """Build the finalizer agent that produces the final report.

    The finalizer takes a draft report and critique, addresses each
    issue while preserving the generator's voice and framing, and
    produces a markdown report. The checkpoint layer wraps this in
    a ``FinalReport`` with a ``stop_reason`` explaining why research
    concluded.

    Returns plain text (``output_type=str``) — the checkpoint layer wraps
    the output in a ``FinalReport`` after extracting section headings.

    **No tools.** The finalizer is a pure revision agent — it works
    only with the evidence already in the ledger.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A :class:`KitaruAgent` with ``str`` output type.
    """
    agent = Agent(
        model_name,
        output_type=str,
        system_prompt=get_prompt("finalizer").text,
    )
    return KitaruAgent(
        agent, name="finalizer", capture=CapturePolicy(tool_capture="full")
    )
