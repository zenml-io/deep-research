"""Generator agent — synthesizes evidence into a draft report."""

from __future__ import annotations

from pydantic_ai import Agent

from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
from research.contracts.reports import DraftReport
from research.prompts import get_prompt


def build_generator_agent(model_name: str):
    """Build the generator agent that produces draft reports from evidence.

    The generator receives the evidence ledger, research plan, and brief,
    and produces a ``DraftReport`` with inline ``[evidence_id]`` citations.

    **No tools.** The generator is a pure synthesis agent — it does not
    search for or fetch additional evidence.

    Args:
        model_name: PydanticAI model string (e.g. ``"google-gla:gemini-2.5-flash"``).

    Returns:
        A :class:`KitaruAgent` with ``DraftReport`` output type.
    """
    agent = Agent(
        model_name,
        output_type=DraftReport,
        system_prompt=get_prompt("generator").text,
    )
    return KitaruAgent(
        agent, name="generator", capture=CapturePolicy(tool_capture="full")
    )
