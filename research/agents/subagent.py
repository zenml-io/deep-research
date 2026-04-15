"""Subagent — executes search tasks and synthesizes findings.

The subagent is the only agent slot bound to the inward tool surface.
It receives search/fetch/code_exec tools and uses them to investigate
specific subtopics assigned by the supervisor.
"""

from __future__ import annotations

from research.agents._factory import _build_agent
from research.contracts.decisions import SubagentFindings


def build_subagent_agent(model_name: str, tools=None):
    """Build a subagent with optional tool access for research tasks."""
    return _build_agent(
        model_name,
        name="subagent",
        prompt_name="subagent",
        output_type=SubagentFindings,
        tools=tools,
    )
