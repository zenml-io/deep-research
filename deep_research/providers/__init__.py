from typing import Any

from deep_research.models import EvidenceLedger, RawToolResult, ResearchPlan
from deep_research.providers.mcp_config import MCPServerConfig, build_mcp_toolsets
from deep_research.providers.search import ProviderRegistry, SearchProvider
from deep_research.tools.bash_executor import run_bash
from deep_research.tools.state_reader import read_gaps, read_plan


def build_supervisor_surface(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    *,
    uncovered_subtopics: list[str] | None,
    tool_timeout_sec: int,
    allow_bash_tool: bool = False,
    mcp_servers: list[MCPServerConfig] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Build the direct provider surface for supervisor execution."""
    toolsets = build_mcp_toolsets(mcp_servers or [])

    def read_plan_tool() -> dict[str, object]:
        """Return the current research plan as a dictionary."""
        return read_plan(plan)

    def read_gaps_tool() -> list[str]:
        """Return subtopics from the plan not yet covered by gathered evidence."""
        if uncovered_subtopics is not None:
            return list(uncovered_subtopics)
        return read_gaps(plan, ledger)

    tools = [read_plan_tool, read_gaps_tool]

    if allow_bash_tool:

        def run_bash_tool(command: str) -> RawToolResult:
            """Execute an allow-listed shell command and return stdout/stderr."""
            return run_bash(command, timeout_sec=tool_timeout_sec)

        tools.append(run_bash_tool)

    return toolsets, tools


__all__ = [
    "MCPServerConfig",
    "ProviderRegistry",
    "SearchProvider",
    "build_mcp_toolsets",
    "build_supervisor_surface",
]
