from deep_research.models import EvidenceLedger, ResearchPlan
from deep_research.providers.mcp_config import MCPServerConfig, build_mcp_toolsets
from deep_research.tools.bash_executor import run_bash
from deep_research.tools.state_reader import read_gaps, read_plan


def build_supervisor_surface(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    *,
    uncovered_subtopics: list[str] | None,
    tool_timeout_sec: int,
    mcp_servers: list[MCPServerConfig] | None = None,
) -> tuple[list[object], list[object]]:
    """Build the direct provider surface for supervisor execution."""
    toolsets = build_mcp_toolsets(mcp_servers or [])

    def read_plan_tool() -> dict:
        return read_plan(plan)

    def read_gaps_tool() -> list[str]:
        if uncovered_subtopics is not None:
            return list(uncovered_subtopics)
        return read_gaps(plan, ledger)

    def run_bash_tool(command: str):
        return run_bash(command, timeout_sec=tool_timeout_sec)

    tools = [read_plan_tool, read_gaps_tool, run_bash_tool]
    return toolsets, tools


__all__ = [
    "MCPServerConfig",
    "build_mcp_toolsets",
    "build_supervisor_surface",
]
