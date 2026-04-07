from deep_research.models import EvidenceLedger, ResearchPlan
from deep_research.providers import build_supervisor_surface
from deep_research.providers.mcp_config import MCPServerConfig


def _sample_plan() -> ResearchPlan:
    """Return a representative research plan fixture for provider surface tests."""
    return ResearchPlan(
        goal="goal",
        key_questions=["k"],
        subtopics=["status", "impact"],
        queries=["q"],
        sections=["Summary"],
        success_criteria=["c"],
    )


def test_build_supervisor_surface_returns_local_tools_and_mcp_toolsets() -> None:
    sentinel_toolset = object()

    toolsets, tools = build_supervisor_surface(
        _sample_plan(),
        EvidenceLedger(entries=[]),
        uncovered_subtopics=["status"],
        tool_timeout_sec=45,
        mcp_servers=[MCPServerConfig(id="brave", factory=lambda: sentinel_toolset)],
    )

    assert toolsets == [sentinel_toolset]
    assert {tool.__name__ for tool in tools} == {
        "read_plan_tool",
        "read_gaps_tool",
        "run_bash_tool",
    }


def test_build_supervisor_surface_read_gaps_tool_prefers_explicit_gap_state() -> None:
    _, tools = build_supervisor_surface(
        _sample_plan(),
        EvidenceLedger(entries=[]),
        uncovered_subtopics=["impact"],
        tool_timeout_sec=45,
    )

    read_gaps_tool = next(tool for tool in tools if tool.__name__ == "read_gaps_tool")

    assert read_gaps_tool() == ["impact"]
