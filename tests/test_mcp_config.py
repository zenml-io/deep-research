from deep_research.providers.mcp_config import MCPServerConfig, build_mcp_toolsets


def test_build_mcp_toolsets_returns_empty_without_servers() -> None:
    assert build_mcp_toolsets([]) == []


def test_build_mcp_toolsets_calls_factories() -> None:
    sentinel = object()
    toolsets = build_mcp_toolsets(
        [MCPServerConfig(id="brave", factory=lambda: sentinel)]
    )
    assert toolsets == [sentinel]
