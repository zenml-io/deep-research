from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerConfig:
    """Stable MCP server configuration metadata plus a toolset factory.

    The ``id`` is reserved for higher layers that need a stable identifier for
    configuration, logging, or reporting. ``build_mcp_toolsets`` does not use
    it yet.
    """

    id: str
    factory: Callable[[], object]


def build_mcp_toolsets(servers: list[MCPServerConfig]) -> list[object]:
    """Build toolsets by invoking server factories in the given order."""

    return [server.factory() for server in servers]
