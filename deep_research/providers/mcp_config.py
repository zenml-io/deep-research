from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MCPServerConfig:
    id: str
    factory: Callable[[], object]


def build_mcp_toolsets(servers: list[MCPServerConfig]) -> list[object]:
    return [server.factory() for server in servers]
