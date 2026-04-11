from __future__ import annotations

from typing import Any

from kitaru.adapters import pydantic_ai as kp


def wrap_agent(agent: Any, tool_capture_config: dict[str, Any] | None = None) -> Any:
    """Return a Kitaru-wrapped agent.

    Raises ImportError or TypeError on adapter incompatibility instead of
    silently falling back to a raw agent. Callers should treat Kitaru as a
    hard dependency.
    """
    return kp.wrap(agent, tool_capture_config=tool_capture_config)
