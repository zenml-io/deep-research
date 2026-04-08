from __future__ import annotations

from typing import Any


def wrap_agent(agent: Any, tool_capture_config: dict[str, Any] | None = None) -> Any:
    """Return a Kitaru-wrapped agent when the installed adapter is compatible.

    Kitaru's PydanticAI adapter currently drifts across released PydanticAI APIs.
    When that import path breaks at runtime, fall back to the plain agent so the
    rest of the research flow can still execute.
    """
    try:
        from kitaru.adapters import pydantic_ai as kp
    except (ImportError, AttributeError):
        return agent

    try:
        return kp.wrap(agent, tool_capture_config=tool_capture_config)
    except TypeError:
        return agent
