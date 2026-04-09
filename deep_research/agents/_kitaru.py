from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def wrap_agent(agent: Any, tool_capture_config: dict[str, Any] | None = None) -> Any:
    """Return a Kitaru-wrapped agent when the installed adapter is compatible.

    Kitaru's PydanticAI adapter currently drifts across released PydanticAI APIs.
    When that import path breaks at runtime, fall back to the plain agent so the
    rest of the research flow can still execute.
    """
    try:
        from kitaru.adapters import pydantic_ai as kp
    except (ImportError, AttributeError):
        logger.warning("kitaru.adapters.pydantic_ai not available; agent will run without Kitaru tracing")
        return agent

    try:
        return kp.wrap(agent, tool_capture_config=tool_capture_config)
    except TypeError:
        logger.warning("kitaru.adapters.pydantic_ai.wrap() signature mismatch; agent will run without Kitaru tracing")
        return agent
