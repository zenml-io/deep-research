"""Agent wrapping infrastructure — single Kitaru adapter import point.

All PydanticAI agents should be wrapped through this module at module scope,
NOT inside checkpoint functions. This ensures Kitaru can track agent calls
for durable execution and replay.
"""

from kitaru.adapters import pydantic_ai as kp


def wrap_agent(agent, *, name=None):
    """Wrap a PydanticAI agent with Kitaru's adapter for durable execution.

    Args:
        agent: A PydanticAI Agent instance to wrap.
        name: Optional name for the wrapped agent (used in Kitaru spans).

    Returns:
        The wrapped agent, as returned by ``kp.wrap()``.
    """
    return kp.wrap(agent, name=name, tool_capture_config={"mode": "full"})
