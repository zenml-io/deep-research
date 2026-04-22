"""Shared agent construction helper — DRY core for all ``build_*_agent()`` factories.

Every public ``build_*_agent()`` function delegates here. Imports are resolved
at call time (not module level) so that test stubs injected into
``sys.modules`` are picked up without requiring test modifications.

Budget wiring
~~~~~~~~~~~~~
The returned agent is wrapped in a ``BudgetAwareAgent`` that intercepts
``run_sync()``, extracts token usage from the PydanticAI result, and
records it via the run-scoped active ``BudgetTracker`` (if one is active).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BudgetAwareAgent:
    """Thin wrapper that records token usage after every ``run_sync()`` call.

    Delegates all attribute access to the wrapped agent so existing
    call sites (``agent.run_sync(prompt).output``) work unchanged.
    """

    __slots__ = ("_wrapped", "_model_name")

    def __init__(self, wrapped: object, *, model_name: str) -> None:
        self._wrapped = wrapped
        self._model_name = model_name

    # -- budget-instrumented entry point --------------------------------

    def run_sync(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped agent and record token usage if a tracker is active."""
        result = self._wrapped.run_sync(*args, **kwargs)
        self._record_usage(result)
        return result

    # -- internals ------------------------------------------------------

    def _record_usage(self, result: object) -> None:
        """Extract usage from the PydanticAI result and record it.

        Reads ``result.usage()`` (a ``RunUsage``) directly, including
        Anthropic cache-read / cache-write token breakdowns that are
        priced distinctly from regular input tokens.
        """
        from research.flows.budget import get_active_tracker

        tracker = get_active_tracker()
        if tracker is None:
            return

        usage_fn = getattr(result, "usage", None)
        if usage_fn is None:
            return
        usage = usage_fn() if callable(usage_fn) else usage_fn

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read_tokens = getattr(usage, "cache_read_tokens", 0)
        cache_write_tokens = getattr(usage, "cache_write_tokens", 0)

        if (input_tokens + output_tokens + cache_read_tokens + cache_write_tokens) > 0:
            tracker.record_usage(
                self._model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )

    # -- transparent delegation -----------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def _build_agent(
    model_name: str,
    *,
    name: str,
    prompt_name: str,
    output_type: type,
    tools: list[Callable[..., Any]] | None = None,
    model_settings: dict[str, Any] | None = None,
) -> BudgetAwareAgent:
    """Construct a PydanticAI Agent wrapped in a KitaruAgent.

    Late imports ensure compatibility with the stub-injection test pattern
    used in ``test_v2_agents.py``.
    """
    from pydantic_ai import Agent

    from kitaru.adapters.pydantic_ai import CapturePolicy, KitaruAgent
    from research.prompts import get_prompt

    kwargs: dict[str, Any] = {
        "output_type": output_type,
        "system_prompt": get_prompt(prompt_name).text,
    }
    if tools:
        kwargs["tools"] = tools
    if model_settings is not None:
        kwargs["model_settings"] = model_settings

    agent = Agent(model_name, **kwargs)
    kitaru_agent = KitaruAgent(
        agent, name=name, capture=CapturePolicy(tool_capture="full")
    )
    return BudgetAwareAgent(kitaru_agent, model_name=model_name)
