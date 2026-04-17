"""Subagent checkpoint — runs a subagent for a single task with graceful degradation."""

import json
import logging
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from kitaru import checkpoint

from research.agents.subagent import build_subagent_agent
from research.contracts.decisions import SubagentFindings
from research.contracts.package import SubagentToolSpec
from research.contracts.plan import SubagentTask

logger = logging.getLogger(__name__)

# Retryable HTTP status codes (transient provider errors).
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Retry configuration: max attempts, base backoff seconds, backoff multiplier.
_MAX_ATTEMPTS = 3
_BASE_BACKOFF_SECONDS = 1.0
_BACKOFF_MULTIPLIER = 2.0


def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a retryable transient HTTP error.

    Checks for PydanticAI's ModelHTTPError with a retryable status code.
    Uses duck-typing to avoid hard import dependency.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is not None and status_code in _RETRYABLE_STATUS_CODES:
        return True
    return False


@lru_cache(maxsize=8)
def _tools_for_spec(
    enabled_providers: tuple[str, ...],
    sandbox_enabled: bool,
    sandbox_backend: str | None,
) -> list[Callable[..., Any]]:
    """Build (and cache, by spec shape) the PydanticAI tool list for a subagent.

    Closures aren't fingerprint-stable across processes, so the flow passes
    a ``SubagentToolSpec`` through the checkpoint boundary. A fresh worker
    process rebuilds on first call; within one process, every subagent in
    the run shares the cached tool list.

    Only the three fields the registry + surface actually read
    (``enabled_providers``, ``sandbox_enabled``, ``sandbox_backend``) are
    keys — no tier-dependent config is consulted.
    """
    from research.config import ResearchConfig
    from research.config.settings import ResearchSettings
    from research.providers.agent_tools import build_tool_surface
    from research.providers.search import ProviderRegistry

    settings = ResearchSettings(
        enabled_providers=",".join(enabled_providers),
        sandbox_enabled=sandbox_enabled,
        sandbox_backend=sandbox_backend,
    )
    # Tier is irrelevant for surface construction — ``for_tier`` is only
    # reached so settings get applied. Any valid tier works.
    cfg = ResearchConfig.for_tier("quick", settings=settings)
    registry = ProviderRegistry(cfg)
    surface = build_tool_surface(cfg, registry)
    return surface.as_pydantic_tools()


def _resolve_tools(
    tools: list[Callable[..., Any]] | None,
    tool_spec: SubagentToolSpec | None,
) -> list[Callable[..., Any]] | None:
    """Resolve a callable tool list from either form.

    ``tools`` is the test-only direct-injection path; production flows pass
    a replay-stable ``tool_spec`` and the surface is rebuilt (cached) here.
    """
    if tool_spec is None:
        return tools
    return _tools_for_spec(
        tuple(tool_spec.enabled_providers),
        tool_spec.sandbox_enabled,
        tool_spec.sandbox_backend,
    )


@checkpoint(type="llm_call")
def run_subagent(
    task: SubagentTask,
    model_name: str,
    tools: list[Callable[..., Any]] | None = None,
    tool_spec: SubagentToolSpec | None = None,
) -> SubagentFindings:
    """Checkpoint: execute a single subagent task.

    On transient HTTP errors (429, 500, 502, 503, 504), retries up to
    ``_MAX_ATTEMPTS`` times with exponential backoff.

    On permanent failure (non-retryable error or retries exhausted),
    returns a degraded SubagentFindings with an error note rather than
    crashing the entire run — following the design spec's requirement
    that subagent failures degrade gracefully.

    Args:
        task: The subagent task to execute.
        model_name: PydanticAI model string for the subagent.
        tools: Direct callable list (tests only) — not replay-stable.
        tool_spec: Serialisable tool-surface spec. Preferred for flow
            callers because it gives Kitaru a stable argument fingerprint.

    Returns:
        SubagentFindings (possibly degraded on failure).
    """
    resolved_tools = _resolve_tools(tools, tool_spec)
    agent = None
    last_exc: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            if agent is None:
                agent = build_subagent_agent(model_name, tools=resolved_tools)
            prompt = json.dumps(task.model_dump(mode="json"), indent=2)
            return agent.run_sync(prompt).output
        except Exception as exc:
            last_exc = exc
            if _is_retryable(exc) and attempt < _MAX_ATTEMPTS - 1:
                backoff = _BASE_BACKOFF_SECONDS * (_BACKOFF_MULTIPLIER**attempt)
                logger.warning(
                    "Subagent transient error (attempt %d/%d) for task %r: %s "
                    "— retrying in %.1fs",
                    attempt + 1,
                    _MAX_ATTEMPTS,
                    task.task_description,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                continue
            # Non-retryable or final attempt — fall through to degraded result
            break

    logger.warning(
        "Subagent failed for task %r after %d attempt(s): %s",
        task.task_description,
        _MAX_ATTEMPTS if last_exc and _is_retryable(last_exc) else 1,
        last_exc,
    )
    return SubagentFindings(
        findings=[],
        confidence_notes=f"Subagent failed: {last_exc}",
    )
