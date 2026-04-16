"""Subagent checkpoint — runs a subagent for a single task with graceful degradation."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from kitaru import checkpoint

from research.agents.subagent import build_subagent_agent
from research.contracts.decisions import SubagentFindings
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


@checkpoint(type="llm_call")
def run_subagent(
    task: SubagentTask,
    model_name: str,
    tools: list[Callable[..., Any]] | None = None,
) -> SubagentFindings:
    """Checkpoint: execute a single subagent task.

    On transient HTTP errors (429, 500, 502, 503, 504), retries up to
    ``_MAX_ATTEMPTS`` times with exponential backoff.

    On permanent failure (non-retryable error or retries exhausted),
    returns a degraded SubagentFindings with an error note rather than
    crashing the entire run — following the design spec's requirement
    that subagent failures degrade gracefully.

    Called via Kitaru ``.submit().load()`` for parallel fan-out at the
    flow level, but this checkpoint handles a single task.

    Args:
        task: The subagent task to execute.
        model_name: PydanticAI model string for the subagent.
        tools: Optional tools to pass to the subagent (search, fetch, etc.).

    Returns:
        SubagentFindings (possibly degraded on failure).
    """
    agent = None
    last_exc: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            if agent is None:
                agent = build_subagent_agent(model_name, tools=tools)
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
