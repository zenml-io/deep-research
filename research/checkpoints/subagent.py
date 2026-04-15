"""Subagent checkpoint — runs a subagent for a single task with graceful degradation."""

import json
import logging

from kitaru import checkpoint

from research.agents.subagent import build_subagent_agent
from research.contracts.decisions import SubagentFindings
from research.contracts.plan import SubagentTask

logger = logging.getLogger(__name__)


@checkpoint(type="llm_call")
def run_subagent(
    task: SubagentTask,
    model_name: str,
    tools: list | None = None,
) -> SubagentFindings:
    """Checkpoint: execute a single subagent task.

    On success, returns the subagent's findings. On failure, returns a
    degraded SubagentFindings with an error note rather than crashing
    the entire run — following the design spec's requirement that
    subagent failures degrade gracefully.

    Called via Kitaru ``.submit().load()`` for parallel fan-out at the
    flow level, but this checkpoint handles a single task.

    Args:
        task: The subagent task to execute.
        model_name: PydanticAI model string for the subagent.
        tools: Optional tools to pass to the subagent (search, fetch, etc.).

    Returns:
        SubagentFindings (possibly degraded on failure).
    """
    try:
        agent = build_subagent_agent(model_name, tools=tools)
        prompt = json.dumps(task.model_dump(mode="json"), indent=2)
        return agent.run_sync(prompt).output
    except Exception as exc:
        logger.warning("Subagent failed for task %r: %s", task.task_description, exc)
        return SubagentFindings(
            findings=[],
            confidence_notes=f"Subagent failed: {exc}",
        )
