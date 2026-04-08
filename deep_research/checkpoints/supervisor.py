import json

from kitaru import checkpoint

from deep_research.agents.supervisor import build_supervisor_agent
from deep_research.config import ResearchConfig
from deep_research.providers import build_supervisor_surface
from deep_research.models import (
    EvidenceLedger,
    ResearchPlan,
    SupervisorCheckpointResult,
)


@checkpoint(type="llm_call")
def run_supervisor(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    config: ResearchConfig,
    uncovered_subtopics: list[str] | None = None,
) -> SupervisorCheckpointResult:
    """Checkpoint: execute a single supervisor search turn and capture tool results."""
    if uncovered_subtopics is None:
        uncovered_subtopics = list(plan.subtopics)
    toolsets, tools = build_supervisor_surface(
        plan,
        ledger,
        uncovered_subtopics=uncovered_subtopics,
        tool_timeout_sec=config.tool_timeout_sec,
    )
    agent = build_supervisor_agent(
        config.supervisor_model,
        toolsets=toolsets,
        tools=tools,
    )
    prompt = {
        "plan": plan.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
        "uncovered_subtopics": uncovered_subtopics,
        "iteration": iteration,
        "tier": config.tier.value,
        "max_tool_calls": config.max_tool_calls_per_cycle,
        "tool_timeout_sec": config.tool_timeout_sec,
    }
    return agent.run_sync(json.dumps(prompt, indent=2)).output
