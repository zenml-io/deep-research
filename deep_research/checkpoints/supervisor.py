import json
from collections.abc import Mapping

from kitaru import checkpoint

from deep_research.config import ModelPricing, ResearchConfig
from deep_research.flow.costing import budget_from_agent_result
from deep_research.providers import build_supervisor_surface
from deep_research.models import (
    EvidenceLedger,
    RawToolResult,
    ResearchPlan,
    SupervisorCheckpointResult,
)
def extract_mcp_raw_results(result: object) -> list[RawToolResult]:
    raw_results: list[RawToolResult] = []
    all_messages = getattr(result, "all_messages", None)
    if not callable(all_messages):
        return raw_results

    for message in all_messages():
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) != "tool-return":
                continue
            content = getattr(part, "content", None)
            if isinstance(content, RawToolResult):
                raw_results.append(content)
                continue
            if not isinstance(content, Mapping):
                continue

            payload = content.get("payload")
            if isinstance(payload, Mapping):
                raw_results.append(
                    RawToolResult(
                        tool_name=str(
                            content.get("tool_name")
                            or getattr(part, "tool_name", "tool")
                        ),
                        provider=str(content.get("provider") or "mcp"),
                        payload=dict(payload),
                        ok=bool(content.get("ok", True)),
                        error=content.get("error"),
                    )
                )
                continue

            if "results" in content or "items" in content or "source_kind" in content:
                raw_results.append(
                    RawToolResult(
                        tool_name=str(
                            content.get("tool_name")
                            or getattr(part, "tool_name", "tool")
                        ),
                        provider=str(content.get("provider") or "mcp"),
                        payload=dict(content),
                        ok=bool(content.get("ok", True)),
                        error=content.get("error"),
                    )
                )
    return raw_results


def execute_supervisor_turn(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    config: ResearchConfig,
    uncovered_subtopics: list[str] | None = None,
) -> SupervisorCheckpointResult:
    from deep_research.agents.supervisor import build_supervisor_agent

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
        "enabled_providers": config.enabled_providers,
    }
    result = agent.run_sync(json.dumps(prompt, indent=2))
    return SupervisorCheckpointResult(
        decision=result.output,
        raw_results=extract_mcp_raw_results(result),
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.supervisor_pricing),
        ),
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
    return execute_supervisor_turn(
        plan,
        ledger,
        iteration,
        config,
        uncovered_subtopics,
    )
