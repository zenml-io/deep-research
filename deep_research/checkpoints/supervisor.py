import json
from collections.abc import Mapping

from kitaru import checkpoint

from deep_research.config import ModelPricing, ResearchConfig
from deep_research.enums import DeliverableMode, PlanningMode
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import ResearchPreferences
from deep_research.providers import build_supervisor_surface
from deep_research.models import (
    EvidenceLedger,
    RawToolResult,
    ResearchPlan,
    SupervisorCheckpointResult,
)


def build_supervisor_guidance(preferences: ResearchPreferences) -> str:
    """Turn structured preferences into natural language guidance for the supervisor."""
    parts: list[str] = []

    if preferences.preferred_source_groups:
        groups = ", ".join(g.value for g in preferences.preferred_source_groups)
        parts.append(
            f"The user prefers these source types: {groups}. Weight your search actions toward them."
        )

    if preferences.excluded_source_groups:
        groups = ", ".join(g.value for g in preferences.excluded_source_groups)
        parts.append(
            f"The user has excluded these source types: {groups}. They are hard-blocked at the provider level."
        )

    if preferences.freshness:
        parts.append(
            f"Freshness preference: {preferences.freshness}. Bias toward recency-constrained queries."
        )

    if preferences.time_window_days:
        parts.append(
            f"Time window: last {preferences.time_window_days} days. Use recency_days in search actions when appropriate."
        )

    if preferences.comparison_targets:
        targets = " vs ".join(preferences.comparison_targets)
        parts.append(
            f"This is a comparison: {targets}. Ensure balanced evidence for all targets."
        )

    if preferences.planning_mode != PlanningMode.BROAD_SCAN:
        parts.append(
            f"Research mode: {preferences.planning_mode.value}. Adapt your search strategy accordingly."
        )

    if preferences.deliverable_mode != DeliverableMode.RESEARCH_PACKAGE:
        parts.append(
            f"Output mode: {preferences.deliverable_mode.value}. Focus evidence gathering on what this output needs."
        )

    if preferences.audience:
        parts.append(f"Target audience: {preferences.audience}.")

    if preferences.cost_bias == "minimize":
        parts.append(
            "Cost sensitivity: minimize. Prefer free providers and fewer queries."
        )
    elif preferences.cost_bias == "no_limit":
        parts.append("Cost sensitivity: none. Use all available providers freely.")

    if preferences.speed_bias == "fast":
        parts.append("Speed preference: fast. Fewer, more targeted queries.")
    elif preferences.speed_bias == "thorough":
        parts.append("Speed preference: thorough. More queries, broader coverage.")

    if not parts:
        return "No specific user preferences. Use your best judgment."

    return " ".join(parts)


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
    brief: str | None = None,
    preferences: ResearchPreferences | None = None,
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
    prompt: dict[str, object] = {
        "plan": plan.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
        "uncovered_subtopics": uncovered_subtopics,
        "iteration": iteration,
        "tier": config.tier.value,
        "max_tool_calls": config.max_tool_calls_per_cycle,
        "tool_timeout_sec": config.tool_timeout_sec,
        "enabled_providers": config.enabled_providers,
    }
    if brief is not None:
        prompt["user_brief"] = brief
    if preferences is not None:
        prompt["preferences"] = preferences.model_dump(mode="json")
        prompt["guidance"] = build_supervisor_guidance(preferences)
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
    brief: str | None = None,
    preferences: ResearchPreferences | None = None,
) -> SupervisorCheckpointResult:
    """Checkpoint: execute a single supervisor search turn and capture tool results."""
    return execute_supervisor_turn(
        plan,
        ledger,
        iteration,
        config,
        uncovered_subtopics,
        brief=brief,
        preferences=preferences,
    )
