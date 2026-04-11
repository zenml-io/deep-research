import inspect

from kitaru import checkpoint

from deep_research.agent_io import (
    ToolResultCollector,
    ToolTraceExtractionStats,
    extract_tool_results,
    extract_tool_results_with_stats,
    serialize_prompt_payload,
)
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.enums import DeliverableMode, PlanningMode
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    EvidenceLedger,
    RawToolResult,
    ResearchPlan,
    ResearchPreferences,
    SupervisorCheckpointResult,
)
from deep_research.observability import bootstrap_logfire, span, warning
from deep_research.providers import build_supervisor_surface


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
    return extract_tool_results(result)


def extract_mcp_raw_results_with_stats(
    result: object,
) -> tuple[list[RawToolResult], ToolTraceExtractionStats]:
    return extract_tool_results_with_stats(result)


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

    bootstrap_logfire()

    if uncovered_subtopics is None:
        uncovered_subtopics = list(plan.subtopics)
    toolsets, tools = build_supervisor_surface(
        plan,
        ledger,
        uncovered_subtopics=uncovered_subtopics,
        tool_timeout_sec=config.tool_timeout_sec,
        allow_bash_tool=config.allow_supervisor_bash,
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
        "allow_supervisor_bash": config.allow_supervisor_bash,
    }
    if brief is not None:
        prompt["user_brief"] = brief
    if preferences is not None:
        prompt["preferences"] = preferences.model_dump(mode="json")
        prompt["guidance"] = build_supervisor_guidance(preferences)

    with span(
        "supervisor turn",
        iteration=iteration,
        allow_supervisor_bash=config.allow_supervisor_bash,
        enabled_provider_count=len(config.enabled_providers),
    ):
        serialized_prompt = serialize_prompt_payload(
            prompt, label="supervisor prompt payload"
        )
        sig = inspect.signature(agent.run_sync)
        supports_hooks = "hooks" in sig.parameters

        if supports_hooks:
            collector = ToolResultCollector()
            result = agent.run_sync(
                serialized_prompt,
                hooks={"after_tool_call": collector.hook},
            )
            raw_results = collector.results
            trace_stats = ToolTraceExtractionStats(
                trace_available=True,
                tool_return_part_count=collector.call_count,
                normalized_result_count=len(collector.results),
            )
        else:
            result = agent.run_sync(serialized_prompt)
            raw_results, trace_stats = extract_mcp_raw_results_with_stats(result)

    if trace_stats.warnings:
        warning(
            "Supervisor tool trace extraction degraded",
            iteration=iteration,
            trace_available=trace_stats.trace_available,
            message_count=trace_stats.message_count,
            tool_return_part_count=trace_stats.tool_return_part_count,
            normalized_result_count=trace_stats.normalized_result_count,
            dropped_part_count=trace_stats.dropped_part_count,
            warnings=trace_stats.warnings,
        )

    return SupervisorCheckpointResult(
        decision=result.output,
        raw_results=raw_results,
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
