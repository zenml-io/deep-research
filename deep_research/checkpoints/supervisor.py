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
from deep_research.evidence.ledger import select_new_this_iteration, truncate_ledger_for_context
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


def _pydantic_ai_version() -> str:
    try:
        import pydantic_ai
        return pydantic_ai.__version__ or "unknown"
    except (ImportError, AttributeError):
        return "unknown"


def _build_trace_warning_result(
    *,
    iteration: int,
    trace_stats: ToolTraceExtractionStats,
    collector_call_count: int | None = None,
) -> RawToolResult:
    warning_codes = list(trace_stats.warnings)
    if collector_call_count is not None and collector_call_count != trace_stats.tool_return_part_count:
        warning_codes.append("hook_capture_mismatch")
    warning_codes = sorted(dict.fromkeys(warning_codes))
    warning_message = (
        f"iteration={iteration} "
        f"dropped_part_count={trace_stats.dropped_part_count} "
        f"warning_codes={','.join(warning_codes) if warning_codes else 'none'} "
        f"pydantic_ai_version={_pydantic_ai_version()}"
    )
    return RawToolResult(
        tool_name="supervisor_trace_warning",
        provider="supervisor",
        payload={
            "iteration": iteration,
            "trace_available": trace_stats.trace_available,
            "tool_return_part_count": trace_stats.tool_return_part_count,
            "normalized_result_count": trace_stats.normalized_result_count,
            "dropped_part_count": trace_stats.dropped_part_count,
            "warning_codes": warning_codes,
            "pydantic_ai_version": _pydantic_ai_version(),
            "message": warning_message,
        },
        ok=False,
        error=warning_message,
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
    unanswered_questions: list[str] | None = None,
    brief: str | None = None,
    preferences: ResearchPreferences | None = None,
) -> SupervisorCheckpointResult:
    """
    Execute one supervisor search turn: build context, run the agent, collect tool
    results, emit a degradation warning if the trace is incomplete, and return a
    SupervisorCheckpointResult.
    """
    from deep_research.agents.supervisor import build_supervisor_agent

    bootstrap_logfire()

    if uncovered_subtopics is None:
        uncovered_subtopics = list(plan.subtopics)
    if unanswered_questions is None:
        unanswered_questions = list(plan.key_questions)

    context_ledger = truncate_ledger_for_context(
        ledger,
        max_chars=config.supervisor_context_budget_chars,
        role="supervisor",
        snippet_budget_chars=config.context_snippet_budget_chars,
        current_iteration=iteration,
    )
    toolsets, tools = build_supervisor_surface(
        plan,
        context_ledger,
        uncovered_subtopics=uncovered_subtopics,
        tool_timeout_sec=config.tool_timeout_sec,
        allow_bash_tool=config.allow_supervisor_bash,
    )
    agent = build_supervisor_agent(
        config.supervisor_model,
        toolsets=toolsets,
        tools=tools,
    )

    # --- build prompt payload ---
    prompt: dict[str, object] = {
        "plan": plan.model_dump(mode="json"),
        "ledger": context_ledger.model_dump(mode="json"),
        "new_this_iteration": [
            candidate.model_dump(mode="json")
            for candidate in select_new_this_iteration(context_ledger, iteration)
        ],
        "uncovered_subtopics": uncovered_subtopics,
        "unanswered_questions": unanswered_questions,
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
        prompt["guidance"] = (
            build_supervisor_guidance(preferences)
            + " Prioritize search actions that close unanswered key questions first."
        )

    # --- run agent ---
    with span(
        "supervisor turn",
        iteration=iteration,
        allow_supervisor_bash=config.allow_supervisor_bash,
        enabled_provider_count=len(config.enabled_providers),
    ):
        serialized_prompt = serialize_prompt_payload(prompt, label="supervisor prompt payload")
        supports_hooks = "hooks" in inspect.signature(agent.run_sync).parameters

        if supports_hooks:
            collector = ToolResultCollector()
            result = agent.run_sync(  # type: ignore[union-attr]
                serialized_prompt,
                hooks={"after_tool_call": collector.hook},
            )
            extracted_results, trace_stats = extract_mcp_raw_results_with_stats(result)
            raw_results = collector.results
            if (
                trace_stats.trace_available
                and collector.call_count != trace_stats.tool_return_part_count
                and extracted_results
            ):
                # Why: fall back to trace extraction when hook count diverges from trace count
                raw_results = extracted_results
        else:
            collector = None
            result = agent.run_sync(serialized_prompt)  # type: ignore[union-attr]
            raw_results, trace_stats = extract_mcp_raw_results_with_stats(result)

    # --- detect trace issues ---
    hook_capture_mismatch = (
        supports_hooks
        and collector is not None
        and trace_stats.trace_available
        and collector.call_count != trace_stats.tool_return_part_count
    )
    collector_call_count = collector.call_count if collector is not None else None
    has_trace_issues = bool(trace_stats.warnings) or trace_stats.dropped_part_count > 0 or hook_capture_mismatch
    if has_trace_issues:
        # Why: build a new list to preserve immutability of the input
        raw_results = [*raw_results, _build_trace_warning_result(
            iteration=iteration, trace_stats=trace_stats, collector_call_count=collector_call_count,
        )]
        warning(
            "Supervisor tool trace extraction degraded",
            iteration=iteration,
            trace_available=trace_stats.trace_available,
            message_count=trace_stats.message_count,
            tool_return_part_count=trace_stats.tool_return_part_count,
            normalized_result_count=trace_stats.normalized_result_count,
            dropped_part_count=trace_stats.dropped_part_count,
            warnings=trace_stats.warnings,
            hook_capture_mismatch=hook_capture_mismatch,
            pydantic_ai_version=_pydantic_ai_version(),
        )

    return SupervisorCheckpointResult(
        decision=result.output,
        raw_results=raw_results,
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.supervisor_pricing),
        ),
        warnings=list(trace_stats.warnings),
    )


@checkpoint(type="llm_call")
def run_supervisor(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    config: ResearchConfig,
    uncovered_subtopics: list[str] | None = None,
    unanswered_questions: list[str] | None = None,
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
        unanswered_questions,
        brief=brief,
        preferences=preferences,
    )
