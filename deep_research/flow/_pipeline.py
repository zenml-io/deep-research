"""Private helpers for ``research_flow``.

These functions are split out of ``research_flow.py`` so that module can stay a
short, readable orchestration script. They are intentionally private (``_``
prefix at module level) because their signatures are tailored to the flow and
are not a public API surface.

Checkpoint lookups go through ``deep_research.flow.research_flow`` so existing
tests that monkeypatch checkpoints on that module continue to work unchanged.
The lazy ``_flow`` accessor avoids a circular import at module load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from types import ModuleType
from typing import NamedTuple

from deep_research.config import ResearchConfig
from deep_research.enums import DeliverableMode, StopReason, Tier
from deep_research.flow.convergence import detect_source_diversity_warning
from deep_research.models import (
    CoherenceResult,
    CoverageScore,
    CritiqueResult,
    EvidenceLedger,
    GroundingResult,
    InvestigationPackage,
    IterationRecord,
    IterationTrace,
    RawToolResult,
    RenderPayload,
    RenderSettingsSnapshot,
    RequestClassification,
    ResearchPlan,
    ResearchPreferences,
    RunSummary,
    SearchExecutionResult,
    SelectionGraph,
    SupervisorCheckpointResult,
    ToolCallRecord,
)
from deep_research.observability import metric, span


APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"

# Deliverable modes the writer fully supports; other modes still render but
# surface a degradation note in the final package.
_FULLY_SUPPORTED_MODES: frozenset[DeliverableMode] = frozenset(
    {DeliverableMode.RESEARCH_PACKAGE, DeliverableMode.FINAL_REPORT}
)


def _flow() -> ModuleType:
    """Return the live ``research_flow`` module for attribute lookup.

    Using an attribute lookup (rather than a direct import) means pytest
    monkeypatches applied to ``deep_research.flow.research_flow.<name>`` are
    honoured by these helpers without re-importing anything.
    """
    from deep_research.flow import research_flow

    return research_flow


@dataclass(frozen=True)
class RunState:
    """Resolved inputs the iteration loop + downstream helpers consume."""

    brief: str
    config: ResearchConfig
    classification: RequestClassification
    preferences: ResearchPreferences


class IterationLoopOutput(NamedTuple):
    ledger: EvidenceLedger
    iteration_history: tuple[IterationRecord, ...]
    provider_usage_summary: dict[str, int]
    spent_usd: float
    stop_reason: StopReason
    council_models: list[str]


class CritiqueBundle(NamedTuple):
    renders: list[RenderPayload]
    critique_result: CritiqueResult | None
    spent_usd: float


class JudgeBundle(NamedTuple):
    grounding_result: GroundingResult | None
    coherence_result: CoherenceResult | None
    spent_usd: float


def merge_provider_counts(
    existing: dict[str, int],
    raw_results: list[RawToolResult],
) -> dict[str, int]:
    """Return a new dict with provider counts updated from ``raw_results``."""
    updated = dict(existing)
    for raw_result in raw_results:
        updated[raw_result.provider] = updated.get(raw_result.provider, 0) + 1
    return updated


def build_tool_call_records(
    raw_results: list[RawToolResult],
) -> list[ToolCallRecord]:
    """Build immutable tool-call records from raw results."""
    records: list[ToolCallRecord] = []
    for raw_result in raw_results:
        provider_label = f" via {raw_result.provider}" if raw_result.provider else ""
        if raw_result.ok:
            summary = f"{raw_result.tool_name}{provider_label} succeeded"
            status = "ok"
        else:
            detail = f": {raw_result.error}" if raw_result.error else ""
            summary = f"{raw_result.tool_name}{provider_label} failed{detail}"
            status = "error"
        records.append(
            ToolCallRecord(
                tool_name=raw_result.tool_name,
                status=status,
                provider=raw_result.provider,
                summary=summary,
            )
        )
    return records


def resolve_config_for_tier(
    user_config: ResearchConfig | None,
    resolved_tier: Tier,
) -> ResearchConfig:
    """Return a config for ``resolved_tier``, layering user overrides on top.

    Uses ``exclude_unset=True`` so Pydantic defaults on the user's config do
    NOT clobber the tier-specific defaults. Always returns a fresh frozen
    ``ResearchConfig``.
    """
    tier_config = ResearchConfig.for_tier(resolved_tier)
    if user_config is None:
        return tier_config
    if user_config.tier == resolved_tier:
        return user_config
    overrides = user_config.model_dump(exclude_unset=True)
    overrides["tier"] = resolved_tier
    return tier_config.model_copy(update=overrides)


def _classify_once(
    brief: str,
    tier: str,
    user_config: ResearchConfig | None,
) -> tuple[RequestClassification, ResearchConfig]:
    """Run classification once and fold the result into a resolved config."""
    flow = _flow()
    classification = flow.classify_request.submit(brief, user_config).load()
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    return classification, resolve_config_for_tier(user_config, resolved_tier)


def resolve_config_and_classify(
    brief: str,
    tier: str,
    user_config: ResearchConfig | None,
) -> RunState:
    """Return a ``RunState``, handling clarification inline when needed."""
    classification, resolved_config = _classify_once(brief, tier, user_config)
    current_brief = brief
    if classification.needs_clarification and classification.clarification_question:
        current_brief = _flow().wait(
            name=CLARIFY_BRIEF_WAIT_NAME,
            schema=str,
            question=classification.clarification_question,
        )
        classification, resolved_config = _classify_once(
            current_brief, tier, user_config
        )
    return RunState(
        brief=current_brief,
        config=resolved_config,
        classification=classification,
        preferences=classification.preferences,
    )


def await_plan_approval_if_required(
    plan: ResearchPlan,
    config: ResearchConfig,
) -> None:
    """Block on a human approval wait when the tier requires one."""
    if not config.require_plan_approval:
        return
    approved = _flow().wait(
        name=APPROVE_PLAN_WAIT_NAME,
        schema=bool,
        question=f"Approve plan for: {plan.goal}?",
    )
    if approved is False:
        raise ValueError("plan not approved")


def _run_supervisor_turn(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    run_state: RunState,
    uncovered_subtopics: list[str] | None,
    unanswered_questions: list[str] | None,
    council_models: list[str],
) -> SupervisorCheckpointResult:
    """Dispatch to the council or single-supervisor branch.

    Uses stable per-iteration checkpoint ids so operators get a replay anchor
    on each turn.
    """
    flow = _flow()
    config = run_state.config

    def _normalize(result: object) -> SupervisorCheckpointResult:
        if isinstance(result, SupervisorCheckpointResult):
            return result
        budget_obj = getattr(result, "budget", None)
        if budget_obj is None:
            budget = getattr(
                SupervisorCheckpointResult.model_fields["budget"],
                "default_factory",
                lambda: None,
            )()
        else:
            budget = budget_obj
            if not hasattr(budget_obj, "model_dump"):
                budget = {
                    "input_tokens": getattr(budget_obj, "input_tokens", 0),
                    "output_tokens": getattr(budget_obj, "output_tokens", 0),
                    "total_tokens": getattr(
                        budget_obj,
                        "total_tokens",
                        getattr(budget_obj, "input_tokens", 0)
                        + getattr(budget_obj, "output_tokens", 0),
                    ),
                    "estimated_cost_usd": getattr(
                        budget_obj, "estimated_cost_usd", 0.0
                    ),
                }
        raw_results = []
        for raw_result in getattr(result, "raw_results", []) or []:
            raw_results.append(
                RawToolResult(
                    tool_name=getattr(raw_result, "tool_name", "tool"),
                    provider=getattr(raw_result, "provider", "unknown"),
                    payload=getattr(raw_result, "payload", {}),
                    ok=getattr(raw_result, "ok", True),
                    error=getattr(raw_result, "error", None),
                )
            )
        return SupervisorCheckpointResult(
            decision=getattr(
                result,
                "decision",
                getattr(
                    result,
                    "output",
                    None,
                )
                or getattr(
                    SupervisorCheckpointResult.model_fields["decision"],
                    "default_factory",
                    lambda: None,
                )(),
            ),
            raw_results=raw_results,
            budget=budget,
            warnings=list(getattr(result, "warnings", []) or []),
        )

    if config.council_mode:
        futures = [
            flow.run_council_generator.submit(
                plan,
                ledger,
                iteration,
                model_name,
                config,
                uncovered_subtopics,
                unanswered_questions=unanswered_questions,
                brief=run_state.brief,
                preferences=run_state.preferences,
                id=f"council_{iteration}_{index}",
            )
            for index, model_name in enumerate(council_models)
        ]
        return _normalize(
            flow.aggregate_council_results.submit(
                [f.load() for f in futures],
            ).load()
        )
    return _normalize(
        flow.run_supervisor.submit(
            plan,
            ledger,
            iteration,
            config,
            uncovered_subtopics,
            unanswered_questions=unanswered_questions,
            brief=run_state.brief,
            preferences=run_state.preferences,
            id=f"supervisor_{iteration}",
        ).load()
    )


def _continue_reason(
    should_stop: bool,
    uncovered_subtopics: list[str],
    unanswered_questions: list[str],
    coverage_delta: float,
) -> str | None:
    """Explain why the loop will keep running (or ``None`` when stopping)."""
    if should_stop:
        return None
    if unanswered_questions:
        return "remaining unanswered questions: " + ", ".join(unanswered_questions)
    if uncovered_subtopics:
        return "remaining uncovered subtopics: " + ", ".join(uncovered_subtopics)
    return f"coverage improved by {coverage_delta:.2f}; continue exploring"


def run_iteration_loop(
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> IterationLoopOutput:
    """Run the supervisor/search/score/converge loop until a stop condition hits.

    Wall-clock observation is delegated to ``snapshot_wall_clock`` so replay
    sees the same elapsed values the original run observed.
    """
    flow = _flow()
    config = run_state.config
    ledger = EvidenceLedger()
    iteration_history: tuple[IterationRecord, ...] = ()
    provider_usage_summary: dict[str, int] = {}
    spent_usd = 0.0
    uncovered_subtopics: list[str] | None = None
    unanswered_questions: list[str] | None = None
    council_models = (
        [config.supervisor_model] * config.council_size if config.council_mode else []
    )
    stop_reason = StopReason.MAX_ITERATIONS
    _replans_used = 0

    with span("iteration_loop", max_iterations=config.max_iterations):
        for iteration in range(config.max_iterations):
            with span("iteration", iteration=iteration):
                supervisor_started = perf_counter()
                supervisor_result = _run_supervisor_turn(
                    plan,
                    ledger,
                    iteration,
                    run_state,
                    uncovered_subtopics,
                    unanswered_questions,
                    council_models,
                )
                supervisor_latency_ms = int((perf_counter() - supervisor_started) * 1000)
                decision = supervisor_result.decision

                # Supervisor can signal completion directly via status field.
                if decision.status == "complete":
                    tool_calls = build_tool_call_records(supervisor_result.raw_results)
                    iteration_record = IterationRecord(
                        iteration=iteration,
                        new_candidate_count=0,
                        accepted_candidate_count=len(ledger.selected),
                        rejected_candidate_count=len(ledger.rejected),
                        coverage=iteration_history[-1].coverage
                        if iteration_history
                        else 0.0,
                        coverage_delta=0.0,
                        uncovered_subtopics=uncovered_subtopics or [],
                        unanswered_questions=unanswered_questions or [],
                        estimated_cost_usd=supervisor_result.budget.estimated_cost_usd,
                        tool_calls=tool_calls,
                        warnings=list(supervisor_result.warnings),
                        stop_reason=StopReason.SUPERVISOR_COMPLETE,
                    )
                    iteration_history = (*iteration_history, iteration_record)
                    stop_reason = StopReason.SUPERVISOR_COMPLETE
                    break

                if decision.search_actions:
                    search_started = perf_counter()
                    search_result = flow.execute_searches.submit(
                        decision, config, preferences=run_state.preferences
                    ).load()
                    search_latency_ms = int((perf_counter() - search_started) * 1000)
                else:
                    search_result = SearchExecutionResult()
                    search_latency_ms = 0
                search_warnings = getattr(search_result, "warnings", [])

                combined_raw_results = [
                    *supervisor_result.raw_results,
                    *search_result.raw_results,
                ]
                supervisor_cost = supervisor_result.budget.estimated_cost_usd
                search_cost = search_result.budget.estimated_cost_usd
                spent_usd = spent_usd + supervisor_cost + search_cost
                provider_usage_summary = merge_provider_counts(
                    provider_usage_summary, combined_raw_results
                )

                candidates = flow.extract_candidates.submit(combined_raw_results).load()
                relevance_started = perf_counter()
                relevance_result = flow.score_relevance.submit(
                    candidates, plan, config
                ).load()
                relevance_latency_ms = int((perf_counter() - relevance_started) * 1000)
                relevance_cost = relevance_result.budget.estimated_cost_usd
                iteration_cost = supervisor_cost + search_cost + relevance_cost
                spent_usd = spent_usd + relevance_cost

                scored_candidates = [
                    candidate
                    if getattr(candidate, "iteration_added", None) is not None
                    or not hasattr(candidate, "model_copy")
                    else candidate.model_copy(update={"iteration_added": iteration})
                    for candidate in relevance_result.candidates
                ]
                ledger = flow.update_ledger.submit(
                    scored_candidates,
                    ledger,
                    config=config,
                ).load()
                ledger = flow.enrich_candidates.submit(ledger, config).load()
                coverage_started = perf_counter()
                coverage = flow.score_coverage.submit(ledger, plan, config).load()
                coverage_latency_ms = int((perf_counter() - coverage_started) * 1000)
                if not isinstance(coverage, CoverageScore):
                    coverage = CoverageScore(
                        subtopic_coverage=getattr(coverage, "subtopic_coverage", coverage.total),
                        plan_fidelity=getattr(coverage, "plan_fidelity", coverage.total),
                        source_diversity=getattr(coverage, "source_diversity", coverage.total),
                        evidence_density=getattr(coverage, "evidence_density", coverage.total),
                        total=getattr(coverage, "total", 0.0),
                        uncovered_subtopics=list(
                            getattr(coverage, "uncovered_subtopics", [])
                        ),
                        unanswered_questions=list(
                            getattr(coverage, "unanswered_questions", [])
                        ),
                    )
                previous_coverage = (
                    iteration_history[-1].coverage if iteration_history else 0.0
                )
                coverage_delta = round(coverage.total - previous_coverage, 6)
                uncovered_subtopics = list(coverage.uncovered_subtopics)
                unanswered_questions = list(coverage.unanswered_questions)
                metric("iteration_coverage", coverage.total, iteration=iteration)
                metric("iteration_cost_usd", iteration_cost, iteration=iteration)
                metric("iteration_candidates", len(candidates), iteration=iteration)
                tool_calls = build_tool_call_records(combined_raw_results)
                warnings = [*supervisor_result.warnings, *search_warnings]
                diversity_warning = detect_source_diversity_warning(ledger)
                if diversity_warning is not None:
                    warnings.append(diversity_warning)
                ledger_payload = (
                    ledger.model_dump(mode="json")
                    if hasattr(ledger, "model_dump")
                    else {
                        "selected": [vars(item) for item in getattr(ledger, "selected", [])],
                        "rejected": [vars(item) for item in getattr(ledger, "rejected", [])],
                    }
                )
                context_budget_used_ratio = round(
                    len(json.dumps(ledger_payload, allow_nan=False))
                    / config.supervisor_context_budget_chars,
                    4,
                )
                iteration_record = IterationRecord(
                    iteration=iteration,
                    new_candidate_count=len(candidates),
                    accepted_candidate_count=len(ledger.selected),
                    rejected_candidate_count=len(ledger.rejected),
                    coverage=coverage.total,
                    coverage_delta=coverage_delta,
                    uncovered_subtopics=uncovered_subtopics,
                    unanswered_questions=unanswered_questions,
                    estimated_cost_usd=iteration_cost,
                    tool_calls=tool_calls,
                    warnings=warnings,
                    context_budget_used_ratio=context_budget_used_ratio,
                    step_costs_usd={
                        "supervisor": supervisor_cost,
                        "search": search_cost,
                        "relevance": relevance_cost,
                        "coverage": 0.0,
                    },
                    step_latencies_ms={
                        "supervisor": supervisor_latency_ms,
                        "search": search_latency_ms,
                        "relevance": relevance_latency_ms,
                        "coverage": coverage_latency_ms,
                    },
                )

                # Wall-clock observation lives behind a checkpoint so replays stay
                # deterministic; the decision itself is pure given its inputs.
                wall_clock = flow.snapshot_wall_clock.submit(stamp.started_at).load()
                decision_stop = flow.check_convergence(
                    coverage,
                    list(iteration_history),
                    spent_usd=spent_usd,
                    elapsed_seconds=wall_clock.elapsed_seconds,
                    max_iterations=config.max_iterations,
                    epsilon=config.convergence_epsilon,
                    min_coverage=config.convergence_min_coverage,
                    budget_limit_usd=config.cost_budget_usd,
                    time_limit_seconds=config.time_box_seconds,
                    new_candidate_count=iteration_record.new_candidate_count,
                )
                continue_reason = _continue_reason(
                    decision_stop.should_stop,
                    uncovered_subtopics,
                    unanswered_questions,
                    coverage_delta,
                )
                iteration_record = iteration_record.model_copy(
                    update={
                        "continue_reason": continue_reason,
                        "stop_reason": (
                            decision_stop.reason if decision_stop.should_stop else None
                        ),
                    }
                )
                iteration_history = (*iteration_history, iteration_record)
                log_kwargs = {
                    "iteration": iteration,
                    "coverage": coverage.total,
                    "coverage_delta": coverage_delta,
                    "uncovered_subtopics": uncovered_subtopics,
                    "new_candidate_count": iteration_record.new_candidate_count,
                    "accepted_candidate_count": iteration_record.accepted_candidate_count,
                    "rejected_candidate_count": iteration_record.rejected_candidate_count,
                    "tool_summaries": [tc.summary for tc in tool_calls],
                    "stop_reason": iteration_record.stop_reason,
                    "continue_reason": continue_reason,
                    "spent_usd": round(spent_usd, 6),
                }
                if unanswered_questions:
                    log_kwargs["unanswered_questions"] = unanswered_questions
                if warnings:
                    log_kwargs["warnings"] = warnings
                flow.log(**log_kwargs)
                if decision_stop.should_stop:
                    # Check replan gate before accepting the stop
                    if (
                        config.max_replans > 0
                        and decision_stop.reason
                        in (StopReason.LOOP_STALL, StopReason.DIMINISHING_RETURNS)
                        and _replans_used < config.max_replans
                    ):
                        replan_decision = flow.evaluate_replan.submit(
                            plan, coverage, list(iteration_history), config
                        ).load()
                        if replan_decision.should_replan:
                            updates: dict[str, object] = {}
                            if replan_decision.updated_subtopics:
                                updates["subtopics"] = replan_decision.updated_subtopics
                            if replan_decision.updated_queries:
                                updates["queries"] = replan_decision.updated_queries
                            if updates:
                                plan = plan.model_copy(update=updates)
                            uncovered_subtopics = list(coverage.uncovered_subtopics)
                            unanswered_questions = list(coverage.unanswered_questions)
                            _replans_used += 1
                            flow.log(
                                replan_triggered=True,
                                replan_number=_replans_used,
                                rationale=replan_decision.rationale,
                                updated_subtopics=bool(
                                    replan_decision.updated_subtopics
                                ),
                                updated_queries=bool(replan_decision.updated_queries),
                                iteration=iteration,
                            )
                            continue
                        else:
                            flow.log(
                                replan_triggered=False,
                                rationale=replan_decision.rationale,
                                iteration=iteration,
                            )
                    # Fall back to MAX_ITERATIONS if the decision didn't name a reason.
                    stop_reason = decision_stop.reason or StopReason.MAX_ITERATIONS
                    break

    return IterationLoopOutput(
        ledger=ledger,
        iteration_history=iteration_history,
        provider_usage_summary=provider_usage_summary,
        spent_usd=spent_usd,
        stop_reason=stop_reason,
        council_models=council_models,
    )


def render_deliverable(
    selection: SelectionGraph,
    iteration_output: IterationLoopOutput,
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> tuple[list[RenderPayload], float, list[str]]:
    """Render deliverable(s) and return ``(renders, added_cost, degradations)``."""
    flow = _flow()
    config = run_state.config
    deliverable_mode = run_state.preferences.deliverable_mode
    degradations: list[str] = []

    if deliverable_mode == DeliverableMode.RESEARCH_PACKAGE:
        reading_future = flow.write_reading_path.submit(
            selection,
            iteration_output.ledger,
            plan,
            config,
            preferences=run_state.preferences,
        )
        backing_future = flow.write_backing_report.submit(
            selection,
            iteration_output.ledger,
            plan,
            IterationTrace(iterations=list(iteration_output.iteration_history)),
            iteration_output.provider_usage_summary,
            iteration_output.stop_reason,
            config,
            preferences=run_state.preferences,
        )
        reading_result = reading_future.load()
        backing_result = backing_future.load()
        added_cost = (
            reading_result.budget.estimated_cost_usd
            + backing_result.budget.estimated_cost_usd
        )
        return [reading_result.render, backing_result.render], added_cost, degradations

    # Non-default modes fall through to a single full-report render. Reuse the
    # run stamp so the scaffold RunSummary shares run_id with the final one.
    partial_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="rendering",
        estimated_cost_usd=round(iteration_output.spent_usd, 6),
        elapsed_seconds=0,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=None,
    )
    partial_package = InvestigationPackage(
        run_summary=partial_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=[],
        preferences=run_state.preferences,
    )
    full_result = flow.write_full_report.submit(
        partial_package, config, preferences=run_state.preferences
    ).load()
    if deliverable_mode not in _FULLY_SUPPORTED_MODES:
        degradations.append(
            f"Deliverable mode '{deliverable_mode.value}' not yet fully supported"
            f" -- rendered as 'final_report' with {deliverable_mode.value} context"
        )
    return [full_result.render], full_result.budget.estimated_cost_usd, degradations


def run_critique_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> CritiqueBundle:
    """Run reviewer and apply revisions when critique is enabled."""
    flow = _flow()
    if not config.critique_enabled:
        return CritiqueBundle(renders=renders, critique_result=None, spent_usd=0.0)
    critique_checkpoint = flow.critique_reports.submit(
        renders, plan, selection, ledger, config
    ).load()
    critique_result = critique_checkpoint.critique
    revision_checkpoint = flow.apply_revisions.submit(
        renders, critique_result, plan, config
    ).load()
    return CritiqueBundle(
        renders=revision_checkpoint.renders,
        critique_result=critique_result,
        spent_usd=(
            critique_checkpoint.budget.estimated_cost_usd
            + revision_checkpoint.budget.estimated_cost_usd
        ),
    )


def run_judges_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> JudgeBundle:
    """Run grounding + coherence judges in parallel when enabled."""
    flow = _flow()
    if not config.judge_enabled:
        return JudgeBundle(grounding_result=None, coherence_result=None, spent_usd=0.0)
    grounding_future = flow.verify_grounding.submit(renders, ledger, config)
    coherence_future = flow.verify_coherence.submit(renders, plan, config)
    grounding_checkpoint = grounding_future.load()
    coherence_checkpoint = coherence_future.load()
    return JudgeBundle(
        grounding_result=grounding_checkpoint.grounding,
        coherence_result=coherence_checkpoint.coherence,
        spent_usd=(
            grounding_checkpoint.budget.estimated_cost_usd
            + coherence_checkpoint.budget.estimated_cost_usd
        ),
    )


def assemble_final_package(
    *,
    stamp,
    finalization,
    run_state: RunState,
    plan: ResearchPlan,
    iteration_output: IterationLoopOutput,
    selection: SelectionGraph,
    renders: list[RenderPayload],
    spent_usd: float,
    critique_result: CritiqueResult | None,
    grounding_result: GroundingResult | None,
    coherence_result: CoherenceResult | None,
    degradations: list[str],
) -> InvestigationPackage:
    """Build the final RunSummary and hand everything to the assemble checkpoint."""
    flow = _flow()
    config = run_state.config
    run_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="completed",
        estimated_cost_usd=round(spent_usd, 6),
        elapsed_seconds=finalization.elapsed_seconds,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=finalization.completed_at,
    )
    return flow.assemble_package.submit(
        run_summary=run_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=renders,
        render_settings=RenderSettingsSnapshot(writer_model=config.writer_model),
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
        preferences=run_state.preferences,
        preference_degradations=degradations,
    ).load()
