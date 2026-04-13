"""Iteration loop orchestration: convergence, replan, and the main loop."""

from __future__ import annotations

import json

from deep_research.config import ResearchConfig
from deep_research.enums import StopReason
from deep_research.flow._types import (
    IterationLoopOutput,
    RunState,
    _flow,
    merge_provider_counts,
)
from deep_research.flow.convergence import ConvergenceSignal, detect_source_diversity_warning
from deep_research.flow.waves import (
    _continue_reason,
    _wave_enrich,
    _wave_feedback,
    _wave_score,
    _wave_search,
    _wave_triage,
)
from deep_research.models import (
    EvidenceLedger,
    IterationRecord,
    ResearchPlan,
)
from deep_research.observability import metric, span


def run_iteration_loop(
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> IterationLoopOutput:
    """Run research waves until a composite stop condition hits or max_iterations is reached."""
    config = run_state.config
    convergence_cfg = config.convergence
    council_models = (
        [config.supervisor_model] * config.council_size if config.council_mode else []
    )
    flow = _flow()

    # --- loop state ---
    ledger: EvidenceLedger = EvidenceLedger()
    iteration_history: tuple[IterationRecord, ...] = ()
    provider_usage_summary: dict[str, int] = {}
    spent_usd: float = 0.0
    total_tokens: int = 0
    active_elapsed_ms: int = 0
    wall_elapsed_seconds: int = 0
    uncovered_subtopics: list[str] | None = None
    unanswered_questions: list[str] | None = None
    carryover_actions: list = []
    stall_count: int = 0
    feedback_iteration_count: int = 0
    replans_used: int = 0
    stop_reason: StopReason = StopReason.MAX_ITERATIONS

    with span("iteration_loop", max_iterations=config.max_iterations):
        for iteration in range(config.max_iterations):
            with span("iteration", iteration=iteration):
                wave_search = _wave_search(
                    plan=plan,
                    ledger=ledger,
                    iteration=iteration,
                    run_state=run_state,
                    uncovered_subtopics=uncovered_subtopics,
                    unanswered_questions=unanswered_questions,
                    council_models=council_models,
                    carryover_actions=carryover_actions,
                )

                if wave_search.decision.status == "complete":
                    # Why: supervisor signals research is complete — skip all downstream waves
                    # and record the early stop before breaking.
                    active_elapsed_ms += sum(wave_search.step_latencies_ms.values())
                    wall_elapsed_seconds = flow.snapshot_wall_clock.submit(
                        stamp.started_at,
                    ).load().wall_elapsed_seconds
                    complete_record = IterationRecord(
                        iteration=iteration,
                        new_candidate_count=0,
                        accepted_candidate_count=len(ledger.selected),
                        rejected_candidate_count=len(ledger.rejected),
                        coverage=iteration_history[-1].coverage if iteration_history else 0.0,
                        coverage_delta=0.0,
                        uncovered_subtopics=uncovered_subtopics or [],
                        unanswered_questions=unanswered_questions or [],
                        estimated_cost_usd=wave_search.supervisor_cost,
                        tool_calls=wave_search.tool_calls,
                        warnings=list(wave_search.warnings),
                        stop_reason=StopReason.SUPERVISOR_COMPLETE,
                        step_costs_usd={
                            "supervisor": wave_search.supervisor_cost,
                            "search": wave_search.search_cost,
                        },
                        step_latencies_ms=wave_search.step_latencies_ms,
                    )
                    iteration_history = (*iteration_history, complete_record)
                    stop_reason = StopReason.SUPERVISOR_COMPLETE
                    break

                provider_usage_summary = merge_provider_counts(
                    provider_usage_summary, wave_search.raw_results
                )
                previous_ledger_size = len(ledger.selected) + len(ledger.rejected)
                wave_triage = _wave_triage(wave_search.raw_results, plan, config, iteration)
                spent_usd = (
                    spent_usd
                    + wave_search.supervisor_cost
                    + wave_search.search_cost
                    + wave_triage.relevance_cost
                )
                total_tokens = total_tokens + wave_search.total_tokens + wave_triage.total_tokens
                wave_enrich = _wave_enrich(wave_triage.candidates, ledger, config)
                ledger = wave_enrich.ledger
                wave_score = _wave_score(ledger, plan, config)
                coverage = wave_score.coverage

                previous_coverage = (
                    iteration_history[-1].coverage if iteration_history else 0.0
                )
                coverage_delta = round(coverage.total - previous_coverage, 6)
                stall_count = stall_count + 1 if coverage_delta <= 0 else 0
                uncovered_subtopics = list(coverage.uncovered_subtopics)
                unanswered_questions = list(coverage.unanswered_questions)

                feedback = _wave_feedback(
                    coverage,
                    plan=plan,
                    config=config,
                    ledger=ledger,
                    feedback_iteration_count=feedback_iteration_count,
                )
                carryover_actions = feedback.carryover_actions
                feedback_iteration_count = feedback.feedback_iteration_count

                metric("iteration_coverage", coverage.total, iteration=iteration)
                metric(
                    "iteration_cost_usd",
                    wave_search.supervisor_cost + wave_search.search_cost + wave_triage.relevance_cost,
                    iteration=iteration,
                )
                metric(
                    "iteration_candidates",
                    len(wave_triage.candidates),
                    iteration=iteration,
                )

                warnings = list(wave_search.warnings)
                if feedback.reason:
                    warnings.append(feedback.reason)
                diversity_warning = detect_source_diversity_warning(ledger)
                if diversity_warning is not None:
                    warnings.append(diversity_warning)
                iteration_cost = (
                    wave_search.supervisor_cost
                    + wave_search.search_cost
                    + wave_triage.relevance_cost
                )
                context_budget_used_ratio = round(
                    len(json.dumps(ledger.model_dump(mode="json"), allow_nan=False))
                    / config.supervisor_context_budget_chars,
                    4,
                )
                iteration_record = IterationRecord(
                    iteration=iteration,
                    new_candidate_count=max(
                        0,
                        len(ledger.selected) + len(ledger.rejected) - previous_ledger_size,
                    ),
                    accepted_candidate_count=len(ledger.selected),
                    rejected_candidate_count=len(ledger.rejected),
                    coverage=coverage.total,
                    coverage_delta=coverage_delta,
                    uncovered_subtopics=uncovered_subtopics,
                    unanswered_questions=unanswered_questions,
                    estimated_cost_usd=iteration_cost,
                    tool_calls=wave_search.tool_calls,
                    warnings=warnings,
                    context_budget_used_ratio=context_budget_used_ratio,
                    step_costs_usd={
                        "supervisor": wave_search.supervisor_cost,
                        "search": wave_search.search_cost,
                        "relevance": wave_triage.relevance_cost,
                        "coverage": 0.0,
                    },
                    step_latencies_ms={
                        **wave_search.step_latencies_ms,
                        **wave_triage.step_latencies_ms,
                        **wave_enrich.step_latencies_ms,
                        **wave_score.step_latencies_ms,
                    },
                )
                active_elapsed_ms += sum(iteration_record.step_latencies_ms.values())

                # --- convergence signal ---
                wall_clock = flow.snapshot_wall_clock.submit(stamp.started_at).load()
                source_group_count = len(
                    {
                        candidate.source_group or candidate.source_kind.value
                        for candidate in ledger.selected
                    }
                )
                token_ratio = (
                    1.0
                    if convergence_cfg.token_budget <= 0
                    else max(0.0, 1.0 - (total_tokens / convergence_cfg.token_budget))
                )
                signal = ConvergenceSignal(
                    coverage_total=coverage.total,
                    subtopic_coverage=coverage.subtopic_coverage,
                    unanswered_questions_count=len(unanswered_questions),
                    considered_count=len(ledger.considered),
                    selected_count=len(ledger.selected),
                    source_group_count=source_group_count,
                    new_candidate_count=iteration_record.new_candidate_count,
                    marginal_info_gain=max(0.0, iteration_record.coverage_delta),
                    stall_count=stall_count,
                    spent_usd=spent_usd,
                    active_elapsed_seconds=wall_clock.active_elapsed_seconds,
                    wall_elapsed_seconds=wall_clock.wall_elapsed_seconds,
                    total_tokens=total_tokens,
                    token_budget_remaining_ratio=token_ratio,
                )
                decision_stop = flow.check_convergence(
                    signal,
                    list(iteration_history),
                    max_iterations=config.max_iterations,
                    coverage_threshold=convergence_cfg.coverage_threshold,
                    strong_coverage_threshold=convergence_cfg.strong_coverage_shortcut,
                    marginal_gain_threshold=convergence_cfg.marginal_gain_threshold,
                    max_stall_count=convergence_cfg.max_stall_count,
                    budget_limit_usd=config.cost_budget_usd,
                    active_time_limit_seconds=config.time_box_seconds,
                    min_considered_floor=convergence_cfg.min_considered_floor,
                    min_selected_for_stop=convergence_cfg.min_selected_for_stop,
                )
                continue_reason = _continue_reason(
                    decision_stop.should_stop,
                    uncovered_subtopics,
                    unanswered_questions,
                    coverage_delta,
                    decision_stop.continue_reason,
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

                # --- iteration log ---
                log_kwargs: dict[str, object] = {
                    "iteration": iteration,
                    "coverage": coverage.total,
                    "coverage_delta": coverage_delta,
                    "uncovered_subtopics": uncovered_subtopics,
                    "new_candidate_count": iteration_record.new_candidate_count,
                    "accepted_candidate_count": iteration_record.accepted_candidate_count,
                    "rejected_candidate_count": iteration_record.rejected_candidate_count,
                    "tool_summaries": [tc.summary for tc in iteration_record.tool_calls],
                    "stop_reason": iteration_record.stop_reason,
                    "continue_reason": continue_reason,
                    "spent_usd": round(spent_usd, 6),
                }
                if unanswered_questions:
                    log_kwargs["unanswered_questions"] = unanswered_questions
                if decision_stop.diagnostics:
                    log_kwargs["convergence"] = decision_stop.diagnostics
                flow.log(**log_kwargs)

                if not decision_stop.should_stop:
                    continue

                # --- replan gate ---
                replan_eligible = (
                    config.max_replans > 0
                    and decision_stop.reason in (StopReason.LOOP_STALL, StopReason.DIMINISHING_RETURNS)
                    and replans_used < config.max_replans
                )
                if replan_eligible:
                    replan_decision = flow.evaluate_replan.submit(
                        plan, coverage, list(iteration_history), config
                    ).load()
                    if replan_decision.should_replan:
                        updates: dict[str, object] = {}
                        if replan_decision.updated_subtopics:
                            updates["subtopics"] = replan_decision.updated_subtopics
                        if replan_decision.updated_queries:
                            updates["queries"] = replan_decision.updated_queries
                        plan = plan.model_copy(update=updates) if updates else plan
                        replans_used += 1
                        uncovered_subtopics = list(coverage.uncovered_subtopics)
                        unanswered_questions = list(coverage.unanswered_questions)
                        flow.log(
                            replan_triggered=True,
                            replan_number=replans_used,
                            rationale=replan_decision.rationale,
                            updated_subtopics=bool(replan_decision.updated_subtopics),
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

                stop_reason = decision_stop.reason or StopReason.MAX_ITERATIONS
                break

    return IterationLoopOutput(
        ledger=ledger,
        iteration_history=iteration_history,
        provider_usage_summary=provider_usage_summary,
        spent_usd=spent_usd,
        stop_reason=stop_reason,
        council_models=council_models,
        active_elapsed_seconds=active_elapsed_ms // 1000,
        wall_elapsed_seconds=wall_elapsed_seconds,
        total_tokens=total_tokens,
    )
