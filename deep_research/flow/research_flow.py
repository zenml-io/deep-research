from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4

from kitaru import flow, log, wait

from deep_research.checkpoints.classify import classify_request
from deep_research.checkpoints.council import (
    aggregate_council_results,
    run_council_generator,
)
from deep_research.checkpoints.assemble import assemble_package
from deep_research.checkpoints.coherence import judge_coherence
from deep_research.checkpoints.evaluate import evaluate_coverage
from deep_research.checkpoints.fetch import fetch_content
from deep_research.checkpoints.grounding import judge_grounding
from deep_research.checkpoints.merge import merge_evidence
from deep_research.checkpoints.normalize import normalize_evidence
from deep_research.checkpoints.plan import build_plan
from deep_research.checkpoints.review import review_renders
from deep_research.checkpoints.relevance import score_relevance
from deep_research.checkpoints.revise import revise_renders
from deep_research.checkpoints.search import execute_searches
from deep_research.checkpoints.select import build_selection_graph
from deep_research.checkpoints.supervisor import run_supervisor
from deep_research.config import ResearchConfig
from deep_research.enums import StopReason, Tier
from deep_research.flow.convergence import check_convergence
from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationRecord,
    IterationTrace,
    RenderSettingsSnapshot,
    RunSummary,
    SearchExecutionResult,
    SupervisorDecision,
    ToolCallRecord,
    CoherenceResult,
    CritiqueResult,
    GroundingResult,
)
from deep_research.checkpoints.rendering import (
    render_backing_report,
    render_reading_path,
)


CLASSIFY_CHECKPOINT_NAME = "classify_request"
PLAN_CHECKPOINT_NAME = "build_plan"
SUPERVISOR_CHECKPOINT_NAME = "run_supervisor"
COUNCIL_GENERATOR_CHECKPOINT_NAME = "run_council_generator"
APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"


@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    """Orchestrate the full research pipeline from brief to investigation package."""
    user_config = config
    classification = classify_request.submit(brief, config).load()
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    config = ResearchConfig.for_tier(resolved_tier)
    if user_config is not None:
        if user_config.tier == resolved_tier:
            config = user_config
        else:
            config = config.model_copy(
                update={**user_config.model_dump(), "tier": resolved_tier}
            )

    if classification.needs_clarification and classification.clarification_question:
        brief = wait(
            name=CLARIFY_BRIEF_WAIT_NAME,
            schema=str,
            question=classification.clarification_question,
        )
        classification = classify_request.submit(brief, config).load()
        resolved_tier = (
            classification.recommended_tier if tier == "auto" else Tier(tier)
        )
        config = ResearchConfig.for_tier(resolved_tier)
        if user_config is not None:
            if user_config.tier == resolved_tier:
                config = user_config
            else:
                config = config.model_copy(
                    update={**user_config.model_dump(), "tier": resolved_tier}
                )

    plan = build_plan.submit(brief, classification, config.tier).load()
    if config.require_plan_approval:
        approved = wait(
            name=APPROVE_PLAN_WAIT_NAME,
            schema=bool,
            question=f"Approve plan for: {plan.goal}?",
        )
        if approved is False:
            raise ValueError("plan not approved")

    ledger = EvidenceLedger()
    iteration_history: list[IterationRecord] = []
    provider_usage_summary: dict[str, int] = {}
    spent_usd = 0.0
    uncovered_subtopics: list[str] | None = None
    council_models = (
        [config.supervisor_model] * config.council_size if config.council_mode else []
    )
    stop_reason = StopReason.MAX_ITERATIONS
    start_time = monotonic()
    elapsed_seconds = 0
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for iteration in range(config.max_iterations):
        if config.council_mode:
            futures = [
                run_council_generator.submit(
                    plan,
                    ledger,
                    iteration,
                    model_name,
                    config,
                    uncovered_subtopics,
                    id=f"council_{iteration}_{index}",
                )
                for index, model_name in enumerate(council_models)
            ]
            supervisor_result = aggregate_council_results([f.load() for f in futures])
        else:
            supervisor_result = run_supervisor.submit(
                plan,
                ledger,
                iteration,
                config,
                uncovered_subtopics,
            ).load()
        decision = getattr(supervisor_result, "decision", None)
        if decision is None:
            decision = SupervisorDecision(rationale="", search_actions=[])
        if decision.search_actions:
            search_result = execute_searches.submit(
                decision,
                config,
            ).load()
        else:
            search_result = SearchExecutionResult()
        combined_raw_results = [
            *supervisor_result.raw_results,
            *search_result.raw_results,
        ]
        supervisor_cost = supervisor_result.budget.estimated_cost_usd
        search_cost = search_result.budget.estimated_cost_usd
        spent_usd += supervisor_cost + search_cost
        for raw_result in combined_raw_results:
            provider = raw_result.provider
            provider_usage_summary[provider] = (
                provider_usage_summary.get(provider, 0) + 1
            )

        candidates = normalize_evidence.submit(
            combined_raw_results,
        ).load()
        relevance_result = score_relevance.submit(
            candidates,
            plan,
            config,
        ).load()
        relevance_cost = relevance_result.budget.estimated_cost_usd
        iteration_cost = supervisor_cost + search_cost + relevance_cost
        spent_usd += relevance_cost

        ledger = merge_evidence.submit(
            relevance_result.candidates,
            ledger,
            config=config,
        ).load()
        ledger = fetch_content.submit(ledger, config).load()
        coverage = evaluate_coverage.submit(ledger, plan).load()
        previous_coverage = iteration_history[-1].coverage if iteration_history else 0.0
        coverage_delta = round(coverage.total - previous_coverage, 6)
        uncovered_subtopics = list(coverage.uncovered_subtopics)
        tool_calls: list[ToolCallRecord] = []
        for raw_result in combined_raw_results:
            provider_label = (
                f" via {raw_result.provider}" if raw_result.provider else ""
            )
            if raw_result.ok:
                summary = f"{raw_result.tool_name}{provider_label} succeeded"
                status = "ok"
            else:
                detail = f": {raw_result.error}" if raw_result.error else ""
                summary = f"{raw_result.tool_name}{provider_label} failed{detail}"
                status = "error"
            tool_calls.append(
                ToolCallRecord(
                    tool_name=raw_result.tool_name,
                    status=status,
                    provider=raw_result.provider,
                    summary=summary,
                )
            )
        iteration_record = IterationRecord(
            iteration=iteration,
            new_candidate_count=len(candidates),
            accepted_candidate_count=len(ledger.selected),
            rejected_candidate_count=len(ledger.rejected),
            coverage=coverage.total,
            coverage_delta=coverage_delta,
            uncovered_subtopics=uncovered_subtopics,
            estimated_cost_usd=iteration_cost,
            tool_calls=tool_calls,
        )

        elapsed_seconds = int(monotonic() - start_time)
        decision = check_convergence(
            coverage,
            iteration_history,
            spent_usd=spent_usd,
            elapsed_seconds=elapsed_seconds,
            max_iterations=config.max_iterations,
            epsilon=config.convergence_epsilon,
            min_coverage=config.convergence_min_coverage,
            budget_limit_usd=config.cost_budget_usd,
            time_limit_seconds=config.time_box_seconds,
            new_candidate_count=iteration_record.new_candidate_count,
        )
        continue_reason = (
            None
            if decision.should_stop
            else (
                "remaining uncovered subtopics: " + ", ".join(uncovered_subtopics)
                if uncovered_subtopics
                else f"coverage improved by {coverage_delta:.2f}; continue exploring"
            )
        )
        iteration_record = iteration_record.model_copy(
            update={
                "continue_reason": continue_reason,
                "stop_reason": decision.reason if decision.should_stop else None,
            }
        )
        iteration_history.append(iteration_record)
        log(
            iteration=iteration,
            coverage=coverage.total,
            coverage_delta=coverage_delta,
            uncovered_subtopics=uncovered_subtopics,
            new_candidate_count=iteration_record.new_candidate_count,
            accepted_candidate_count=iteration_record.accepted_candidate_count,
            rejected_candidate_count=iteration_record.rejected_candidate_count,
            tool_summaries=[tool_call.summary for tool_call in tool_calls],
            stop_reason=iteration_record.stop_reason,
            continue_reason=continue_reason,
            spent_usd=round(spent_usd, 6),
        )
        if decision.should_stop:
            stop_reason = decision.reason or StopReason.MAX_ITERATIONS
            break

    selection = build_selection_graph.submit(ledger, plan, config).load()
    reading_future = render_reading_path.submit(selection, ledger, plan, config)
    backing_future = render_backing_report.submit(
        selection,
        ledger,
        plan,
        IterationTrace(iterations=iteration_history),
        provider_usage_summary,
        stop_reason,
        config,
    )
    reading_result = reading_future.load()
    backing_result = backing_future.load()
    spent_usd += reading_result.budget.estimated_cost_usd
    spent_usd += backing_result.budget.estimated_cost_usd
    critique_result: CritiqueResult | None = None
    grounding_result: GroundingResult | None = None
    coherence_result: CoherenceResult | None = None
    renders = [reading_result.render, backing_result.render]

    if config.critique_enabled:
        critique_checkpoint = review_renders.submit(
            renders,
            plan,
            selection,
            ledger,
            config,
        ).load()
        critique_result = critique_checkpoint.critique
        spent_usd += critique_checkpoint.budget.estimated_cost_usd
        renders = revise_renders.submit(renders, critique_result, plan).load()

    if config.judge_enabled:
        grounding_future = judge_grounding.submit(renders, ledger, config)
        coherence_future = judge_coherence.submit(renders, plan, config)
        grounding_checkpoint = grounding_future.load()
        coherence_checkpoint = coherence_future.load()
        grounding_result = grounding_checkpoint.grounding
        coherence_result = coherence_checkpoint.coherence
        spent_usd += grounding_checkpoint.budget.estimated_cost_usd
        spent_usd += coherence_checkpoint.budget.estimated_cost_usd

    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elapsed_seconds = int(monotonic() - start_time)
    return assemble_package(
        run_summary=RunSummary(
            run_id=f"run-{uuid4()}",
            brief=brief,
            tier=config.tier,
            stop_reason=stop_reason,
            status="completed",
            estimated_cost_usd=round(spent_usd, 6),
            elapsed_seconds=elapsed_seconds,
            iteration_count=len(iteration_history),
            provider_usage_summary=provider_usage_summary,
            council_enabled=config.council_mode,
            council_size=config.council_size if config.council_mode else 1,
            council_models=council_models,
            started_at=started_at,
            completed_at=completed_at,
        ),
        research_plan=plan,
        evidence_ledger=ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(iterations=iteration_history),
        renders=renders,
        render_settings=RenderSettingsSnapshot(writer_model=config.writer_model),
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
    )
