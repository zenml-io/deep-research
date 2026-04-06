from time import monotonic
from uuid import uuid4

from kitaru import flow, log, wait

from deep_research.checkpoints.classify import classify_request
from deep_research.checkpoints.council import (
    aggregate_council_results,
    run_council_generator,
)
from deep_research.checkpoints.assemble import assemble_package
from deep_research.checkpoints.evaluate import evaluate_coverage
from deep_research.checkpoints.merge import merge_evidence
from deep_research.checkpoints.normalize import normalize_evidence
from deep_research.checkpoints.plan import build_plan
from deep_research.checkpoints.relevance import score_relevance
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
    RunSummary,
)
from deep_research.renderers.backing_report import render_backing_report
from deep_research.renderers.reading_path import render_reading_path

CLASSIFY_CHECKPOINT_NAME = "classify_request"
PLAN_CHECKPOINT_NAME = "build_plan"
SUPERVISOR_CHECKPOINT_NAME = "run_supervisor"
COUNCIL_GENERATOR_CHECKPOINT_NAME = "run_council_generator"
APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"


def _run_iteration(plan, ledger, iteration, config, council_models):
    """Execute one research iteration via council mode or single supervisor."""
    if config.council_mode:
        futures = [
            run_council_generator.submit(
                plan, ledger, iteration, model_name, config,
                id=f"council_{index}",
            )
            for index, model_name in enumerate(council_models)
        ]
        return aggregate_council_results([f.load() for f in futures])
    return run_supervisor.submit(plan, ledger, iteration, config).load()


def _resolve_runtime_config(
    config: ResearchConfig | None,
    resolved_tier: Tier,
) -> ResearchConfig:
    """Return a ResearchConfig aligned to the resolved tier, merging any overrides."""
    base_config = ResearchConfig.for_tier(resolved_tier)
    if config is None:
        return base_config
    if config.tier == resolved_tier:
        return config
    overrides = config.model_dump()
    overrides["tier"] = resolved_tier
    return base_config.model_copy(update=overrides)


@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    """Orchestrate the full research pipeline from brief to investigation package."""
    classification = classify_request.submit(brief, config).load()
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    config = _resolve_runtime_config(config, resolved_tier)

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
        config = _resolve_runtime_config(config, resolved_tier)

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
    spent_usd = 0.0
    council_models = [config.supervisor_model] * config.council_size
    stop_reason = StopReason.MAX_ITERATIONS
    start_time = monotonic()

    for iteration in range(config.max_iterations):
        supervisor_result = _run_iteration(
            plan, ledger, iteration, config, council_models,
        )
        spent_usd += supervisor_result.budget.estimated_cost_usd

        candidates = normalize_evidence.submit(
            supervisor_result.raw_results,
        ).load()
        relevance_result = score_relevance.submit(
            candidates, plan, config,
        ).load()
        spent_usd += relevance_result.budget.estimated_cost_usd

        ledger = merge_evidence.submit(
            relevance_result.candidates, ledger,
        ).load()
        coverage = evaluate_coverage.submit(ledger, plan).load()
        iteration_history.append(
            IterationRecord(
                iteration=iteration,
                new_candidate_count=len(candidates),
                coverage=coverage.total,
            )
        )

        decision = check_convergence(
            coverage,
            iteration_history[:-1],
            spent_usd=spent_usd,
            elapsed_seconds=int(monotonic() - start_time),
            max_iterations=config.max_iterations,
            epsilon=config.convergence_epsilon,
            min_coverage=config.convergence_min_coverage,
            budget_limit_usd=config.cost_budget_usd,
            time_limit_seconds=config.time_box_seconds,
        )
        log(iteration=iteration, coverage=coverage.total, spent_usd=spent_usd)
        if decision.should_stop:
            stop_reason = decision.reason or StopReason.MAX_ITERATIONS
            break

    selection = build_selection_graph.submit(ledger, plan, config).load()
    reading_future = render_reading_path.submit(selection)
    backing_future = render_backing_report.submit(selection, ledger, plan)
    return assemble_package(
        run_summary=RunSummary(
            run_id=f"run-{uuid4()}",
            brief=brief,
            tier=config.tier,
            stop_reason=stop_reason,
            status="completed",
        ),
        research_plan=plan,
        evidence_ledger=ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(iterations=iteration_history),
        renders=[reading_future.load(), backing_future.load()],
    )
