from time import monotonic
from uuid import uuid4

from kitaru import flow, log, wait

from deep_research.checkpoints.classify import classify_request
from deep_research.checkpoints.council import (
    aggregate_council_results,
    run_council_generator,
)
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
from deep_research.package.assembly import assemble_package
from deep_research.renderers.backing_report import render_backing_report
from deep_research.renderers.reading_path import render_reading_path


class _ImmediateFlowHandle:
    def __init__(self, value):
        self._value = value

    def wait(self):
        return self._value

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __getitem__(self, key):
        return self._value[key]


def _resolve_council_models(config: ResearchConfig) -> list[str]:
    return [config.supervisor_model] * config.council_size


def _run_council_iteration(plan, ledger, iteration, config, council_models):
    futures = [
        run_council_generator.submit(
            plan,
            ledger,
            iteration,
            model_name,
            config,
            id=f"council_{index}",
        )
        for index, model_name in enumerate(council_models)
    ]
    return aggregate_council_results([future.load() for future in futures])


def _run_iteration(plan, ledger, iteration, config, council_models):
    if config.council_mode:
        return _run_council_iteration(plan, ledger, iteration, config, council_models)
    return run_supervisor(plan, ledger, iteration, config)


def _resolve_runtime_config(
    config: ResearchConfig | None,
    resolved_tier: Tier,
) -> ResearchConfig:
    base_config = ResearchConfig.for_tier(resolved_tier)
    if config is None:
        return base_config
    if config.tier == resolved_tier:
        return config
    overrides = config.model_dump()
    overrides["tier"] = resolved_tier
    return base_config.model_copy(update=overrides)


def _coerce_flow_handle(result):
    if hasattr(result, "wait"):
        return result
    return _ImmediateFlowHandle(result)


@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    classification = classify_request(brief, config)
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    config = _resolve_runtime_config(config, resolved_tier)

    if classification.needs_clarification and classification.clarification_question:
        brief = wait(
            name="clarify_brief",
            schema=str,
            question=classification.clarification_question,
        )
        classification = classify_request(brief, config)
        resolved_tier = (
            classification.recommended_tier if tier == "auto" else Tier(tier)
        )
        config = _resolve_runtime_config(config, resolved_tier)

    plan = build_plan(brief, classification, config.tier)
    if config.require_plan_approval:
        approved = wait(
            name="approve_plan",
            schema=bool,
            question=f"Approve plan for: {plan.goal}?",
        )
        if approved is False:
            raise ValueError("plan not approved")

    ledger = EvidenceLedger()
    iteration_history: list[IterationRecord] = []
    spent_usd = 0.0
    council_models = _resolve_council_models(config)
    stop_reason = StopReason.MAX_ITERATIONS
    start_time = monotonic()

    for iteration in range(config.max_iterations):
        supervisor_result = _run_iteration(
            plan, ledger, iteration, config, council_models
        )
        spent_usd += supervisor_result.budget.estimated_cost_usd

        candidates = normalize_evidence(supervisor_result.raw_results)
        relevance_result = score_relevance(candidates, plan, config)
        spent_usd += relevance_result.budget.estimated_cost_usd

        scored = relevance_result.candidates
        ledger = merge_evidence(scored, ledger)
        coverage = evaluate_coverage(ledger, plan)
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
            epsilon=0.05,
            min_coverage=0.60,
            budget_limit_usd=config.cost_budget_usd,
            time_limit_seconds=config.time_box_seconds,
        )
        log(iteration=iteration, coverage=coverage.total, spent_usd=spent_usd)
        if decision.should_stop:
            stop_reason = decision.reason or StopReason.MAX_ITERATIONS
            break

    selection = build_selection_graph(ledger, plan)
    reading_path = render_reading_path(selection)
    backing_report = render_backing_report(selection, ledger, plan)
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
        renders=[reading_path, backing_report],
    )


_research_flow_run = research_flow.run
research_flow.run = lambda *args, **kwargs: _coerce_flow_handle(
    _research_flow_run(*args, **kwargs)
)
