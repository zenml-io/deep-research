import inspect
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


def _call_with_optional_dependency(func, *args, after=None, id=None, **kwargs):
    call_kwargs = dict(kwargs)
    if after is not None:
        call_kwargs["after"] = after
    if id is not None:
        call_kwargs["id"] = id
    try:
        return func(*args, **call_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        return func(*args, **fallback_kwargs)


def _submit_with_optional_dependency(func, *args, after=None, id=None, **kwargs):
    if inspect.isfunction(func) or inspect.ismethod(func):
        return _call_with_optional_dependency(
            func,
            *args,
            after=after,
            id=id,
            **kwargs,
        )

    submit = getattr(func, "submit")
    submit_kwargs = dict(kwargs)
    if after is not None:
        submit_kwargs["after"] = after
    if id is not None:
        submit_kwargs["id"] = id
    try:
        return submit(*args, **submit_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return submit(*args, **kwargs)


def _resolve_checkpoint_output(value):
    loader = getattr(value, "load", None)
    if callable(loader):
        return loader()
    return value


def _resolve_council_models(config: ResearchConfig) -> list[str]:
    return [config.supervisor_model] * config.council_size


def _run_council_iteration(plan, ledger, iteration, config, council_models):
    futures = [
        _submit_with_optional_dependency(
            run_council_generator,
            plan,
            ledger,
            iteration,
            model_name,
            config,
            id=f"council_{index}",
        )
        for index, model_name in enumerate(council_models)
    ]
    return aggregate_council_results(
        [_resolve_checkpoint_output(future) for future in futures]
    )


def _run_council_iteration_with_dependency(
    plan,
    ledger,
    iteration,
    config,
    council_models,
    after=None,
):
    futures = [
        _submit_with_optional_dependency(
            run_council_generator,
            plan,
            ledger,
            iteration,
            model_name,
            config,
            id=f"council_{index}",
            after=after,
        )
        for index, model_name in enumerate(council_models)
    ]
    return (
        aggregate_council_results(
            [_resolve_checkpoint_output(future) for future in futures]
        ),
        futures,
    )


def _run_iteration(plan, ledger, iteration, config, council_models):
    if config.council_mode:
        return _resolve_checkpoint_output(
            _run_council_iteration(plan, ledger, iteration, config, council_models)
        )
    return _resolve_checkpoint_output(run_supervisor(plan, ledger, iteration, config))


_DEFAULT_RUN_ITERATION = _run_iteration


def _run_iteration_with_dependency(
    plan,
    ledger,
    iteration,
    config,
    council_models,
    after=None,
):
    if config.council_mode:
        return _run_council_iteration_with_dependency(
            plan,
            ledger,
            iteration,
            config,
            council_models,
            after=after,
        )
    checkpoint_output = _submit_with_optional_dependency(
        run_supervisor,
        plan,
        ledger,
        iteration,
        config,
        after=after,
    )
    return _resolve_checkpoint_output(checkpoint_output), checkpoint_output


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


@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    classification_dependency = _submit_with_optional_dependency(
        classify_request,
        brief,
        config,
    )
    classification = _resolve_checkpoint_output(classification_dependency)
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    config = _resolve_runtime_config(config, resolved_tier)

    if classification.needs_clarification and classification.clarification_question:
        brief = wait(
            name=CLARIFY_BRIEF_WAIT_NAME,
            schema=str,
            question=classification.clarification_question,
        )
        classification_dependency = _submit_with_optional_dependency(
            classify_request,
            brief,
            config,
            after=classification_dependency,
        )
        classification = _resolve_checkpoint_output(classification_dependency)
        resolved_tier = (
            classification.recommended_tier if tier == "auto" else Tier(tier)
        )
        config = _resolve_runtime_config(config, resolved_tier)

    plan_dependency = _submit_with_optional_dependency(
        build_plan,
        brief,
        classification_dependency,
        config.tier,
    )
    plan = _resolve_checkpoint_output(plan_dependency)
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
    council_models = _resolve_council_models(config)
    stop_reason = StopReason.MAX_ITERATIONS
    start_time = monotonic()
    iteration_dependency = plan_dependency
    ledger_dependency = None

    for iteration in range(config.max_iterations):
        if _run_iteration is _DEFAULT_RUN_ITERATION:
            supervisor_result, supervisor_dependency = _run_iteration_with_dependency(
                plan_dependency,
                ledger_dependency or ledger,
                iteration,
                config,
                council_models,
                after=iteration_dependency,
            )
        else:
            supervisor_result = _run_iteration(
                plan,
                ledger,
                iteration,
                config,
                council_models,
            )
            supervisor_dependency = iteration_dependency
        spent_usd += supervisor_result.budget.estimated_cost_usd

        normalized_dependency = _submit_with_optional_dependency(
            normalize_evidence,
            supervisor_result.raw_results,
            after=supervisor_dependency,
        )
        candidates = _resolve_checkpoint_output(normalized_dependency)
        relevance_result = _resolve_checkpoint_output(
            relevance_dependency := _submit_with_optional_dependency(
                score_relevance,
                normalized_dependency,
                plan_dependency,
                config,
            )
        )
        spent_usd += relevance_result.budget.estimated_cost_usd

        scored = relevance_result.candidates
        ledger_dependency = _submit_with_optional_dependency(
            merge_evidence,
            scored,
            ledger_dependency or ledger,
            after=relevance_dependency,
        )
        ledger = _resolve_checkpoint_output(ledger_dependency)
        coverage_dependency = _submit_with_optional_dependency(
            evaluate_coverage,
            ledger_dependency,
            plan_dependency,
        )
        coverage = _resolve_checkpoint_output(coverage_dependency)
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
        iteration_dependency = coverage_dependency
        if decision.should_stop:
            stop_reason = decision.reason or StopReason.MAX_ITERATIONS
            break

    selection = _submit_with_optional_dependency(
        build_selection_graph,
        ledger_dependency or ledger,
        plan_dependency,
        after=iteration_dependency,
    )
    reading_path = _submit_with_optional_dependency(
        render_reading_path,
        selection,
    )
    backing_report = _submit_with_optional_dependency(
        render_backing_report,
        selection,
        ledger_dependency or ledger,
        plan_dependency,
    )
    return _call_with_optional_dependency(
        assemble_package,
        run_summary=RunSummary(
            run_id=f"run-{uuid4()}",
            brief=brief,
            tier=config.tier,
            stop_reason=stop_reason,
            status="completed",
        ),
        research_plan=plan_dependency,
        evidence_ledger=ledger_dependency or ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(iterations=iteration_history),
        renders=[reading_path, backing_report],
    )
