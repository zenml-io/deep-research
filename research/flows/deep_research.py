"""Default flow orchestration for deep research.

Thin @flow entry point that delegates orchestration to checkpoints and
small pure helpers.
Pipeline phases:
  1. scope -> ResearchBrief
  2. plan  -> ResearchPlan -> optional plan-approval wait
  3. Iteration loop: supervisor -> subagents -> merge -> convergence
  4. draft -> critique -> (optional supplemental loop) -> finalize -> assemble
  5. optional durable export checkpoint
"""

import logging
from collections.abc import Callable
from typing import Any

from kitaru import flow, wait

from research.flows.errors import (
    FinalizerError,
    FlowTimeoutError,
    PlanApprovalRejectedError,
    SupervisorError,
)

# Import all checkpoints at module top level (enables monkeypatching in tests)
from research.checkpoints.assemble import assemble_package
from research.checkpoints.critique import run_critique
from research.checkpoints.draft import run_draft
from research.checkpoints.export import export_package
from research.checkpoints.finalize import run_finalize
from research.checkpoints.metadata import (
    finalize_run_metadata,
    snapshot_wall_clock,
    stamp_run_metadata,
)
from research.checkpoints.plan import run_plan
from research.checkpoints.scope import run_scope
from research.checkpoints.subagent import run_subagent
from research.checkpoints.supervisor import run_supervisor
from research.config.settings import ResearchConfig
from research.contracts.brief import ResearchBrief
from research.contracts.decisions import SubagentFindings, SupervisorDecision
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import InvestigationPackage, RunMetadata, ToolProviderManifest
from research.contracts.plan import ResearchPlan, SubagentTask
from research.flows.convergence import StopDecision
from research.flows.budget import BudgetTracker, reset_active_tracker, set_active_tracker
from research.flows.convergence import check_convergence
from research.ledger.ledger import ManagedLedger
from research.ledger.projection import format_projection, project_ledger
from research.package.export import resolve_package_run_dir
from research.providers.agent_tools import build_tool_provider_manifest, build_tool_surface
from research.providers.search import ProviderRegistry

logger = logging.getLogger(__name__)



def _wait_for_input(
    *,
    schema,
    name: str,
    question: str,
    timeout_seconds: int,
    timeout_message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Resolve a flow-level wait with typed timeout handling."""
    try:
        value = wait(
            schema=schema,
            name=name,
            question=question,
            timeout=timeout_seconds,
            metadata=metadata,
        )
    except Exception as exc:  # pragma: no cover - exercised by wait stubs/tests
        raise FlowTimeoutError(timeout_message) from exc

    if value is None:
        raise FlowTimeoutError(timeout_message)
    return value


def _await_plan_approval(question: str, plan, cfg: ResearchConfig) -> None:
    """Pause for operator approval of the generated plan."""
    approved = _wait_for_input(
        schema=bool,
        name="approve_research_plan",
        question=(
            "Approve the generated research plan before the investigation runs?"
        ),
        timeout_seconds=cfg.wait_timeout_seconds,
        timeout_message=(
            "Timed out waiting for plan approval after "
            f"{cfg.wait_timeout_seconds} seconds"
        ),
        metadata={
            "raw_question": question,
            "plan": plan.model_dump(mode="json"),
        },
    )
    if approved is not True:
        raise PlanApprovalRejectedError("Research plan was not approved by the operator")


def _build_tools_and_manifest(
    cfg: ResearchConfig,
) -> tuple[list[Callable[..., Any]] | None, ToolProviderManifest]:
    """Resolve the subagent tool surface and a durable audit manifest."""
    registry: ProviderRegistry | None = None
    surface = None
    tools: list[Callable[..., Any]] | None = None
    degradation_reasons: list[str] = []

    try:
        registry = ProviderRegistry(cfg)
    except Exception as exc:
        logger.warning("Failed to build provider registry: %s", exc)
        degradation_reasons.append(f"provider_registry_failed: {exc}")
    else:
        try:
            surface = build_tool_surface(cfg, registry)
            tools = surface.as_pydantic_tools()
        except Exception as exc:
            logger.warning(
                "Failed to build tool surface; subagents will run without tools: %s",
                exc,
            )
            degradation_reasons.append(f"tool_surface_build_failed: {exc}")

    manifest = build_tool_provider_manifest(
        cfg,
        registry,
        surface,
        degradation_reasons=degradation_reasons,
    )
    return tools, manifest


def _fan_out_subagents(
    tasks: list[SubagentTask],
    model_name: str,
    max_parallel: int,
    tools: list[Callable[..., Any]] | None = None,
) -> tuple[list[SubagentFindings], list[Any]]:
    """Fan-out subagent tasks in batches of max_parallel, never spilling across iterations.

    Returns (results, submit_handles) — callers need the handles for DAG edge tracking.
    """
    results = []
    all_handles = []
    for batch_start in range(0, len(tasks), max_parallel):
        batch = tasks[batch_start : batch_start + max_parallel]
        handles = [run_subagent.submit(t, model_name, tools=tools) for t in batch]
        all_handles.extend(handles)
        results.extend(h.load() for h in handles)
    return results, all_handles


def _run_supervisor_with_retry(
    brief,
    plan,
    projection: str,
    remaining: float,
    iteration_index: int,
    gen_model: str,
    ledger: EvidenceLedger,
    breadth_first: bool = False,
    max_iterations: int = 5,
    ledger_size: int = 0,
):
    """Call run_supervisor with one retry on failure.

    On first failure: log warning, retry once.
    On second failure: raise SupervisorError with the last valid ledger.

    Each attempt uses a distinct checkpoint ``id`` so Kitaru does not
    serve a cached failure for the retry.

    Returns (decision, submit_handles) — callers need the handles for DAG edge tracking.
    """
    # Return type: tuple[SupervisorDecision, list[Any]]
    handles: list[Any] = []
    try:
        h = run_supervisor.submit(
            brief,
            plan,
            projection,
            remaining,
            iteration_index,
            gen_model,
            breadth_first=breadth_first,
            max_iterations=max_iterations,
            ledger_size=ledger_size,
            id=f"supervisor_{iteration_index}_a1",
        )
        handles.append(h)
        return h.load(), handles
    except Exception as first_err:
        logger.warning(
            "Supervisor failed (attempt 1/2) at iteration %d: %s",
            iteration_index,
            first_err,
        )
        try:
            h = run_supervisor.submit(
                brief,
                plan,
                projection,
                remaining,
                iteration_index,
                gen_model,
                breadth_first=breadth_first,
                max_iterations=max_iterations,
                ledger_size=ledger_size,
                id=f"supervisor_{iteration_index}_a2",
            )
            handles.append(h)
            return h.load(), handles
        except Exception as second_err:
            raise SupervisorError(
                f"Supervisor failed twice at iteration {iteration_index}: "
                f"first={first_err!r}, second={second_err!r}",
                ledger=ledger,
            ) from second_err


def _run_iteration(
    cfg: ResearchConfig,
    brief: ResearchBrief,
    plan: ResearchPlan,
    managed_ledger: ManagedLedger,
    iteration_index: int,
    started_at: str,
    tools: list[Callable[..., Any]] | None = None,
    pinned_ids: list[str] | None = None,
) -> tuple[IterationRecord, StopDecision, SupervisorDecision, list[Any]]:
    """Execute one research iteration: supervisor -> subagents -> merge -> record.

    Returns (record, convergence, decision, submit_handles) — callers need
    the handles for DAG edge tracking and decision for pinned_ids propagation.
    """
    iter_handles: list[Any] = []

    clock_h = snapshot_wall_clock.submit(started_at)
    iter_handles.append(clock_h)
    clock = clock_h.load()

    gen_model = cfg.slots["generator"].model_string
    sub_model = cfg.slots["subagent"].model_string

    ledger = managed_ledger.ledger
    projection = format_projection(
        project_ledger(
            ledger,
            iteration_index,
            pinned_ids=pinned_ids,
            window_iterations=cfg.ledger_window_iterations,
        )
    )
    remaining = cfg.budget.soft_budget_usd - cfg.budget.spent_usd

    decision, sup_handles = _run_supervisor_with_retry(
        brief,
        plan,
        projection,
        remaining,
        iteration_index,
        gen_model,
        ledger,
        breadth_first=cfg.breadth_first,
        max_iterations=cfg.max_iterations,
        ledger_size=managed_ledger.size,
    )
    iter_handles.extend(sup_handles)

    subagent_results, sub_handles = _fan_out_subagents(
        decision.subagent_tasks, sub_model, cfg.max_parallel_subagents, tools=tools
    )
    iter_handles.extend(sub_handles)

    for findings in subagent_results:
        if findings.findings:
            added = managed_ledger.merge_findings(findings, iteration_index)
            if added:
                logger.info(
                    "Iteration %d: merged %d evidence items (ledger total: %d)",
                    iteration_index,
                    len(added),
                    managed_ledger.size,
                )

    record = IterationRecord(
        iteration_index=iteration_index,
        supervisor_decision=decision,
        subagent_results=subagent_results,
        ledger_size=managed_ledger.size,
    )

    convergence = check_convergence(
        budget=cfg.budget,
        elapsed_seconds=clock.elapsed_seconds,
        time_limit_seconds=cfg.wait_timeout_seconds,
        supervisor_decision=decision,
        iteration_index=iteration_index,
        max_iterations=cfg.max_iterations,
        respect_supervisor_done=cfg.respect_supervisor_done,
    )

    return record, convergence, decision, iter_handles


def _run_deep_research_pipeline(
    question: str,
    *,
    tier: str,
    cfg: ResearchConfig,
    output_dir: str | None = None,
    require_plan_approval: bool = True,
):
    """Submit the deep research pipeline and return a handle.

    Wraps ``deep_research.run()`` to expose the ``.load()`` interface
    expected by ``council_research``. All council tests monkeypatch this
    function, so the implementation is only exercised in production.
    """
    import types

    handle = deep_research.run(
        question,
        tier=tier,
        config=cfg,
        output_dir=output_dir,
        require_plan_approval=require_plan_approval,
    )
    return types.SimpleNamespace(load=handle.wait)


@flow
def deep_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
    output_dir: str | None = None,
    require_plan_approval: bool = True,
) -> InvestigationPackage:
    """Orchestrate a full deep-research investigation."""
    cfg = config or ResearchConfig.for_tier(tier)
    tracker = BudgetTracker(
        budget=cfg.budget,
        strict_unknown_model_cost=cfg.strict_unknown_model_cost,
    )
    tracker_token = set_active_tracker(tracker)
    try:
        all_handles: list[Any] = []

        stamp_h = stamp_run_metadata.submit()
        all_handles.append(stamp_h)
        stamp = stamp_h.load()

        # Phase 1: Scope
        gen_model = cfg.slots["generator"].model_string
        scope_model = cfg.scope_override.model_string if cfg.scope_override else gen_model
        scope_h = run_scope.submit(question, scope_model)
        all_handles.append(scope_h)
        brief = scope_h.load()

        # Phase 2: Plan
        plan_h = run_plan.submit(brief, gen_model)
        all_handles.append(plan_h)
        plan = plan_h.load()

        if require_plan_approval:
            _await_plan_approval(question, plan, cfg)

        # Phase 3: Build run-scoped tool surface for subagents + durable manifest
        tools, tool_provider_manifest = _build_tools_and_manifest(cfg)

        # Phase 4: Iteration loop
        managed_ledger = ManagedLedger()
        iterations: list[IterationRecord] = []
        stop_reason: str | None = None
        pinned_ids: list[str] = []

        for i in range(cfg.max_iterations):
            record, convergence, decision, iter_handles = _run_iteration(
                cfg,
                brief,
                plan,
                managed_ledger,
                i,
                stamp.started_at,
                tools=tools,
                pinned_ids=pinned_ids,
            )
            all_handles.extend(iter_handles)
            iterations.append(record)
            pinned_ids = decision.pinned_evidence_ids

            if convergence.should_stop:
                stop_reason = convergence.reason.value
                break

        ledger = managed_ledger.ledger

        # Phase 5: Draft
        draft_h = run_draft.submit(brief, plan, ledger, gen_model)
        all_handles.append(draft_h)
        draft = draft_h.load()

        # Phase 6: Critique
        reviewer_model = cfg.slots["reviewer"].model_string
        second = cfg.second_reviewer.model_string if cfg.second_reviewer else None
        critique_h = run_critique.submit(draft, plan, ledger, reviewer_model, second)
        all_handles.append(critique_h)
        critique = critique_h.load()

        # Phase 7: Supplemental loop (capped at max_supplemental_loops)
        supplemental_loops = 0
        while (
            critique.require_more_research
            and supplemental_loops < cfg.max_supplemental_loops
        ):
            supplemental_loops += 1
            idx = len(iterations)
            record, _conv, _decision, iter_handles = _run_iteration(
                cfg,
                brief,
                plan,
                managed_ledger,
                idx,
                stamp.started_at,
                tools=tools,
                pinned_ids=pinned_ids,
            )
            all_handles.extend(iter_handles)
            iterations.append(record)
            ledger = managed_ledger.ledger

            draft_h = run_draft.submit(brief, plan, ledger, gen_model)
            all_handles.append(draft_h)
            draft = draft_h.load()

            critique_h = run_critique.submit(
                draft, plan, ledger, reviewer_model, second
            )
            all_handles.append(critique_h)
            critique = critique_h.load()
            # Second require_more_research is recorded but the while guard
            # stops because supplemental_loops == max_supplemental_loops.

        # Phase 8: Finalize
        try:
            finalize_h = run_finalize.submit(
                draft, critique, ledger, gen_model, stop_reason
            )
            all_handles.append(finalize_h)
            final_report = finalize_h.load()
        except Exception as finalize_err:
            logger.warning("Finalizer failed: %s", finalize_err)
            final_report = None

        if final_report is None:
            if cfg.allow_unfinalized_package:
                logger.warning(
                    "Finalizer produced no report; proceeding with unfinalized package."
                )
                stop_reason = stop_reason or "finalizer_failed"
            else:
                raise FinalizerError(
                    "Finalizer failed and allow_unfinalized_package is False. "
                    "Draft and critique are preserved in checkpoint history for replay."
                )

        # Phase 9: Assemble
        fin_h = finalize_run_metadata.submit(stamp.started_at)
        all_handles.append(fin_h)
        fin = fin_h.load()

        export_path = (
            str(resolve_package_run_dir(output_dir, stamp.run_id))
            if output_dir is not None
            else None
        )
        run_metadata = RunMetadata(
            run_id=stamp.run_id,
            tier=cfg.tier,
            started_at=stamp.started_at,
            completed_at=fin.completed_at,
            total_cost_usd=cfg.budget.spent_usd,
            total_iterations=len(iterations),
            stop_reason=stop_reason,
            export_path=export_path,
        )

        assemble_h = assemble_package.submit(
            metadata=run_metadata,
            brief=brief,
            plan=plan,
            ledger=ledger,
            iterations=iterations,
            draft=draft,
            critique=critique,
            final_report=final_report,
            tool_provider_manifest=tool_provider_manifest,
            grounding_min_ratio=cfg.grounding_min_ratio,
            strict_grounding=cfg.strict_grounding,
            after=all_handles,
        )
        all_handles.append(assemble_h)

        if output_dir is None:
            return assemble_h

        package = assemble_h.load()
        return export_package.submit(package, output_dir, after=all_handles)
    finally:
        reset_active_tracker(tracker_token)
