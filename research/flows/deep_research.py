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
from research.checkpoints.verify import run_verify
from research.checkpoints.plan import run_plan
from research.checkpoints.replan import run_plan_revision
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
from research.contracts.reports import CritiqueReport, VerificationReport
from research.flows.convergence import StopDecision
from research.flows.budget import BudgetTracker, reset_active_tracker, set_active_tracker
from research.flows.convergence import check_convergence
from research.ledger.ledger import ManagedLedger
from research.ledger.projection import format_projection, project_ledger
from research.package.export import resolve_package_run_dir
from research.providers.agent_tools import build_tool_provider_manifest, build_tool_surface
from research.providers.search import ProviderRegistry

logger = logging.getLogger(__name__)

_LOW_CRITIQUE_SCORE_THRESHOLD = 0.7
_MAX_CRITIQUE_ISSUES = 4
_MAX_LOW_SCORE_DIMENSIONS = 3
_MAX_CRITIQUE_FEEDBACK_CHARS = 1200


def _summarise_critique_for_supervisor(critique: CritiqueReport) -> str | None:
    """Project critique findings into a compact, replay-safe supervisor hint."""
    issues: list[str] = []
    seen_issues: set[str] = set()
    for issue in critique.issues:
        cleaned = issue.strip()
        if not cleaned or cleaned in seen_issues:
            continue
        seen_issues.add(cleaned)
        issues.append(cleaned)

    low_dimensions = [
        dimension
        for dimension in critique.dimensions
        if dimension.score < _LOW_CRITIQUE_SCORE_THRESHOLD
    ]

    low_dimension_heading = (
        f"Low-scoring dimensions (< {_LOW_CRITIQUE_SCORE_THRESHOLD:.2f}):"
    )
    if not issues and not low_dimensions:
        if critique.require_more_research and critique.dimensions:
            low_dimensions = sorted(
                critique.dimensions,
                key=lambda dimension: (dimension.score, dimension.dimension),
            )[:_MAX_LOW_SCORE_DIMENSIONS]
            low_dimension_heading = (
                f"Weakest dimensions (all scores >= "
                f"{_LOW_CRITIQUE_SCORE_THRESHOLD:.2f}):"
            )
        elif critique.require_more_research:
            return (
                "Supplemental critique feedback:\n"
                "Reviewer requested more research; tighten the next iteration "
                "against the latest draft review."
            )
        else:
            return None

    lines = ["Supplemental critique feedback:"]

    if issues:
        lines.append("Issues to address:")
        for issue in issues[:_MAX_CRITIQUE_ISSUES]:
            lines.append(f"- {issue}")
        remaining_issues = len(issues) - _MAX_CRITIQUE_ISSUES
        if remaining_issues > 0:
            lines.append(f"- ...and {remaining_issues} more issues.")

    if low_dimensions:
        lines.append(low_dimension_heading)
        for dimension in low_dimensions[:_MAX_LOW_SCORE_DIMENSIONS]:
            explanation = dimension.explanation.strip()
            line = f"- {dimension.dimension} ({dimension.score:.2f})"
            if explanation:
                line = f"{line}: {explanation}"
            lines.append(line)
        remaining_dimensions = len(low_dimensions) - _MAX_LOW_SCORE_DIMENSIONS
        if remaining_dimensions > 0:
            lines.append(
                f"- ...and {remaining_dimensions} more low-scoring dimensions."
            )

    summary = "\n".join(lines)
    if len(summary) > _MAX_CRITIQUE_FEEDBACK_CHARS:
        summary = summary[: _MAX_CRITIQUE_FEEDBACK_CHARS - 3].rstrip() + "..."
    return summary



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


def _apply_brief_recency_default(
    tasks: list[SubagentTask],
    brief: ResearchBrief,
) -> list[SubagentTask]:
    """Apply brief-level recency default to tasks that leave it unset."""
    if brief.recency_days is None:
        return tasks

    resolved_tasks: list[SubagentTask] = []
    for task in tasks:
        if task.recency_days is None:
            resolved_tasks.append(
                task.model_copy(update={"recency_days": brief.recency_days})
            )
        else:
            resolved_tasks.append(task)
    return resolved_tasks


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
    critique_feedback: str | None = None,
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

    def _submit(attempt: int):
        kwargs = {
            "breadth_first": breadth_first,
            "max_iterations": max_iterations,
            "ledger_size": ledger_size,
            "id": f"supervisor_{iteration_index}_a{attempt}",
        }
        if critique_feedback is not None:
            kwargs["critique_feedback"] = critique_feedback
        return run_supervisor.submit(
            brief,
            plan,
            projection,
            remaining,
            iteration_index,
            gen_model,
            **kwargs,
        )

    try:
        h = _submit(1)
        handles.append(h)
        return h.load(), handles
    except Exception as first_err:
        logger.warning(
            "Supervisor failed (attempt 1/2) at iteration %d: %s",
            iteration_index,
            first_err,
        )
        try:
            h = _submit(2)
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
    critique_feedback: str | None = None,
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
        critique_feedback=critique_feedback,
    )
    iter_handles.extend(sup_handles)

    resolved_tasks = _apply_brief_recency_default(decision.subagent_tasks, brief)
    if resolved_tasks is not decision.subagent_tasks:
        decision = decision.model_copy(update={"subagent_tasks": resolved_tasks})

    subagent_results, sub_handles = _fan_out_subagents(
        resolved_tasks, sub_model, cfg.max_parallel_subagents, tools=tools
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
        supervisor_done_ignored=decision.done and not cfg.respect_supervisor_done,
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
        original_plan = plan
        revised_plan: ResearchPlan | None = None

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
        critique_h = run_critique.submit(
            draft,
            plan,
            ledger,
            reviewer_model,
            second,
            disagreement_threshold=cfg.critique_disagreement_threshold,
        )
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
            critique_feedback = _summarise_critique_for_supervisor(critique)
            if cfg.enable_plan_revision and supplemental_loops == 1:
                revised_projection = format_projection(
                    project_ledger(
                        managed_ledger.ledger,
                        idx,
                        pinned_ids=pinned_ids,
                        window_iterations=cfg.ledger_window_iterations,
                    )
                )
                try:
                    replan_h = run_plan_revision.submit(
                        brief,
                        plan,
                        critique,
                        revised_projection,
                        gen_model,
                    )
                    all_handles.append(replan_h)
                    plan = replan_h.load()
                    if plan != original_plan:
                        revised_plan = plan
                except Exception as exc:
                    logger.warning(
                        "Plan revision failed (%s); continuing with original plan",
                        exc,
                    )
            record, _conv, _decision, iter_handles = _run_iteration(
                cfg,
                brief,
                plan,
                managed_ledger,
                idx,
                stamp.started_at,
                tools=tools,
                pinned_ids=pinned_ids,
                critique_feedback=critique_feedback,
            )
            all_handles.extend(iter_handles)
            iterations.append(record)
            ledger = managed_ledger.ledger

            draft_h = run_draft.submit(brief, plan, ledger, gen_model)
            all_handles.append(draft_h)
            draft = draft_h.load()

            critique_h = run_critique.submit(
                draft,
                plan,
                ledger,
                reviewer_model,
                second,
                disagreement_threshold=cfg.critique_disagreement_threshold,
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

        verification: VerificationReport | None = None
        if cfg.enable_verification:
            try:
                verify_h = run_verify.submit(
                    final_report or draft,
                    ledger,
                    cfg.slots["reviewer"].model_string,
                )
                all_handles.append(verify_h)
                verification = verify_h.load()
            except Exception as verify_err:
                logger.warning(
                    "Verification phase failed: %s — continuing without verification",
                    verify_err,
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
            plan=original_plan,
            ledger=ledger,
            iterations=iterations,
            draft=draft,
            critique=critique,
            final_report=final_report,
            tool_provider_manifest=tool_provider_manifest,
            revised_plan=revised_plan,
            grounding_min_ratio=cfg.grounding_min_ratio,
            strict_grounding=cfg.strict_grounding,
            verification=verification,
            after=all_handles,
        )
        all_handles.append(assemble_h)

        if output_dir is None:
            return assemble_h

        package = assemble_h.load()
        return export_package.submit(package, output_dir, after=all_handles)
    finally:
        reset_active_tracker(tracker_token)
