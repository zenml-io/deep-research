"""Default flow orchestration for deep research.

Thin @flow entry point (~130 lines) that delegates ALL logic to checkpoints.
Pipeline phases:
  1. scope -> ResearchBrief
  2. plan  -> ResearchPlan
  3. Iteration loop: supervisor -> subagents -> merge -> convergence
  4. draft -> critique -> (optional supplemental loop) -> finalize -> assemble
"""

import logging

from kitaru import flow

# Import all checkpoints at module top level (enables monkeypatching in tests)
from research.checkpoints.assemble import assemble_package
from research.checkpoints.critique import run_critique
from research.checkpoints.draft import run_draft
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
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import InvestigationPackage, RunMetadata
from research.flows.convergence import check_convergence
from research.ledger.ledger import ManagedLedger
from research.ledger.projection import format_projection, project_ledger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class FlowTimeoutError(Exception):
    """Raised when a flow-level wait() exceeds config.wait_timeout_seconds."""


class SupervisorError(Exception):
    """Raised when the supervisor fails twice consecutively.

    Carries the last valid ledger state so operators can inspect progress
    and replay from the last good checkpoint.
    """

    def __init__(self, message: str, ledger: EvidenceLedger | None = None):
        super().__init__(message)
        self.ledger = ledger


class FinalizerError(Exception):
    """Raised when the finalizer fails and allow_unfinalized_package is False.

    Draft and critique are preserved in the flow's checkpoint history for
    replay after the underlying issue is fixed.
    """


def _fan_out_subagents(tasks, model_name: str, max_parallel: int):
    """Fan-out subagent tasks in batches of max_parallel, never spilling across iterations.

    Returns (results, submit_handles) — callers need the handles for DAG edge tracking.
    """
    results = []
    all_handles = []
    for batch_start in range(0, len(tasks), max_parallel):
        batch = tasks[batch_start : batch_start + max_parallel]
        handles = [run_subagent.submit(t, model_name) for t in batch]
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
):
    """Call run_supervisor with one retry on failure.

    On first failure: log warning, retry once.
    On second failure: raise SupervisorError with the last valid ledger.

    Each attempt uses a distinct checkpoint ``id`` so Kitaru does not
    serve a cached failure for the retry.

    Returns (decision, submit_handles) — callers need the handles for DAG edge tracking.
    """
    handles: list = []
    try:
        h = run_supervisor.submit(
            brief,
            plan,
            projection,
            remaining,
            iteration_index,
            gen_model,
            breadth_first=breadth_first,
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
    brief,
    plan,
    managed_ledger: ManagedLedger,
    iteration_index: int,
    started_at: str,
):
    """Execute one research iteration: supervisor -> subagents -> merge -> record.

    Returns (record, convergence, submit_handles) — callers need the handles for DAG edge tracking.
    """
    iter_handles: list = []

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
    )
    iter_handles.extend(sup_handles)

    # Fan-out subagents (batched by max_parallel_subagents)
    subagent_results, sub_handles = _fan_out_subagents(
        decision.subagent_tasks, sub_model, cfg.max_parallel_subagents
    )
    iter_handles.extend(sub_handles)

    # Merge subagent findings into the evidence ledger
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

    # Check convergence after iteration completes
    convergence = check_convergence(
        budget=cfg.budget,
        elapsed_seconds=clock.elapsed_seconds,
        time_limit_seconds=cfg.wait_timeout_seconds,
        supervisor_decision=decision,
        iteration_index=iteration_index,
        max_iterations=cfg.max_iterations,
        respect_supervisor_done=cfg.respect_supervisor_done,
    )

    return record, convergence, iter_handles


@flow
def deep_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    """Orchestrate a full deep-research investigation.

    Thin flow — all logic lives in checkpoints and pure helpers.

    All submit handles are collected into ``all_handles`` and passed as
    ``after=`` to the final ``assemble_package.submit()`` call.  This
    creates explicit DAG edges so that Kitaru's ``.wait()`` finds exactly
    one terminal step and extracts the ``InvestigationPackage`` correctly.
    """
    cfg = config or ResearchConfig.for_tier(tier)
    all_handles: list = []

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

    # Phase 3: Iteration loop
    managed_ledger = ManagedLedger()
    iterations: list[IterationRecord] = []
    stop_reason: str | None = None

    for i in range(cfg.max_iterations):
        record, convergence, iter_handles = _run_iteration(
            cfg, brief, plan, managed_ledger, i, stamp.started_at
        )
        all_handles.extend(iter_handles)
        iterations.append(record)

        if convergence.should_stop:
            stop_reason = convergence.reason.value
            break

    ledger = managed_ledger.ledger

    # Phase 4: Draft
    draft_h = run_draft.submit(brief, plan, ledger, gen_model)
    all_handles.append(draft_h)
    draft = draft_h.load()

    # Phase 5: Critique
    reviewer_model = cfg.slots["reviewer"].model_string
    second = cfg.second_reviewer.model_string if cfg.second_reviewer else None
    critique_h = run_critique.submit(draft, reviewer_model, second)
    all_handles.append(critique_h)
    critique = critique_h.load()

    # Phase 6: Supplemental loop (capped at max_supplemental_loops)
    supplemental_loops = 0
    while (
        critique.require_more_research
        and supplemental_loops < cfg.max_supplemental_loops
    ):
        supplemental_loops += 1
        idx = len(iterations)
        record, _conv, iter_handles = _run_iteration(
            cfg, brief, plan, managed_ledger, idx, stamp.started_at
        )
        all_handles.extend(iter_handles)
        iterations.append(record)
        ledger = managed_ledger.ledger

        draft_h = run_draft.submit(brief, plan, ledger, gen_model)
        all_handles.append(draft_h)
        draft = draft_h.load()

        critique_h = run_critique.submit(draft, reviewer_model, second)
        all_handles.append(critique_h)
        critique = critique_h.load()
        # Second require_more_research is recorded but the while guard
        # stops because supplemental_loops == max_supplemental_loops.

    # Phase 7: Finalize
    try:
        finalize_h = run_finalize.submit(draft, critique, gen_model, stop_reason)
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

    # Phase 8: Assemble
    fin_h = finalize_run_metadata.submit(stamp.started_at)
    all_handles.append(fin_h)
    fin = fin_h.load()

    run_metadata = RunMetadata(
        run_id=stamp.run_id,
        tier=cfg.tier,
        started_at=stamp.started_at,
        completed_at=fin.completed_at,
        total_cost_usd=cfg.budget.spent_usd,
        total_iterations=len(iterations),
        stop_reason=stop_reason,
    )

    # Return the submit handle (without .load()) so Kitaru's DAG registers
    # this checkpoint as the sole terminal step.  The ``after=all_handles``
    # creates explicit dependency edges from every prior checkpoint to this
    # one, ensuring all other steps appear in upstream_steps and only
    # assemble_package is terminal.  This lets .wait() extract the result.
    return assemble_package.submit(
        metadata=run_metadata,
        brief=brief,
        plan=plan,
        ledger=ledger,
        iterations=iterations,
        draft=draft,
        critique=critique,
        final_report=final_report,
        grounding_min_ratio=cfg.grounding_min_ratio,
        after=all_handles,
    )
