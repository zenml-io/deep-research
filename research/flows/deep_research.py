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
    """Fan-out subagent tasks in batches of max_parallel, never spilling across iterations."""
    results = []
    for batch_start in range(0, len(tasks), max_parallel):
        batch = tasks[batch_start : batch_start + max_parallel]
        handles = [run_subagent.submit(t, model_name) for t in batch]
        results.extend(h.load() for h in handles)
    return results


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
    """
    try:
        return run_supervisor.submit(
            brief,
            plan,
            projection,
            remaining,
            iteration_index,
            gen_model,
            breadth_first=breadth_first,
            id=f"supervisor_{iteration_index}_a1",
        ).load()
    except Exception as first_err:
        logger.warning(
            "Supervisor failed (attempt 1/2) at iteration %d: %s",
            iteration_index,
            first_err,
        )
        try:
            return run_supervisor.submit(
                brief,
                plan,
                projection,
                remaining,
                iteration_index,
                gen_model,
                breadth_first=breadth_first,
                id=f"supervisor_{iteration_index}_a2",
            ).load()
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
    ledger: EvidenceLedger,
    iteration_index: int,
    started_at: str,
):
    """Execute one research iteration: supervisor -> subagents -> record."""
    clock = snapshot_wall_clock.submit(started_at).load()

    gen_model = cfg.slots["generator"].model_string
    sub_model = cfg.slots["subagent"].model_string

    projection = format_projection(
        project_ledger(
            ledger,
            iteration_index,
            window_iterations=cfg.ledger_window_iterations,
        )
    )
    remaining = cfg.budget.soft_budget_usd - cfg.budget.spent_usd

    decision = _run_supervisor_with_retry(
        brief,
        plan,
        projection,
        remaining,
        iteration_index,
        gen_model,
        ledger,
        breadth_first=cfg.breadth_first,
    )

    # Fan-out subagents (batched by max_parallel_subagents)
    subagent_results = _fan_out_subagents(
        decision.subagent_tasks, sub_model, cfg.max_parallel_subagents
    )

    record = IterationRecord(
        iteration_index=iteration_index,
        supervisor_decision=decision,
        subagent_results=subagent_results,
        ledger_size=len(ledger.items),
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

    return record, convergence


@flow
def deep_research(
    question: str,
    tier: str = "standard",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    """Orchestrate a full deep-research investigation.

    Thin flow — all logic lives in checkpoints and pure helpers.
    """
    cfg = config or ResearchConfig.for_tier(tier)
    stamp = stamp_run_metadata.submit().load()

    # Phase 1: Scope
    gen_model = cfg.slots["generator"].model_string
    scope_model = cfg.scope_override.model_string if cfg.scope_override else gen_model
    brief = run_scope.submit(question, scope_model).load()

    # Phase 2: Plan
    plan = run_plan.submit(brief, gen_model).load()

    # Phase 3: Iteration loop
    ledger = EvidenceLedger(items=[])
    iterations: list[IterationRecord] = []
    stop_reason: str | None = None

    for i in range(cfg.max_iterations):
        record, convergence = _run_iteration(
            cfg, brief, plan, ledger, i, stamp.started_at
        )
        iterations.append(record)

        if convergence.should_stop:
            stop_reason = convergence.reason.value
            break

    # Phase 4: Draft
    draft = run_draft.submit(brief, plan, ledger, gen_model).load()

    # Phase 5: Critique
    reviewer_model = cfg.slots["reviewer"].model_string
    second = cfg.second_reviewer.model_string if cfg.second_reviewer else None
    critique = run_critique.submit(draft, reviewer_model, second).load()

    # Phase 6: Supplemental loop (capped at max_supplemental_loops)
    supplemental_loops = 0
    while (
        critique.require_more_research
        and supplemental_loops < cfg.max_supplemental_loops
    ):
        supplemental_loops += 1
        idx = len(iterations)
        record, _conv = _run_iteration(cfg, brief, plan, ledger, idx, stamp.started_at)
        iterations.append(record)
        draft = run_draft.submit(brief, plan, ledger, gen_model).load()
        critique = run_critique.submit(draft, reviewer_model, second).load()
        # Second require_more_research is recorded but the while guard
        # stops because supplemental_loops == max_supplemental_loops.

    # Phase 7: Finalize
    try:
        final_report = run_finalize.submit(draft, critique, gen_model).load()
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
    fin = finalize_run_metadata.submit(stamp.started_at).load()
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
    # this checkpoint as the terminal step.  All intermediate checkpoints
    # use .submit().load() which materialises immediately but severs the
    # DAG edge.  Keeping the final step un-loaded lets .wait() find exactly
    # one terminal step and extract the InvestigationPackage correctly.
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
    )
