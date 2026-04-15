"""Default flow orchestration for deep research.

Thin @flow entry point (~130 lines) that delegates ALL logic to checkpoints.
Pipeline phases:
  1. scope -> ResearchBrief
  2. plan  -> ResearchPlan
  3. Iteration loop: supervisor -> fan-out subagents -> merge -> convergence
  4. draft -> critique -> (optional supplemental loop) -> finalize -> assemble
"""

from __future__ import annotations

from kitaru import flow, wait

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


class FlowTimeoutError(Exception):
    """Raised when a flow-level wait() exceeds config.wait_timeout_seconds."""


def _fan_out_subagents(tasks, model_name: str, max_parallel: int):
    """Fan-out subagent tasks in batches of max_parallel, never spilling across iterations."""
    results = []
    for batch_start in range(0, len(tasks), max_parallel):
        batch = tasks[batch_start : batch_start + max_parallel]
        handles = [run_subagent.submit(t, model_name) for t in batch]
        results.extend(h.load() for h in handles)
    return results


def _run_iteration(
    cfg: ResearchConfig,
    brief,
    plan,
    ledger: EvidenceLedger,
    iteration_index: int,
    started_at: str,
):
    """Execute one research iteration: supervisor -> subagents -> record."""
    clock = snapshot_wall_clock(started_at)

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

    decision = run_supervisor(
        brief, plan, projection, remaining, iteration_index, gen_model
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
    brief = run_scope(question, gen_model)

    # Phase 2: Plan
    plan = run_plan(brief, gen_model)

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
    draft = run_draft(brief, plan, ledger, gen_model)

    # Phase 5: Critique
    reviewer_model = cfg.slots["reviewer"].model_string
    second = cfg.second_reviewer.model_string if cfg.second_reviewer else None
    critique = run_critique(draft, reviewer_model, second)

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
        draft = run_draft(brief, plan, ledger, gen_model)
        critique = run_critique(draft, reviewer_model, second)
        # Second require_more_research is recorded but the while guard
        # stops because supplemental_loops == max_supplemental_loops.

    # Phase 7: Finalize
    final_report = run_finalize(draft, critique, gen_model)

    # Phase 8: Assemble
    fin = finalize_run_metadata(stamp.started_at)
    run_metadata = RunMetadata(
        run_id=stamp.run_id,
        tier=cfg.tier,
        started_at=stamp.started_at,
        completed_at=fin.completed_at,
        total_cost_usd=cfg.budget.spent_usd,
        total_iterations=len(iterations),
        stop_reason=stop_reason,
    )

    package = assemble_package(
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
    return package
