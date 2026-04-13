"""Deliverable rendering: reading path, backing report, and full report."""

from __future__ import annotations

from deep_research.enums import DeliverableMode
from deep_research.flow._types import (
    IterationLoopOutput,
    RunState,
    _FULLY_SUPPORTED_MODES,
    _flow,
)
from deep_research.models import (
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
)


def render_deliverable(
    selection: SelectionGraph,
    iteration_output: IterationLoopOutput,
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> tuple[list[RenderPayload], float, list[str]]:
    """Render deliverable(s) and return ``(renders, added_cost, degradations)``."""
    flow = _flow()
    config = run_state.config
    deliverable_mode = run_state.preferences.deliverable_mode
    degradations: list[str] = []

    if deliverable_mode == DeliverableMode.RESEARCH_PACKAGE:
        reading_future = flow.write_reading_path.submit(
            selection,
            iteration_output.ledger,
            plan,
            config,
            preferences=run_state.preferences,
        )
        backing_future = flow.write_backing_report.submit(
            selection,
            iteration_output.ledger,
            plan,
            IterationTrace(iterations=list(iteration_output.iteration_history)),
            iteration_output.provider_usage_summary,
            iteration_output.stop_reason,
            config,
            preferences=run_state.preferences,
        )
        reading_result = reading_future.load()
        backing_result = backing_future.load()
        added_cost = (
            reading_result.budget.estimated_cost_usd
            + backing_result.budget.estimated_cost_usd
        )
        return [reading_result.render, backing_result.render], added_cost, degradations

    # Non-default modes fall through to a single full-report render. Reuse the
    # run stamp so the scaffold RunSummary shares run_id with the final one.
    partial_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="rendering",
        estimated_cost_usd=round(iteration_output.spent_usd, 6),
        elapsed_seconds=0,
        active_elapsed_seconds=iteration_output.active_elapsed_seconds,
        wall_elapsed_seconds=iteration_output.wall_elapsed_seconds,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=None,
    )
    partial_package = InvestigationPackage(
        run_summary=partial_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=[],
        preferences=run_state.preferences,
    )
    full_result = flow.write_full_report.submit(
        partial_package, config, preferences=run_state.preferences
    ).load()
    if deliverable_mode not in _FULLY_SUPPORTED_MODES:
        degradations.append(
            f"Deliverable mode '{deliverable_mode.value}' not yet fully supported"
            f" -- rendered as 'final_report' with {deliverable_mode.value} context"
        )
    return [full_result.render], full_result.budget.estimated_cost_usd, degradations
