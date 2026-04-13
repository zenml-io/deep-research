"""Final package assembly: build RunSummary and hand off to the checkpoint."""

from __future__ import annotations

from deep_research.config import ResearchConfig
from deep_research.flow._types import (
    IterationLoopOutput,
    RunState,
    _flow,
)
from deep_research.models import (
    ClaimInventory,
    CoherenceResult,
    CritiqueResult,
    GroundingResult,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    RenderSettingsSnapshot,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
)


def assemble_final_package(
    *,
    stamp,
    finalization,
    run_state: RunState,
    plan: ResearchPlan,
    iteration_output: IterationLoopOutput,
    selection: SelectionGraph,
    renders: list[RenderPayload],
    spent_usd: float,
    critique_result: CritiqueResult | None,
    grounding_result: GroundingResult | None,
    coherence_result: CoherenceResult | None,
    degradations: list[str],
    claim_inventory: ClaimInventory | None = None,
) -> InvestigationPackage:
    """Build the final RunSummary and hand everything to the assemble checkpoint."""
    flow = _flow()
    config = run_state.config
    run_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="completed",
        estimated_cost_usd=round(spent_usd, 6),
        elapsed_seconds=finalization.elapsed_seconds,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=finalization.completed_at,
    )
    run_summary = run_summary.model_copy(
        update={
            "active_elapsed_seconds": iteration_output.active_elapsed_seconds,
            "wall_elapsed_seconds": finalization.elapsed_seconds,
        }
    )
    return flow.assemble_package.submit(
        run_summary=run_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=renders,
        render_settings=RenderSettingsSnapshot(writer_model=config.writer_model),
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
        preferences=run_state.preferences,
        preference_degradations=degradations,
        claim_inventory=claim_inventory,
    ).load()
