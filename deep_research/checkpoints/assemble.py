from kitaru import checkpoint

from deep_research.models import (
    CoherenceResult,
    CritiqueResult,
    EvidenceLedger,
    GroundingResult,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    RenderSettingsSnapshot,
    ResearchPlan,
    ResearchPreferences,
    RunSummary,
    SelectionGraph,
)


@checkpoint(type="tool_call")
def assemble_package(
    run_summary: RunSummary,
    research_plan: ResearchPlan,
    evidence_ledger: EvidenceLedger,
    selection_graph: SelectionGraph,
    iteration_trace: IterationTrace,
    renders: list[RenderPayload],
    render_settings: RenderSettingsSnapshot | None = None,
    critique_result: CritiqueResult | None = None,
    grounding_result: GroundingResult | None = None,
    coherence_result: CoherenceResult | None = None,
    preferences: ResearchPreferences | None = None,
    preference_degradations: list[str] | None = None,
) -> InvestigationPackage:
    """Checkpoint: combine all research artifacts into the final investigation package."""
    return InvestigationPackage(
        run_summary=run_summary,
        research_plan=research_plan,
        evidence_ledger=evidence_ledger,
        selection_graph=selection_graph,
        iteration_trace=iteration_trace,
        renders=renders,
        render_settings=render_settings,
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
        preferences=preferences,
        preference_degradations=preference_degradations or [],
    )
