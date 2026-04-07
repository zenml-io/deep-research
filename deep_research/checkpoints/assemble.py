from kitaru import checkpoint

from deep_research.models import (
    CoherenceResult,
    CritiqueResult,
    EvidenceLedger,
    GroundingResult,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
)
from deep_research.package.assembly import assemble_package as _assemble_package


@checkpoint(type="tool_call")
def assemble_package(
    run_summary: RunSummary,
    research_plan: ResearchPlan,
    evidence_ledger: EvidenceLedger,
    selection_graph: SelectionGraph,
    iteration_trace: IterationTrace,
    renders: list[RenderPayload],
    critique_result: CritiqueResult | None = None,
    grounding_result: GroundingResult | None = None,
    coherence_result: CoherenceResult | None = None,
) -> InvestigationPackage:
    """Checkpoint: combine all research artifacts into the final investigation package."""
    return _assemble_package(
        run_summary=run_summary,
        research_plan=research_plan,
        evidence_ledger=evidence_ledger,
        selection_graph=selection_graph,
        iteration_trace=iteration_trace,
        renders=renders,
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
    )
