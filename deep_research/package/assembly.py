from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionGraph,
)


def assemble_package(
    *,
    run_summary: RunSummary,
    research_plan: ResearchPlan,
    evidence_ledger: EvidenceLedger,
    selection_graph: SelectionGraph,
    iteration_trace: IterationTrace,
    renders: list[RenderPayload],
) -> InvestigationPackage:
    return InvestigationPackage(
        run_summary=run_summary,
        research_plan=research_plan,
        evidence_ledger=evidence_ledger,
        selection_graph=selection_graph,
        iteration_trace=iteration_trace,
        renders=renders,
    )
