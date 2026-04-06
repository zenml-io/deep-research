from kitaru import checkpoint

from deep_research.models import (
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)


@checkpoint(type="llm_call")
def render_backing_report(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> RenderPayload:
    del ledger
    return RenderPayload(
        name="backing_report",
        content_markdown=(
            f"# Backing Report\n\nGoal: {plan.goal}\n\nSelected: {len(selection.items)}"
        ),
    )
