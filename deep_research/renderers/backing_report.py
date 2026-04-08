from datetime import UTC, datetime

from deep_research.models import (
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)


def render_backing_report(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> RenderPayload:
    """Build the backing-report markdown and metadata from the plan and selected evidence."""
    del ledger
    selection_keys = [item.candidate_key for item in selection.items]
    citation_map = {
        f"[{index}]": item.candidate_key
        for index, item in enumerate(selection.items, start=1)
    }
    selected_lines = [
        f"- [{index}] {item.candidate_key}: {item.rationale}"
        for index, item in enumerate(selection.items, start=1)
    ]
    content_markdown = (
        f"# Backing Report\n\nGoal: {plan.goal}\n\nSelected: {len(selection.items)}"
    )
    if selected_lines:
        content_markdown = f"{content_markdown}\n\n## Selected Evidence\n" + "\n".join(
            selected_lines
        )
    return RenderPayload(
        name="backing_report",
        content_markdown=content_markdown,
        structured_content={
            "goal": plan.goal,
            "selected_count": len(selection.items),
            "selection_keys": selection_keys,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
