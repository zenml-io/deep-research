from datetime import UTC, datetime

from deep_research.models import (
    EvidenceLedger,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
    StopReason,
)


def render_backing_report(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    iteration_trace: IterationTrace,
    provider_usage_summary: dict[str, int],
    stop_reason: StopReason,
) -> RenderPayload:
    """Build a deterministic backing-report scaffold from research state."""
    citation_map = {
        f"[{index}]": item.candidate_key
        for index, item in enumerate(selection.items, start=1)
    }
    return RenderPayload(
        name="backing_report",
        content_markdown="",
        structured_content={
            "goal": plan.goal,
            "selection_items": [
                item.model_dump(mode="json") for item in selection.items
            ],
            "selected_candidates": [
                candidate.model_dump(mode="json") for candidate in ledger.selected
            ],
            "rejected_candidates": [
                candidate.model_dump(mode="json") for candidate in ledger.rejected
            ],
            "iterations": [
                iteration.model_dump(mode="json")
                for iteration in iteration_trace.iterations
            ],
            "provider_usage_summary": provider_usage_summary,
            "stop_reason": stop_reason.value,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
