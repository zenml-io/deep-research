from datetime import UTC, datetime

from kitaru import checkpoint

from deep_research.models import RenderPayload, SelectionGraph


@checkpoint(type="llm_call")
def render_reading_path(selection: SelectionGraph) -> RenderPayload:
    """Checkpoint: render a markdown reading path from the curated selection."""
    lines = ["# Reading Path", ""]
    items: list[dict[str, object]] = []
    citation_map: dict[str, str] = {}
    for index, item in enumerate(selection.items, start=1):
        citation = f"[{index}]"
        lines.append(f"- {citation} {item.candidate_key}: {item.rationale}")
        if item.bridge_note:
            lines.append(f"  Bridge: {item.bridge_note}")
        if item.matched_subtopics:
            lines.append(f"  Subtopics: {', '.join(item.matched_subtopics)}")
        if item.ordering_rationale:
            lines.append(f"  Why here: {item.ordering_rationale}")
        if item.reading_time_minutes is not None:
            lines.append(f"  Reading time: {item.reading_time_minutes} minutes")
        items.append(
            {
                "citation": citation,
                "candidate_key": item.candidate_key,
                "rationale": item.rationale,
                "bridge_note": item.bridge_note,
                "matched_subtopics": item.matched_subtopics,
                "reading_time_minutes": item.reading_time_minutes,
                "ordering_rationale": item.ordering_rationale,
            }
        )
        citation_map[citation] = item.candidate_key
    if selection.gap_coverage_summary:
        lines.extend(
            [
                "",
                f"Uncovered subtopics: {', '.join(selection.gap_coverage_summary)}",
            ]
        )
    return RenderPayload(
        name="reading_path",
        content_markdown="\n".join(lines),
        structured_content={
            "items": items,
            "gap_coverage_summary": selection.gap_coverage_summary,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
