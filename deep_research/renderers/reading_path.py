from datetime import UTC, datetime

from kitaru import checkpoint

from deep_research.models import RenderPayload, SelectionGraph


@checkpoint(type="llm_call")
def render_reading_path(selection: SelectionGraph) -> RenderPayload:
    """Checkpoint: render a markdown reading path from the curated selection."""
    lines = ["# Reading Path", ""]
    items: list[dict[str, str]] = []
    citation_map: dict[str, str] = {}
    for index, item in enumerate(selection.items, start=1):
        citation = f"[{index}]"
        lines.append(f"- {citation} {item.candidate_key}: {item.rationale}")
        items.append(
            {
                "citation": citation,
                "candidate_key": item.candidate_key,
                "rationale": item.rationale,
            }
        )
        citation_map[citation] = item.candidate_key
    return RenderPayload(
        name="reading_path",
        content_markdown="\n".join(lines),
        structured_content={"items": items},
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
