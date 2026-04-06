from kitaru import checkpoint

from deep_research.models import RenderPayload, SelectionGraph


@checkpoint(type="llm_call")
def render_reading_path(selection: SelectionGraph) -> RenderPayload:
    """Checkpoint: render a markdown reading path from the curated selection."""
    lines = ["# Reading Path", ""]
    for item in selection.items:
        lines.append(f"- {item.candidate_key}: {item.rationale}")
    return RenderPayload(name="reading_path", content_markdown="\n".join(lines))
