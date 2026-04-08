from datetime import UTC, datetime

from deep_research.models import InvestigationPackage, RenderPayload


def render_full_report(package: InvestigationPackage) -> RenderPayload:
    """Build the canonical full-report markdown and metadata from package state."""
    selection_keys = [item.candidate_key for item in package.selection_graph.items]
    citation_map = {
        f"[{index}]": item.candidate_key
        for index, item in enumerate(package.selection_graph.items, start=1)
    }
    lines = [
        "# Full Report",
        "",
        f"Brief: {package.run_summary.brief}",
        "",
        f"Goal: {package.research_plan.goal}",
        "",
        "## Selected Evidence",
    ]
    for index, item in enumerate(package.selection_graph.items, start=1):
        lines.append(f"- [{index}] {item.candidate_key}: {item.rationale}")
    return RenderPayload(
        name="full_report",
        content_markdown="\n".join(lines),
        structured_content={
            "run_id": package.run_summary.run_id,
            "selected_count": len(package.selection_graph.items),
            "selection_keys": selection_keys,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
