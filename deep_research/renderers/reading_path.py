from datetime import UTC, datetime

from deep_research.models import (
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)


def render_reading_path(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> RenderPayload:
    """Build a deterministic reading-path scaffold from selection and ledger data."""
    items: list[dict[str, object]] = []
    citation_map: dict[str, str] = {}
    candidates_by_key = {candidate.key: candidate for candidate in ledger.entries}

    for index, item in enumerate(selection.items, start=1):
        citation = f"[{index}]"
        candidate = candidates_by_key.get(item.candidate_key)
        items.append(
            {
                "citation": citation,
                "candidate_key": item.candidate_key,
                "title": candidate.title if candidate else item.candidate_key,
                "url": str(candidate.url) if candidate else None,
                "provider": candidate.provider if candidate else None,
                "rationale": item.rationale,
                "bridge_note": item.bridge_note,
                "matched_subtopics": item.matched_subtopics,
                "ordering_rationale": item.ordering_rationale,
                "snippets": [
                    snippet.text
                    for snippet in (candidate.snippets[:2] if candidate else [])
                ],
            }
        )
        citation_map[citation] = item.candidate_key

    return RenderPayload(
        name="reading_path",
        content_markdown="",
        structured_content={
            "goal": plan.goal,
            "key_questions": plan.key_questions,
            "items": items,
            "gap_coverage_summary": selection.gap_coverage_summary,
        },
        citation_map=citation_map,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
