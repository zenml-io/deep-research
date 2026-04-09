from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.models import (
    EvidenceLedger,
    ResearchPlan,
    SelectionGraph,
    SelectionItem,
)


@checkpoint(type="tool_call")
def rank_evidence(
    ledger: EvidenceLedger, plan: ResearchPlan, config: ResearchConfig | None = None
) -> SelectionGraph:
    """Checkpoint: build a deterministic reading order from selected evidence."""
    _ = config

    selected_entries = sorted(
        resolve_selected_entries(ledger),
        key=lambda candidate: (
            -candidate.quality_score,
            -candidate.authority_score,
            -candidate.relevance_score,
            candidate.key,
        ),
    )
    items = [
        SelectionItem(
            candidate_key=candidate.key,
            rationale=f"Selected for {candidate.provider} quality and relevance.",
            bridge_note=(
                f"Bridges {', '.join(candidate.matched_subtopics)}."
                if candidate.matched_subtopics
                else None
            ),
            matched_subtopics=candidate.matched_subtopics,
            reading_time_minutes=len(candidate.snippets) * 3 or None,
            ordering_rationale="Higher quality, authority, and relevance sources appear earlier.",
        )
        for candidate in selected_entries
    ]
    covered = set()
    entry_text = {
        candidate.key: " ".join(
            part.strip().lower()
            for part in [
                candidate.title,
                *(snippet.text for snippet in candidate.snippets),
            ]
            if part.strip()
        )
        for candidate in selected_entries
    }
    for candidate in selected_entries:
        if candidate.matched_subtopics:
            covered.update(subtopic.lower() for subtopic in candidate.matched_subtopics)
            continue
        covered.update(
            subtopic.strip().lower()
            for subtopic in plan.subtopics
            if subtopic.strip().lower() in entry_text[candidate.key]
        )
    return SelectionGraph(
        items=items,
        gap_coverage_summary=[
            subtopic
            for subtopic in plan.subtopics
            if subtopic.strip().lower() not in covered
        ],
    )
