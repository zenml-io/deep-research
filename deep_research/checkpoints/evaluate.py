from kitaru import checkpoint

from deep_research.models import CoverageScore, EvidenceLedger, ResearchPlan


def _text_for_candidate(candidate_text_parts: list[str]) -> str:
    """Join and lowercase text parts for substring coverage matching."""
    return " ".join(
        part.strip().lower() for part in candidate_text_parts if part.strip()
    )


@checkpoint(type="tool_call")
def evaluate_coverage(ledger: EvidenceLedger, plan: ResearchPlan) -> CoverageScore:
    """Checkpoint: compute subtopic, diversity, and density coverage scores."""
    entries = ledger.entries
    if not entries:
        return CoverageScore(
            subtopic_coverage=0.0,
            source_diversity=0.0,
            evidence_density=0.0,
            total=0.0,
        )

    entry_text = []
    for entry in entries:
        snippet_text = [snippet.text for snippet in entry.snippets]
        entry_text.append(_text_for_candidate([entry.title, *snippet_text]))

    covered_subtopics = 0
    for subtopic in plan.subtopics:
        token = subtopic.strip().lower()
        if token and any(token in text for text in entry_text):
            covered_subtopics += 1

    subtopic_coverage = (
        covered_subtopics / len(plan.subtopics) if plan.subtopics else 1.0
    )
    source_diversity = min(len({entry.provider for entry in entries}) / 3.0, 1.0)
    evidence_density = min(len(entries) / max(len(plan.key_questions), 1), 1.0)
    total = round(
        (subtopic_coverage + source_diversity + evidence_density) / 3.0,
        4,
    )

    return CoverageScore(
        subtopic_coverage=round(subtopic_coverage, 4),
        source_diversity=round(source_diversity, 4),
        evidence_density=round(evidence_density, 4),
        total=total,
    )
