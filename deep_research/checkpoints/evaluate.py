from kitaru import checkpoint

from deep_research.evidence.resolution import resolve_coverage_entries
from deep_research.models import CoverageScore, EvidenceLedger, ResearchPlan


@checkpoint(type="tool_call")
def evaluate_coverage(ledger: EvidenceLedger, plan: ResearchPlan) -> CoverageScore:
    """Checkpoint: compute subtopic, diversity, and density coverage scores."""
    entries = resolve_coverage_entries(ledger)
    if not entries:
        return CoverageScore(
            subtopic_coverage=0.0,
            source_diversity=0.0,
            evidence_density=0.0,
            total=0.0,
            uncovered_subtopics=plan.subtopics,
        )

    entry_text = []
    for entry in entries:
        snippet_text = [snippet.text for snippet in entry.snippets]
        entry_text.append(
            " ".join(
                part.strip().lower()
                for part in [entry.title, *snippet_text]
                if part.strip()
            )
        )

    uncovered_subtopics = []
    for subtopic in plan.subtopics:
        token = subtopic.strip().lower()
        is_covered = token and any(
            token in {matched.lower() for matched in entry.matched_subtopics}
            or token in text
            for entry, text in zip(entries, entry_text, strict=False)
        )
        if not is_covered:
            uncovered_subtopics.append(subtopic)

    covered_subtopics = len(plan.subtopics) - len(uncovered_subtopics)

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
        uncovered_subtopics=uncovered_subtopics,
    )
