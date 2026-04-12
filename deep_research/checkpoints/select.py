from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.models import (
    EvidenceLedger,
    ResearchPlan,
    SelectionGraph,
    SelectionItem,
)
from deep_research.observability import span, warning


def _gap_summary(
    selected_entries,
    plan: ResearchPlan,
) -> list[str]:
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
    return [
        subtopic
        for subtopic in plan.subtopics
        if subtopic.strip().lower() not in covered
    ]


def _build_deterministic_selection_graph(
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> SelectionGraph:
    selected_entries = sorted(
        resolve_selected_entries(ledger),
        key=lambda candidate: (
            -candidate.quality_score,
            -candidate.authority_score,
            -candidate.relevance_score,
            candidate.key,
        ),
    )
    return SelectionGraph(
        items=[
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
        ],
        gap_coverage_summary=_gap_summary(selected_entries, plan),
    )


def _validate_curator_graph(
    graph: SelectionGraph,
    *,
    selected_entries: list,
) -> None:
    valid_keys = {candidate.key for candidate in selected_entries}
    returned_keys: list[str] = []
    for item in graph.items:
        if item.candidate_key not in valid_keys:
            raise ValueError(f"unknown curator candidate key: {item.candidate_key}")
        returned_keys.append(item.candidate_key)
    if len(returned_keys) != len(set(returned_keys)):
        raise ValueError("curator returned duplicate candidate keys")
    if selected_entries and not graph.items:
        raise ValueError("curator returned no items for non-empty selection")


@checkpoint(type="tool_call")
def rank_evidence(
    ledger: EvidenceLedger, plan: ResearchPlan, config: ResearchConfig | None = None
) -> SelectionGraph:
    """Checkpoint: build a deterministic reading order from selected evidence."""
    with span("rank_evidence"):
        deterministic = _build_deterministic_selection_graph(ledger, plan)
        if config is None or not config.use_curator_for_selection:
            return deterministic

        selected_entries = resolve_selected_entries(ledger)
        if not selected_entries:
            return deterministic

        from deep_research.agents.curator import build_curator_agent

        prompt_payload = {
            "plan": {
                "goal": plan.goal,
                "key_questions": plan.key_questions,
                "subtopics": plan.subtopics,
            },
            "selected_evidence": [
                {
                    "candidate_key": candidate.key,
                    "title": candidate.title,
                    "provider": candidate.provider,
                    "source_kind": candidate.source_kind.value,
                    "quality_score": candidate.quality_score,
                    "relevance_score": candidate.relevance_score,
                    "authority_score": candidate.authority_score,
                    "matched_subtopics": candidate.matched_subtopics,
                    "snippet_preview": [snippet.text for snippet in candidate.snippets[:2]],
                }
                for candidate in selected_entries
            ],
        }
        try:
            result = build_curator_agent(config.curator_model).run_sync(
                serialize_prompt_payload(prompt_payload, label="curator prompt")
            )
            _validate_curator_graph(result.output, selected_entries=selected_entries)
            return result.output
        except Exception as exc:
            warning("Curator selection failed; falling back to deterministic ordering", error=str(exc))
            return deterministic
