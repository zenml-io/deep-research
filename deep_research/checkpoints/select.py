from __future__ import annotations

from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig, SelectionPolicyConfig
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.evidence.scoring import combined_candidate_score, infer_source_group
from deep_research.models import (
    EvidenceCandidate,
    EvidenceLedger,
    ResearchPlan,
    SelectionGraph,
    SelectionItem,
)
from deep_research.observability import span, warning


def _gap_summary(
    selected_entries: list[EvidenceCandidate],
    plan: ResearchPlan,
) -> list[str]:
    """Return plan subtopics not covered by any selected entry."""
    covered: set[str] = set()
    for candidate in selected_entries:
        if candidate.matched_subtopics:
            covered.update(subtopic.lower() for subtopic in candidate.matched_subtopics)
            continue
        entry_text = " ".join(
            part.strip().lower()
            for part in [candidate.title, *(s.text for s in candidate.snippets)]
            if part.strip()
        )
        covered.update(
            subtopic.strip().lower()
            for subtopic in plan.subtopics
            if subtopic.strip().lower() in entry_text
        )
    return [
        subtopic
        for subtopic in plan.subtopics
        if subtopic.strip().lower() not in covered
    ]


def _selection_item(candidate: EvidenceCandidate) -> SelectionItem:
    """Build a SelectionItem from a ranked candidate."""
    return SelectionItem(
        candidate_key=candidate.key,
        rationale=(
            f"Selected for {candidate.provider} relevance={candidate.relevance_score:.2f}, "
            f"authority={candidate.authority_score:.2f}, "
            f"selection={candidate.selection_score:.2f}."
        ),
        bridge_note=(
            f"Bridges {', '.join(candidate.matched_subtopics)}."
            if candidate.matched_subtopics
            else None
        ),
        matched_subtopics=candidate.matched_subtopics,
        reading_time_minutes=len(candidate.snippets) * 3 or None,
        ordering_rationale=(
            "MMR greedy selection balances relevance against content and source-group diversity."
        ),
    )


def _build_deterministic_selection_graph(
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig | None = None,
) -> SelectionGraph:
    """Rank selected ledger entries deterministically via MMR and wrap as a SelectionGraph."""
    policy = config.selection_policy if config is not None else None
    selected_entries = _rank_candidates(
        resolve_selected_entries(ledger), plan, policy=policy
    )
    return SelectionGraph(
        items=[_selection_item(candidate) for candidate in selected_entries],
        gap_coverage_summary=_gap_summary(selected_entries, plan),
    )


def _validate_curator_graph(
    graph: SelectionGraph,
    *,
    selected_entries: list[EvidenceCandidate],
) -> None:
    """Raise if curator output references unknown keys, duplicates keys, or drops all items."""
    valid_keys = {candidate.key for candidate in selected_entries}
    returned_keys = [item.candidate_key for item in graph.items]
    for key in returned_keys:
        if key not in valid_keys:
            raise ValueError(f"unknown curator candidate key: {key}")
    if len(returned_keys) != len(set(returned_keys)):
        raise ValueError("curator returned duplicate candidate keys")
    if selected_entries and not graph.items:
        raise ValueError("curator returned no items for non-empty selection")


def _candidate_token_set(candidate: EvidenceCandidate) -> set[str]:
    """Cheap token bag from title + first snippets for similarity scoring."""
    parts = [candidate.title, *(snippet.text for snippet in candidate.snippets[:3])]
    tokens: set[str] = set()
    for part in parts:
        for token in part.lower().split():
            cleaned = token.strip(".,;:()[]\"'`")
            if len(cleaned) > 2:
                tokens.add(cleaned)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _max_similarity(candidate: EvidenceCandidate, selected: list[EvidenceCandidate]) -> float:
    """Max jaccard similarity of candidate against any already-selected item."""
    if not selected:
        return 0.0
    candidate_tokens = _candidate_token_set(candidate)
    return max(
        _jaccard(candidate_tokens, _candidate_token_set(other)) for other in selected
    )


def _source_type_concentration(
    candidate: EvidenceCandidate, selected: list[EvidenceCandidate]
) -> float:
    """Fraction of already-selected items sharing this candidate's source group."""
    if not selected:
        return 0.0
    candidate_group = infer_source_group(candidate)
    same_group = sum(
        1 for other in selected if infer_source_group(other) == candidate_group
    )
    return same_group / len(selected)


def _mmr_score(
    candidate: EvidenceCandidate,
    base_scores: dict[str, float],
    ordered: list[EvidenceCandidate],
    policy: SelectionPolicyConfig,
) -> float:
    """MMR score for a candidate given already-selected items and policy lambdas."""
    return (
        policy.mmr_relevance_lambda * base_scores[candidate.key]
        - policy.mmr_diversity_lambda * _max_similarity(candidate, ordered)
        - policy.mmr_source_type_lambda * _source_type_concentration(candidate, ordered)
    )


def _rank_candidates(
    candidates,
    plan: ResearchPlan,
    policy: SelectionPolicyConfig | None = None,
    target_selected: int | None = None,
    mmr_floor: float | None = None,
):
    """Greedy MMR selection over already-selected candidates.

    mmr(d) = lambda1 * combined_score(d)
           - lambda2 * max_similarity(d, already_selected)
           - lambda3 * source_type_concentration(source_group(d), already_selected)
    """
    if not candidates:
        return []
    policy = policy or SelectionPolicyConfig()
    pool = list(candidates)
    base_scores = {
        candidate.key: combined_candidate_score(candidate, plan=plan)
        for candidate in pool
    }
    ordered: list[EvidenceCandidate] = []
    cap = target_selected if target_selected is not None else len(pool)

    while pool and len(ordered) < cap:
        # Stable tie-break by candidate key (ascending).
        pool.sort(key=lambda c: (-_mmr_score(c, base_scores, ordered, policy), c.key))
        best = pool[0]
        if mmr_floor is not None and _mmr_score(best, base_scores, ordered, policy) < mmr_floor:
            break
        ordered.append(best)
        pool.pop(0)
    return ordered


def _curator_prompt_payload(
    plan: ResearchPlan, selected_entries: list[EvidenceCandidate]
) -> dict:
    """Build the structured prompt payload fed to the curator agent."""
    return {
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


@checkpoint(type="tool_call")
def rank_evidence(
    ledger: EvidenceLedger, plan: ResearchPlan, config: ResearchConfig | None = None
) -> SelectionGraph:
    """Checkpoint: build a deterministic reading order from selected evidence."""
    with span("rank_evidence"):
        deterministic = _build_deterministic_selection_graph(ledger, plan, config)
        if config is None or not config.use_curator_for_selection:
            return deterministic

        selected_entries = resolve_selected_entries(ledger)
        if not selected_entries:
            return deterministic

        from deep_research.agents.curator import build_curator_agent

        try:
            result = build_curator_agent(config.curator_model).run_sync(
                serialize_prompt_payload(
                    _curator_prompt_payload(plan, selected_entries),
                    label="curator prompt",
                )
            )
            _validate_curator_graph(result.output, selected_entries=selected_entries)
            return result.output
        except Exception as exc:
            warning(
                "Curator selection failed; falling back to deterministic ordering",
                error=str(exc),
            )
            return deterministic
