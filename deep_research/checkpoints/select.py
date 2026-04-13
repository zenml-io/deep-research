from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig, SelectionPolicyConfig
from deep_research.evidence.dedup import is_near_duplicate
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.evidence.scoring import combined_candidate_score, infer_source_group
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
    config: ResearchConfig | None = None,
) -> SelectionGraph:
    policy = config.selection_policy if config is not None else None
    selected_entries = _rank_candidates(
        resolve_selected_entries(ledger), plan, policy=policy
    )
    return SelectionGraph(
        items=[
            SelectionItem(
                candidate_key=candidate.key,
                rationale=(
                    f"Selected for {candidate.provider} relevance={candidate.relevance_score:.2f}, "
                    f"authority={candidate.authority_score:.2f}, "
                    f"selection={float(candidate.raw_metadata.get('selection_score', 0.0)):.2f}."
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


def _similarity(left, right) -> float:
    if is_near_duplicate(left, right):
        return 1.0
    title_similarity = SequenceMatcher(
        a=left.title.strip().lower(),
        b=right.title.strip().lower(),
    ).ratio()
    overlapping_subtopics = len(
        set(subtopic.lower() for subtopic in left.matched_subtopics)
        & set(subtopic.lower() for subtopic in right.matched_subtopics)
    )
    if overlapping_subtopics:
        title_similarity = max(title_similarity, min(1.0, 0.65 + overlapping_subtopics * 0.1))
    return title_similarity


def _coverage_bonus(candidate, covered_subtopics: set[str], plan: ResearchPlan) -> float:
    if not plan.subtopics:
        return 0.0
    missing = {
        subtopic.strip().lower()
        for subtopic in plan.subtopics
        if subtopic.strip().lower() not in covered_subtopics
    }
    if not missing:
        return 0.0
    candidate_subtopics = {subtopic.strip().lower() for subtopic in candidate.matched_subtopics}
    newly_covered = len(candidate_subtopics & missing)
    return min(newly_covered / max(len(plan.subtopics), 1), 1.0)


def _entity_bonus(candidate, plan: ResearchPlan) -> float:
    target_text = " ".join(plan.key_questions + plan.subtopics + plan.sections).lower()
    title_and_snippets = " ".join(
        [
            candidate.title.lower(),
            *(snippet.text.lower() for snippet in candidate.snippets[:2]),
        ]
    )
    target_hits = 0
    for token in {token for token in target_text.split() if len(token) > 4}:
        if token in title_and_snippets:
            target_hits += 1
    return min(target_hits / 6, 1.0)


def _candidate_token_set(candidate) -> set[str]:
    """Cheap token bag from title + snippet text for similarity scoring."""
    parts = [candidate.title]
    parts.extend(snippet.text for snippet in candidate.snippets[:3])
    tokens: set[str] = set()
    for part in parts:
        for token in part.lower().split():
            cleaned = token.strip(".,;:()[]\"'`")
            if len(cleaned) > 2:
                tokens.add(cleaned)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    union = len(left | right)
    return intersection / union if union else 0.0


def _max_similarity(candidate, selected: list) -> float:
    if not selected:
        return 0.0
    candidate_tokens = _candidate_token_set(candidate)
    return max(
        _jaccard(candidate_tokens, _candidate_token_set(other)) for other in selected
    )


def _source_type_concentration(candidate, selected: list) -> float:
    if not selected:
        return 0.0
    candidate_group = infer_source_group(candidate)
    same_group = sum(
        1 for other in selected if infer_source_group(other) == candidate_group
    )
    return same_group / len(selected)


def _rank_candidates(
    candidates,
    plan: ResearchPlan,
    policy: SelectionPolicyConfig | None = None,
    target_selected: int | None = None,
    mmr_floor: float | None = None,
):
    """Greedy MMR selection over already-selected candidates.

    mmr(d) = λ1 * combined_score(d)
           − λ2 * max_similarity(d, already_selected)
           − λ3 * source_type_concentration(source_group(d), already_selected)
    """
    if not candidates:
        return []
    policy = policy or SelectionPolicyConfig()
    lambda_relevance = policy.mmr_relevance_lambda
    lambda_diversity = policy.mmr_diversity_lambda
    lambda_source = policy.mmr_source_type_lambda

    pool = list(candidates)
    base_scores = {
        candidate.key: combined_candidate_score(candidate, plan=plan)
        for candidate in pool
    }
    ordered: list = []
    cap = target_selected if target_selected is not None else len(pool)

    while pool and len(ordered) < cap:
        def mmr_key(candidate):
            base = base_scores[candidate.key]
            diversity_penalty = _max_similarity(candidate, ordered)
            source_penalty = _source_type_concentration(candidate, ordered)
            score = (
                lambda_relevance * base
                - lambda_diversity * diversity_penalty
                - lambda_source * source_penalty
            )
            # Stable tie-break by candidate key (ascending).
            return (-score, candidate.key)

        pool.sort(key=mmr_key)
        best = pool[0]
        best_score = -mmr_key(best)[0]
        if mmr_floor is not None and best_score < mmr_floor:
            break
        ordered.append(best)
        pool.pop(0)
    return ordered


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
