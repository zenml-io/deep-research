import json
from collections import Counter

from deep_research.evidence.dedup import match_precedence_keys
from deep_research.evidence.resolution import resolve_coverage_entries, resolve_selected_entries
from deep_research.evidence.scoring import (
    combined_candidate_score,
    infer_source_group,
    score_candidate_novelty,
)
from deep_research.models import (
    DedupeEvent,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
)


def dedupe_snippets(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
    """Keep snippet order while removing duplicates by text and locator."""
    seen: set[tuple[str, str | None]] = set()
    deduped: list[EvidenceSnippet] = []
    for snippet in snippets:
        key = (snippet.text, snippet.source_locator)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    return deduped


def ratchet_merge(
    existing: EvidenceCandidate, incoming: EvidenceCandidate
) -> EvidenceCandidate:
    """Merge duplicate candidates by keeping the strongest known fields from both."""
    existing_selection_score = float(existing.raw_metadata.get("selection_score", 0.0))
    incoming_selection_score = float(incoming.raw_metadata.get("selection_score", 0.0))
    existing_novelty_score = float(existing.raw_metadata.get("novelty_score", 0.0))
    incoming_novelty_score = float(incoming.raw_metadata.get("novelty_score", 0.0))
    source_group = incoming.raw_metadata.get("source_group") or existing.raw_metadata.get(
        "source_group"
    )
    return existing.model_copy(
        update={
            "quality_score": max(existing.quality_score, incoming.quality_score),
            "relevance_score": max(existing.relevance_score, incoming.relevance_score),
            "authority_score": max(existing.authority_score, incoming.authority_score),
            "freshness_score": max(existing.freshness_score, incoming.freshness_score),
            "matched_subtopics": list(
                dict.fromkeys(
                    [*existing.matched_subtopics, *incoming.matched_subtopics]
                )
            ),
            "snippets": dedupe_snippets([*existing.snippets, *incoming.snippets]),
            "doi": existing.doi or incoming.doi,
            "arxiv_id": existing.arxiv_id or incoming.arxiv_id,
            "raw_metadata": {
                **existing.raw_metadata,
                **incoming.raw_metadata,
                "source_group": source_group,
                "selection_score": round(
                    max(existing_selection_score, incoming_selection_score), 4
                ),
                "novelty_score": round(
                    max(existing_novelty_score, incoming_novelty_score), 4
                ),
            },
            "iteration_added": existing.iteration_added or incoming.iteration_added,
            "selected": existing.selected or incoming.selected,
        }
    )


def canonicalize_candidates(
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], list[DedupeEvent]]:
    """Deduplicate candidates, merge duplicate records, and emit the dedupe event log.

    This helper walks candidates in order, finds an existing canonical record using the
    shared precedence rules, ratchets fields together, and records every duplicate match.
    """
    seen: dict[tuple[str, str], EvidenceCandidate] = {}
    canonical: list[EvidenceCandidate] = []
    dedupe_log: list[DedupeEvent] = []

    for candidate in candidates:
        existing = None
        match_basis = None
        for key in match_precedence_keys(candidate):
            existing = seen.get(key)
            if existing is not None:
                match_basis = key[0]
                break

        if existing is None:
            canonical.append(candidate)
            for key in match_precedence_keys(candidate):
                seen[key] = candidate
            continue

        merged = ratchet_merge(existing, candidate)
        canonical = [merged if item.key == existing.key else item for item in canonical]
        for key in match_precedence_keys(merged):
            seen[key] = merged
        dedupe_log.append(
            DedupeEvent(
                duplicate_key=candidate.key,
                canonical_key=existing.key,
                match_basis=match_basis,
            )
        )

    return canonical, dedupe_log


def merge_candidates(
    existing: list[EvidenceCandidate],
    incoming: list[EvidenceCandidate],
    *,
    quality_floor: float = 0.3,
    iteration: int | None = None,
) -> EvidenceLedger:
    """Combine candidates into a deduplicated ledger with runtime inclusion buckets.

    Backward compatibility:
    - ``ledger.selected`` remains the runtime inclusion set consumed by downstream
      checkpoints such as enrichment and context truncation.
    - Final deliverable ordering still happens later in ``rank_evidence``.
    """
    normalized_incoming = [
        candidate
        if candidate.iteration_added is not None or iteration is None
        else candidate.model_copy(update={"iteration_added": iteration})
        for candidate in incoming
    ]
    canonical, dedupe_log = canonicalize_candidates([*existing, *normalized_incoming])

    domain_counts = Counter(
        str(candidate.url.host).lower() if hasattr(candidate.url, "host") else ""
        for candidate in canonical
    )
    title_counts = Counter(candidate.title.strip().lower() for candidate in canonical)

    considered: list[EvidenceCandidate] = []
    for candidate in canonical:
        source_group = infer_source_group(candidate)
        novelty_score = score_candidate_novelty(
            candidate,
            domain_counts=domain_counts,
            title_counts=title_counts,
        )
        selection_score = combined_candidate_score(
            candidate,
            domain_counts=domain_counts,
            title_counts=title_counts,
        )
        runtime_selected = candidate.quality_score >= quality_floor
        considered.append(
            candidate.model_copy(
                update={
                    "selected": runtime_selected,
                    "raw_metadata": {
                        **candidate.raw_metadata,
                        "source_group": source_group.value,
                        "novelty_score": novelty_score,
                        "selection_score": selection_score,
                        "runtime_selected": runtime_selected,
                    },
                }
            )
        )
    selected = [candidate for candidate in considered if candidate.selected]
    rejected = [candidate for candidate in considered if not candidate.selected]
    return EvidenceLedger(
        considered=considered,
        selected=selected,
        rejected=rejected,
        dedupe_log=dedupe_log,
    )


def select_new_this_iteration(
    ledger: EvidenceLedger, iteration: int
) -> list[EvidenceCandidate]:
    return [
        candidate
        for candidate in ledger.considered
        if candidate.iteration_added == iteration
    ]


def _truncate_snippets(
    candidate: EvidenceCandidate, snippet_budget_chars: int
) -> EvidenceCandidate:
    snippets: list[EvidenceSnippet] = []
    spent = 0
    for snippet in candidate.snippets:
        remaining = snippet_budget_chars - spent
        if remaining <= 0:
            break
        text = snippet.text[:remaining].strip()
        if not text:
            continue
        snippets.append(
            snippet.model_copy(
                update={"text": text},
            )
        )
        spent += len(text)
    return candidate.model_copy(update={"snippets": snippets})


def _context_sort_key(candidate: EvidenceCandidate) -> tuple[float, float, int, str]:
    selection_score = float(candidate.raw_metadata.get("selection_score", 0.0))
    return (
        -selection_score,
        -candidate.relevance_score,
        -candidate.quality_score,
        -(candidate.iteration_added or -1),
        candidate.key,
    )


def _truncate_bucket(
    candidates: list[EvidenceCandidate],
    *,
    remaining_chars: int,
    snippet_budget_chars: int,
) -> tuple[list[EvidenceCandidate], int]:
    kept: list[EvidenceCandidate] = []
    for candidate in sorted(candidates, key=_context_sort_key):
        trimmed = _truncate_snippets(candidate, snippet_budget_chars)
        candidate_chars = len(json.dumps(trimmed.model_dump(mode="json"), allow_nan=False))
        if kept and candidate_chars > remaining_chars:
            continue
        if not kept and candidate_chars > remaining_chars:
            kept.append(trimmed)
            remaining_chars = 0
            break
        kept.append(trimmed)
        remaining_chars -= candidate_chars
    return kept, remaining_chars


def truncate_ledger_for_context(
    ledger: EvidenceLedger,
    *,
    max_chars: int,
    role: str,
    snippet_budget_chars: int,
    current_iteration: int | None = None,
) -> EvidenceLedger:
    """Return a context-friendly ledger snapshot bounded by character budget."""
    if max_chars <= 0:
        return EvidenceLedger()

    remaining_chars = max_chars
    selected_source = (
        resolve_selected_entries(ledger)
        if role in {"supervisor", "writer"}
        else resolve_coverage_entries(ledger)
    )
    selected, remaining_chars = _truncate_bucket(
        list(selected_source),
        remaining_chars=remaining_chars,
        snippet_budget_chars=snippet_budget_chars,
    )

    considered: list[EvidenceCandidate] = list(selected)
    rejected: list[EvidenceCandidate] = []

    if role == "relevance":
        non_selected = [
            candidate
            for candidate in ledger.considered
            if candidate.key not in {entry.key for entry in considered}
        ]
        extra, remaining_chars = _truncate_bucket(
            non_selected,
            remaining_chars=remaining_chars,
            snippet_budget_chars=snippet_budget_chars,
        )
        considered.extend(extra)

    if role == "supervisor" and current_iteration is not None:
        new_entries = [
            candidate
            for candidate in select_new_this_iteration(ledger, current_iteration)
            if candidate.key not in {entry.key for entry in considered}
        ]
        extra, _ = _truncate_bucket(
            new_entries,
            remaining_chars=remaining_chars,
            snippet_budget_chars=snippet_budget_chars,
        )
        considered.extend(extra)

    seen: set[str] = set()
    normalized_considered: list[EvidenceCandidate] = []
    for candidate in considered:
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        normalized_considered.append(candidate)

    return EvidenceLedger(
        considered=normalized_considered,
        selected=[candidate for candidate in normalized_considered if candidate.selected],
        rejected=rejected,
        dedupe_log=list(ledger.dedupe_log),
    )
