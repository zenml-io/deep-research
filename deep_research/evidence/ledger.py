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


_SUPERVISOR_ROLES = {"supervisor", "writer"}


def _metadata_float(metadata: dict[str, object], field: str) -> float:
    return float(metadata.get(field, 0.0))


def _ratchet_field(
    existing: dict[str, object], incoming: dict[str, object], field: str
) -> float:
    """Return the higher of two metadata float fields, rounded to 4 decimals."""
    return round(max(_metadata_float(existing, field), _metadata_float(incoming, field)), 4)


def _ratchet_metadata(
    existing: dict[str, object], incoming: dict[str, object]
) -> dict[str, object]:
    """Merge raw metadata, ratcheting selection_score/novelty_score upward."""
    source_group = incoming.get("source_group") or existing.get("source_group")
    return {
        **existing,
        **incoming,
        "source_group": source_group,
        "selection_score": _ratchet_field(existing, incoming, "selection_score"),
        "novelty_score": _ratchet_field(existing, incoming, "novelty_score"),
    }


def _dedupe_snippets(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
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


def _ratchet_merge(
    existing: EvidenceCandidate, incoming: EvidenceCandidate
) -> EvidenceCandidate:
    """Merge duplicate candidates, keeping the strongest known fields from both."""
    return existing.model_copy(
        update={
            "quality_score": max(existing.quality_score, incoming.quality_score),
            "relevance_score": max(existing.relevance_score, incoming.relevance_score),
            "authority_score": max(existing.authority_score, incoming.authority_score),
            "freshness_score": max(existing.freshness_score, incoming.freshness_score),
            "matched_subtopics": list(
                dict.fromkeys([*existing.matched_subtopics, *incoming.matched_subtopics])
            ),
            "snippets": _dedupe_snippets([*existing.snippets, *incoming.snippets]),
            "doi": existing.doi or incoming.doi,
            "arxiv_id": existing.arxiv_id or incoming.arxiv_id,
            "raw_metadata": _ratchet_metadata(existing.raw_metadata, incoming.raw_metadata),
            "iteration_added": existing.iteration_added or incoming.iteration_added,
            "selected": existing.selected or incoming.selected,
        }
    )


def _find_canonical_match(
    candidate: EvidenceCandidate,
    seen: dict[tuple[str, str], EvidenceCandidate],
) -> tuple[EvidenceCandidate | None, str | None]:
    """Return the first existing canonical record matching this candidate, with match basis."""
    for key in match_precedence_keys(candidate):
        existing = seen.get(key)
        if existing is not None:
            return existing, key[0]
    return None, None


def _register_identities(
    candidate: EvidenceCandidate, seen: dict[tuple[str, str], EvidenceCandidate]
) -> None:
    """Register every precedence key for a canonical candidate."""
    for key in match_precedence_keys(candidate):
        seen[key] = candidate


def _canonicalize_candidates(
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], list[DedupeEvent]]:
    """Deduplicate candidates, merge duplicate records, and emit the dedupe event log."""
    seen: dict[tuple[str, str], EvidenceCandidate] = {}
    canonical: list[EvidenceCandidate] = []
    dedupe_log: list[DedupeEvent] = []

    for candidate in candidates:
        existing, match_basis = _find_canonical_match(candidate, seen)
        if existing is None:
            canonical.append(candidate)
            _register_identities(candidate, seen)
            continue

        merged = _ratchet_merge(existing, candidate)
        canonical = [merged if item.key == existing.key else item for item in canonical]
        _register_identities(merged, seen)
        dedupe_log.append(
            DedupeEvent(
                duplicate_key=candidate.key,
                canonical_key=existing.key,
                match_basis=match_basis,
            )
        )

    return canonical, dedupe_log


def _stamp_iteration(
    candidates: list[EvidenceCandidate], iteration: int | None
) -> list[EvidenceCandidate]:
    """Fill in iteration_added on candidates that don't already have one."""
    if iteration is None:
        return list(candidates)
    return [
        c if c.iteration_added is not None else c.model_copy(update={"iteration_added": iteration})
        for c in candidates
    ]


def _score_candidate(
    candidate: EvidenceCandidate,
    *,
    domain_counts: Counter[str],
    title_counts: Counter[str],
    quality_floor: float,
) -> EvidenceCandidate:
    """Apply novelty/selection scoring and runtime selection gate to a single candidate."""
    source_group = infer_source_group(candidate)
    novelty_score = score_candidate_novelty(
        candidate, domain_counts=domain_counts, title_counts=title_counts
    )
    selection_score = combined_candidate_score(
        candidate, domain_counts=domain_counts, title_counts=title_counts
    )
    runtime_selected = candidate.quality_score >= quality_floor
    return candidate.model_copy(
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


def merge_candidates(
    existing: list[EvidenceCandidate],
    incoming: list[EvidenceCandidate],
    *,
    quality_floor: float = 0.3,
    iteration: int | None = None,
) -> EvidenceLedger:
    """Combine candidates into a deduplicated ledger with runtime inclusion buckets."""
    canonical, dedupe_log = _canonicalize_candidates(
        [*existing, *_stamp_iteration(incoming, iteration)]
    )

    domain_counts = Counter(str(c.url.host).lower() for c in canonical)
    title_counts = Counter(c.title.strip().lower() for c in canonical)

    considered = [
        _score_candidate(
            c,
            domain_counts=domain_counts,
            title_counts=title_counts,
            quality_floor=quality_floor,
        )
        for c in canonical
    ]
    return EvidenceLedger(
        considered=considered,
        selected=[c for c in considered if c.selected],
        rejected=[c for c in considered if not c.selected],
        dedupe_log=dedupe_log,
    )


def select_new_this_iteration(
    ledger: EvidenceLedger, iteration: int
) -> list[EvidenceCandidate]:
    return [c for c in ledger.considered if c.iteration_added == iteration]


def _truncate_snippets(
    candidate: EvidenceCandidate, snippet_budget_chars: int
) -> EvidenceCandidate:
    """Trim candidate snippets to fit within a character budget."""
    snippets: list[EvidenceSnippet] = []
    spent = 0
    for snippet in candidate.snippets:
        remaining = snippet_budget_chars - spent
        if remaining <= 0:
            break
        text = snippet.text[:remaining].strip()
        if not text:
            continue
        snippets.append(snippet.model_copy(update={"text": text}))
        spent += len(text)
    return candidate.model_copy(update={"snippets": snippets})


def _context_sort_key(c: EvidenceCandidate) -> tuple[float, float, float, int, str]:
    return (
        -_metadata_float(c.raw_metadata, "selection_score"),
        -c.relevance_score,
        -c.quality_score,
        -(c.iteration_added or -1),
        c.key,
    )


def _candidate_char_size(candidate: EvidenceCandidate) -> int:
    return len(json.dumps(candidate.model_dump(mode="json"), allow_nan=False))


def _truncate_bucket(
    candidates: list[EvidenceCandidate],
    *,
    remaining_chars: int,
    snippet_budget_chars: int,
) -> tuple[list[EvidenceCandidate], int]:
    """Fit as many candidates as possible into the character budget, best-scoring first.

    # Why: first candidate always passes even if oversized — guarantees at least one result.
    """
    kept: list[EvidenceCandidate] = []
    for candidate in sorted(candidates, key=_context_sort_key):
        trimmed = _truncate_snippets(candidate, snippet_budget_chars)
        size = _candidate_char_size(trimmed)
        if kept and size > remaining_chars:
            continue
        if not kept and size > remaining_chars:
            kept.append(trimmed)
            return kept, 0
        kept.append(trimmed)
        remaining_chars -= size
    return kept, remaining_chars


def _append_truncated(
    considered: list[EvidenceCandidate],
    source: list[EvidenceCandidate],
    remaining_chars: int,
    snippet_budget_chars: int,
) -> tuple[list[EvidenceCandidate], int]:
    """Filter source to unknowns, truncate to budget, append to considered."""
    known = {c.key for c in considered}
    extra, remaining_chars = _truncate_bucket(
        [c for c in source if c.key not in known],
        remaining_chars=remaining_chars,
        snippet_budget_chars=snippet_budget_chars,
    )
    return [*considered, *extra], remaining_chars


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

    selected_source = (
        resolve_selected_entries(ledger)
        if role in _SUPERVISOR_ROLES
        else resolve_coverage_entries(ledger)
    )
    considered, remaining_chars = _truncate_bucket(
        list(selected_source),
        remaining_chars=max_chars,
        snippet_budget_chars=snippet_budget_chars,
    )

    if role == "relevance":
        considered, remaining_chars = _append_truncated(
            considered, ledger.considered, remaining_chars, snippet_budget_chars
        )

    if role == "supervisor" and current_iteration is not None:
        new_entries = select_new_this_iteration(ledger, current_iteration)
        considered, _ = _append_truncated(
            considered, new_entries, remaining_chars, snippet_budget_chars
        )

    normalized = list({c.key: c for c in considered}.values())
    return EvidenceLedger(
        considered=normalized,
        selected=[c for c in normalized if c.selected],
        rejected=[],
        dedupe_log=list(ledger.dedupe_log),
    )
