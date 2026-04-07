from deep_research.evidence.dedup import _match_precedence_keys
from deep_research.models import (
    DedupeEvent,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
)


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
    """Merge duplicate candidates by keeping the strongest known fields from both."""
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
            "snippets": _dedupe_snippets([*existing.snippets, *incoming.snippets]),
            "doi": existing.doi or incoming.doi,
            "arxiv_id": existing.arxiv_id or incoming.arxiv_id,
            "raw_metadata": {**existing.raw_metadata, **incoming.raw_metadata},
            "selected": existing.selected or incoming.selected,
        }
    )


def _canonicalize_candidates(
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], list[DedupeEvent]]:
    """Deduplicate candidates in precedence order and record duplicate matches."""
    seen: dict[tuple[str, str], EvidenceCandidate] = {}
    canonical: list[EvidenceCandidate] = []
    dedupe_log: list[DedupeEvent] = []

    for candidate in candidates:
        existing = None
        match_basis = None
        for key in _match_precedence_keys(candidate):
            existing = seen.get(key)
            if existing is not None:
                match_basis = key[0]
                break

        if existing is None:
            canonical.append(candidate)
            for key in _match_precedence_keys(candidate):
                seen[key] = candidate
            continue

        merged = _ratchet_merge(existing, candidate)
        canonical = [merged if item.key == existing.key else item for item in canonical]
        for key in _match_precedence_keys(merged):
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
) -> EvidenceLedger:
    """Combine candidates into a deduplicated ledger with ratcheted scores."""
    canonical, dedupe_log = _canonicalize_candidates([*existing, *incoming])

    considered = [
        candidate.model_copy(
            update={"selected": candidate.quality_score >= quality_floor}
        )
        for candidate in canonical
    ]
    selected = [candidate for candidate in considered if candidate.selected]
    rejected = [candidate for candidate in considered if not candidate.selected]
    return EvidenceLedger(
        considered=considered,
        selected=selected,
        rejected=rejected,
        dedupe_log=dedupe_log,
    )
