from collections.abc import Iterable

from deep_research.evidence.url import canonicalize_url
from deep_research.models import DedupeEvent, EvidenceCandidate


def candidate_identities(candidate: EvidenceCandidate) -> list[tuple[str, str]]:
    """Collect every normalized identity that can participate in deduplication matching.

    The returned identities intentionally include DOI, arXiv ID, canonical URL, and
    normalized title so later precedence logic can choose the strongest available match.
    """
    identities: list[tuple[str, str]] = []
    doi = (candidate.doi or "").strip().lower()
    if doi:
        identities.append(("doi", doi))
    arxiv_id = (candidate.arxiv_id or "").strip().lower()
    if arxiv_id:
        identities.append(("arxiv_id", arxiv_id))
    canonical_url = canonicalize_url(str(candidate.url))
    if canonical_url:
        identities.append(("canonical_url", canonical_url))
    title = candidate.title.strip().lower()
    if title:
        identities.append(("title", title))
    return identities


def match_precedence_keys(candidate: EvidenceCandidate) -> Iterable[tuple[str, str]]:
    """Yield deduplication identities in the exact precedence order used for matching.

    Consumers rely on this order to prefer DOI matches first, then arXiv IDs, then
    canonical URLs, and only fall back to title matching as the weakest signal.
    """
    identities = dict(candidate_identities(candidate))
    for basis in ("doi", "arxiv_id", "canonical_url", "title"):
        identity = identities.get(basis)
        if identity:
            yield (basis, identity)


def dedupe_candidates(
    candidates: list[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], list[DedupeEvent]]:
    """Remove duplicate candidates using Phase 2 identifier precedence."""
    seen: dict[tuple[str, str], EvidenceCandidate] = {}
    deduped: list[EvidenceCandidate] = []
    dedupe_log: list[DedupeEvent] = []

    for candidate in candidates:
        canonical = None
        match_basis = None
        for key in match_precedence_keys(candidate):
            canonical = seen.get(key)
            if canonical is not None:
                match_basis = key[0]
                break
        if canonical is None:
            deduped.append(candidate)
            for key in match_precedence_keys(candidate):
                seen[key] = candidate
            continue
        dedupe_log.append(
            DedupeEvent(
                duplicate_key=candidate.key,
                canonical_key=canonical.key,
                match_basis=match_basis,
            )
        )
        for key in match_precedence_keys(candidate):
            seen.setdefault(key, canonical)

    return deduped, dedupe_log
