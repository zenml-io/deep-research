from deep_research.models import EvidenceCandidate


def dedupe_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    seen: dict[str, EvidenceCandidate] = {}

    for candidate in candidates:
        key = str(candidate.url).strip().lower() or candidate.title.strip().lower()
        seen.setdefault(key, candidate)

    return list(seen.values())
