from deep_research.models import EvidenceCandidate


def score_candidate_quality(candidate: EvidenceCandidate) -> float:
    if candidate.source_kind == "paper":
        return 0.9
    if candidate.source_kind == "docs":
        return 0.8
    if candidate.source_kind == "web":
        return 0.6
    return 0.4
