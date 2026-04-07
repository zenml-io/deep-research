from deep_research.enums import SourceKind
from deep_research.models import EvidenceCandidate

_QUALITY_BY_SOURCE_KIND: dict[SourceKind, float] = {
    SourceKind.PAPER: 0.9,
    SourceKind.DOCS: 0.8,
    SourceKind.WEB: 0.6,
    SourceKind.DATASET: 0.4,
}

_DEFAULT_QUALITY = 0.4


def score_candidate_quality(candidate: EvidenceCandidate) -> float:
    """Return a quality score based on the candidate's source kind."""
    base_score = _QUALITY_BY_SOURCE_KIND.get(candidate.source_kind, _DEFAULT_QUALITY)
    return round(max(base_score, candidate.authority_score), 4)
