from deep_research.evidence.dedup import dedupe_candidates
from deep_research.models import EvidenceCandidate, EvidenceLedger


def merge_candidates(
    existing: list[EvidenceCandidate], incoming: list[EvidenceCandidate]
) -> EvidenceLedger:
    """Combine existing and incoming candidates into a deduplicated ledger."""
    combined = dedupe_candidates([*existing, *incoming])
    return EvidenceLedger(entries=combined)
