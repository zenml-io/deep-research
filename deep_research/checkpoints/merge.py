from kitaru import checkpoint

from deep_research.evidence.ledger import merge_candidates
from deep_research.models import EvidenceCandidate, EvidenceLedger


@checkpoint(type="tool_call")
def merge_evidence(
    scored: list[EvidenceCandidate], ledger: EvidenceLedger
) -> EvidenceLedger:
    """Checkpoint: merge newly scored candidates into the existing evidence ledger."""
    return merge_candidates(ledger.entries, scored)
