from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.evidence.ledger import merge_candidates
from deep_research.models import DedupeEvent, EvidenceCandidate, EvidenceLedger


def _merge_dedupe_history(events: list[DedupeEvent]) -> list[DedupeEvent]:
    """Merge dedupe events while preserving first-seen order and removing repeats."""
    merged: list[DedupeEvent] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        identity = (event.duplicate_key, event.canonical_key, event.match_basis)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(event)
    return merged


@checkpoint(type="tool_call")
def merge_evidence(
    scored: list[EvidenceCandidate],
    ledger: EvidenceLedger,
    config: ResearchConfig | None = None,
) -> EvidenceLedger:
    """Checkpoint: merge newly scored candidates into the existing evidence ledger."""
    quality_floor = (
        config.source_quality_floor
        if config is not None
        else ResearchConfig.for_tier(Tier.STANDARD).source_quality_floor
    )
    merged = merge_candidates(ledger.entries, scored, quality_floor=quality_floor)
    return merged.model_copy(
        update={
            "dedupe_log": _merge_dedupe_history(
                [*ledger.dedupe_log, *merged.dedupe_log]
            )
        }
    )
