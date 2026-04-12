from kitaru import checkpoint

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.evidence.ledger import merge_candidates
from deep_research.models import DedupeEvent, EvidenceCandidate, EvidenceLedger
from deep_research.observability import span


@checkpoint(type="tool_call")
def update_ledger(
    scored: list[EvidenceCandidate],
    ledger: EvidenceLedger,
    config: ResearchConfig | None = None,
    iteration: int | None = None,
) -> EvidenceLedger:
    """Checkpoint: merge newly scored candidates into the existing evidence ledger."""
    with span("update_ledger", incoming_count=len(scored)):
        quality_floor = (
            config.source_quality_floor
            if config is not None
            else ResearchConfig.for_tier(Tier.STANDARD).source_quality_floor
        )
        merged = merge_candidates(
            ledger.entries,
            scored,
            quality_floor=quality_floor,
            iteration=iteration,
        )
        dedupe_log: list[DedupeEvent] = []
        seen_events: set[tuple[str, str, str]] = set()
        for event in [*ledger.dedupe_log, *merged.dedupe_log]:
            identity = (event.duplicate_key, event.canonical_key, event.match_basis)
            if identity in seen_events:
                continue
            seen_events.add(identity)
            dedupe_log.append(event)
        return merged.model_copy(update={"dedupe_log": dedupe_log})
