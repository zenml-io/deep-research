from kitaru import checkpoint

from deep_research.models import EvidenceCandidate, RawToolResult
from deep_research.observability import span
from deep_research.providers.normalization import normalize_tool_results


@checkpoint(type="tool_call")
def extract_candidates(raw_results: list[RawToolResult]) -> list[EvidenceCandidate]:
    """Checkpoint: convert raw tool outputs into normalized evidence candidates."""
    with span("extract_candidates", raw_count=len(raw_results)):
        candidates: list[EvidenceCandidate] = []
        for result in raw_results:
            if result.ok:
                candidates.extend(
                    normalize_tool_results(
                        [result],
                        provider=result.provider,
                        source_kind=result.payload.get("source_kind", "web"),
                    )
                )
        return candidates
