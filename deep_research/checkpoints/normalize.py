from kitaru import checkpoint

from deep_research.models import EvidenceCandidate, RawToolResult
from deep_research.providers.normalization import normalize_tool_results


@checkpoint(type="tool_call")
def normalize_evidence(raw_results: list[RawToolResult]) -> list[EvidenceCandidate]:
    candidates: list[EvidenceCandidate] = []
    for result in raw_results:
        if result.ok:
            candidates.extend(
                normalize_tool_results(
                    result.payload.get("results", []),
                    provider=result.provider,
                    source_kind=result.payload.get("source_kind", "web"),
                )
            )
    return candidates
