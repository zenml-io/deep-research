import logging

from kitaru import checkpoint

from deep_research.config import ResearchConfig

logger = logging.getLogger(__name__)
from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.models import EvidenceCandidate, EvidenceLedger, EvidenceSnippet
from deep_research.observability import span
from deep_research.providers.search.fetcher import fetch_url_content


@checkpoint(type="tool_call")
def enrich_candidates(ledger: EvidenceLedger, config: ResearchConfig) -> EvidenceLedger:
    with span("enrich_candidates", candidate_count=len(ledger.selected)):
        selected = sorted(
            resolve_selected_entries(ledger),
            key=lambda candidate: (
                -candidate.quality_score,
                -candidate.authority_score,
                -candidate.relevance_score,
                candidate.key,
            ),
        )[: config.max_fetch_candidates_per_iteration]

        candidates_by_key: dict[str, EvidenceCandidate] = {}
        for bucket in (ledger.considered, ledger.selected, ledger.rejected):
            for candidate in bucket:
                candidates_by_key.setdefault(candidate.key, candidate)

        updated_by_key: dict[str, EvidenceCandidate] = {}
        for candidate in selected:
            canonical_candidate = candidates_by_key.get(candidate.key)
            if canonical_candidate is None:
                continue
            if any(
                snippet.source_locator == "fetched:body"
                for snippet in canonical_candidate.snippets
            ):
                continue
            try:
                fetched = fetch_url_content(
                    str(canonical_candidate.url),
                    timeout_sec=config.tool_timeout_sec,
                    max_chars=config.max_fetched_chars_per_candidate,
                )
            except Exception:
                logger.warning(
                    "Failed to fetch content for %s: %s",
                    canonical_candidate.url,
                    candidate.key,
                    exc_info=True,
                )
                continue
            if not fetched:
                continue
            updated_by_key[candidate.key] = canonical_candidate.model_copy(
                update={
                    "snippets": [
                        *canonical_candidate.snippets,
                        EvidenceSnippet(text=fetched, source_locator="fetched:body"),
                    ]
                }
            )

        if not updated_by_key:
            return ledger

        return ledger.model_copy(
            update={
                "considered": [
                    updated_by_key.get(candidate.key, candidate)
                    for candidate in ledger.considered
                ],
                "selected": [
                    updated_by_key.get(candidate.key, candidate)
                    for candidate in ledger.selected
                ],
                "rejected": [
                    updated_by_key.get(candidate.key, candidate)
                    for candidate in ledger.rejected
                ],
            }
        )
