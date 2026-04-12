from __future__ import annotations

from datetime import UTC, datetime, timedelta

import arxiv

from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import RawToolResult
from deep_research.providers.search import failure_result
from deep_research.providers.search._http import DEFAULT_RETRY_POLICY, call_with_retry


def is_recent_enough(published: datetime, recency_days: int | None) -> bool:
    if recency_days is None:
        return True
    published_at = published
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    cutoff = datetime.now(UTC) - timedelta(days=recency_days)
    return published_at >= cutoff


class ArxivSearchProvider:
    name = "arxiv"
    source_group = SourceGroup.PAPERS
    supported_source_kinds = (SourceKind.PAPER,)

    def is_available(self) -> bool:
        return True

    def estimate_cost_usd(self, query_count: int) -> float:
        return 0.0

    def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[RawToolResult]:
        client = arxiv.Client(page_size=max_results_per_query, delay_seconds=3.0)
        results: list[RawToolResult] = []
        for query in queries:
            try:
                papers = call_with_retry(
                    lambda: list(
                        client.results(
                            arxiv.Search(
                                query=query, max_results=max_results_per_query
                            )
                        )
                    ),
                    retry_policy=DEFAULT_RETRY_POLICY,
                    is_retryable=_is_retryable_arxiv_error,
                )
                items = []
                for paper in papers:
                    if not is_recent_enough(paper.published, recency_days):
                        continue
                    items.append(
                        {
                            "title": paper.title,
                            "url": paper.entry_id,
                            "description": paper.summary,
                            "arxiv_id": paper.get_short_id(),
                            "raw_published": paper.published.isoformat(),
                            "authors": [author.name for author in paper.authors],
                        }
                    )
                results.append(
                    RawToolResult(
                        tool_name="search",
                        provider=self.name,
                        payload={"source_kind": "paper", "results": items},
                    )
                )
            except Exception as exc:
                results.append(failure_result(self.name, "paper", exc))
        return results


def _is_retryable_arxiv_error(exc: Exception) -> bool:
    return not isinstance(exc, (TypeError, ValueError))
