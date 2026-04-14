"""arXiv search provider.

Always available — no API key required. Uses the ``arxiv`` Python
library for structured queries.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import arxiv

from research.providers.search import SearchResult

logger = logging.getLogger(__name__)


def _is_recent_enough(published: datetime, recency_days: int | None) -> bool:
    """Return True if ``published`` is within ``recency_days`` of now."""
    if recency_days is None:
        return True
    published_at = published
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=UTC)
    cutoff = datetime.now(UTC) - timedelta(days=recency_days)
    return published_at >= cutoff


class ArxivSearchProvider:
    """Search arXiv papers via the ``arxiv`` Python library."""

    name = "arxiv"

    def is_available(self) -> bool:
        return True

    async def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        client = arxiv.Client(page_size=max_results_per_query, delay_seconds=3.0)
        results: list[SearchResult] = []

        for query in queries:
            try:
                search = arxiv.Search(query=query, max_results=max_results_per_query)
                papers = list(client.results(search))
                for paper in papers:
                    if not _is_recent_enough(paper.published, recency_days):
                        continue
                    results.append(
                        SearchResult(
                            url=paper.entry_id,
                            title=paper.title,
                            snippet=paper.summary[:500] if paper.summary else "",
                            provider=self.name,
                            raw_metadata={
                                "arxiv_id": paper.get_short_id(),
                                "published": paper.published.isoformat(),
                                "authors": [a.name for a in paper.authors],
                            },
                        )
                    )
            except Exception:
                logger.exception("arXiv search failed for query: %s", query)

        return results
