"""Semantic Scholar search provider.

Always available — works without an API key but gets higher rate
limits when ``SEMANTIC_SCHOLAR_API_KEY`` is set.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta

from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    build_async_client,
    request_with_retry,
)
from research.providers.search import SearchResult

logger = logging.getLogger(__name__)

_S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

_S2_FIELDS = (
    "title,url,abstract,year,citationCount,authors,"
    "externalIds,fieldsOfStudy,isOpenAccess,publicationDate"
)


def _published_on_or_after(
    publication_date: str | None,
    year: int | None,
    recency_days: int | None,
) -> bool:
    """Return True if the paper's publication date is within recency window."""
    if recency_days is None:
        return True
    published_at: date | None = None
    if publication_date:
        try:
            published_at = datetime.fromisoformat(
                publication_date.replace("Z", "+00:00")
            ).date()
        except ValueError:
            published_at = None
    if published_at is None and year is not None:
        published_at = date(year, 1, 1)
    if published_at is None:
        return True
    cutoff = (datetime.now(UTC) - timedelta(days=recency_days)).date()
    return published_at >= cutoff


class SemanticScholarProvider:
    """Semantic Scholar Academic Graph API provider."""

    name = "semantic_scholar"

    def __init__(self) -> None:
        self._api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    def is_available(self) -> bool:
        return True

    async def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        results: list[SearchResult] = []

        async with build_async_client() as client:
            for query in queries:
                try:
                    response = await request_with_retry(
                        client,
                        "GET",
                        _S2_SEARCH_URL,
                        retry_policy=DEFAULT_RETRY_POLICY,
                        headers=headers,
                        params={
                            "query": query,
                            "limit": max_results_per_query,
                            "fields": _S2_FIELDS,
                        },
                    )
                    payload = response.json()
                    for paper in payload.get("data") or []:
                        if not _published_on_or_after(
                            paper.get("publicationDate"),
                            paper.get("year"),
                            recency_days,
                        ):
                            continue
                        external_ids = paper.get("externalIds") or {}
                        results.append(
                            SearchResult(
                                url=paper.get("url", ""),
                                title=paper.get("title", ""),
                                snippet=paper.get("abstract", "") or "",
                                provider=self.name,
                                raw_metadata={
                                    "doi": external_ids.get("DOI"),
                                    "arxiv_id": external_ids.get("ArXiv"),
                                    "citation_count": paper.get("citationCount", 0),
                                    "year": paper.get("year"),
                                    "query": query,
                                },
                            )
                        )
                except Exception:
                    logger.exception(
                        "Semantic Scholar search failed for query: %s", query
                    )

        return results
