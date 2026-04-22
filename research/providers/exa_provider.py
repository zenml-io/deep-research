"""Exa search provider.

No-op when ``EXA_API_KEY`` environment variable is absent or empty.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    build_async_client,
    request_with_retry,
)
from research.providers.search import SearchResult

logger = logging.getLogger(__name__)

_EXA_SEARCH_URL = "https://api.exa.ai/search"


class ExaSearchProvider:
    """Exa neural search API provider."""

    name = "exa"

    def __init__(self) -> None:
        self._api_key = os.environ.get("EXA_API_KEY", "")

    def is_available(self) -> bool:
        return bool(self._api_key)

    async def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        if not self.is_available():
            return []

        start_published_date: str | None = None
        if recency_days is not None:
            start_published_date = (
                datetime.now(UTC) - timedelta(days=recency_days)
            ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        results: list[SearchResult] = []

        async with build_async_client() as client:
            for query in queries:
                try:
                    body: dict[str, object] = {
                        "query": query,
                        "numResults": max_results_per_query,
                    }
                    if start_published_date:
                        body["startPublishedDate"] = start_published_date

                    response = await request_with_retry(
                        client,
                        "POST",
                        _EXA_SEARCH_URL,
                        retry_policy=DEFAULT_RETRY_POLICY,
                        headers={"x-api-key": self._api_key},
                        json=body,
                    )
                    payload = response.json()
                    for item in payload.get("results", []):
                        results.append(
                            SearchResult(
                                url=item.get("url", ""),
                                title=item.get("title", ""),
                                snippet=item.get("text", "") or "",
                                provider=self.name,
                                raw_metadata={
                                    "published_date": item.get("publishedDate"),
                                    "query": query,
                                },
                            )
                        )
                except Exception:
                    logger.exception("Exa search failed for query: %s", query)

        return results
