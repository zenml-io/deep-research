"""Tavily search provider.

No-op when ``TAVILY_API_KEY`` environment variable is absent or empty.
"""

from __future__ import annotations

import logging
import os

from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    build_async_client,
    request_with_retry,
)
from research.providers.search import SearchResult

logger = logging.getLogger(__name__)

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Map recency_days ranges to Tavily's time_range parameter.
_RECENCY_TO_TIME_RANGE: list[tuple[int, str]] = [
    (1, "day"),
    (7, "week"),
    (30, "month"),
    (365, "year"),
]


def _recency_to_time_range(recency_days: int) -> str:
    """Convert a recency_days integer to the closest Tavily time_range value."""
    for threshold, label in _RECENCY_TO_TIME_RANGE:
        if recency_days <= threshold:
            return label
    return "year"


class TavilySearchProvider:
    """Tavily web search API provider."""

    name = "tavily"

    def __init__(self) -> None:
        self._api_key = os.environ.get("TAVILY_API_KEY", "")

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

        time_range: str | None = None
        if recency_days is not None:
            time_range = _recency_to_time_range(recency_days)

        results: list[SearchResult] = []

        async with build_async_client() as client:
            for query in queries:
                try:
                    body: dict[str, object] = {
                        "query": query,
                        "max_results": min(max_results_per_query, 20),
                        "search_depth": "basic",
                    }
                    if time_range:
                        body["time_range"] = time_range

                    response = await request_with_retry(
                        client,
                        "POST",
                        _TAVILY_SEARCH_URL,
                        retry_policy=DEFAULT_RETRY_POLICY,
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                    )
                    payload = response.json()
                    for item in payload.get("results", []):
                        results.append(
                            SearchResult(
                                url=item.get("url", ""),
                                title=item.get("title", ""),
                                snippet=item.get("content", "") or "",
                                provider=self.name,
                                raw_metadata={
                                    "score": item.get("score"),
                                    "query": query,
                                },
                            )
                        )
                except Exception:
                    logger.exception("Tavily search failed for query: %s", query)

        return results
