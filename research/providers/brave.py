"""Brave web-search provider.

No-op when ``BRAVE_API_KEY`` environment variable is absent or empty.
"""

from __future__ import annotations

import logging
import os

import httpx

from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    build_async_client,
    request_with_retry,
)
from research.providers.search import SearchResult

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider:
    """Brave Search API provider."""

    name = "brave"

    def __init__(self) -> None:
        self._api_key = os.environ.get("BRAVE_API_KEY", "")

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

        freshness = _recency_to_freshness(recency_days)
        results: list[SearchResult] = []

        async with build_async_client() as client:
            for query in queries:
                try:
                    params: dict[str, object] = {
                        "q": query,
                        "count": min(max_results_per_query, 20),
                    }
                    if freshness:
                        params["freshness"] = freshness

                    response = await request_with_retry(
                        client,
                        "GET",
                        _BRAVE_SEARCH_URL,
                        retry_policy=DEFAULT_RETRY_POLICY,
                        headers={"X-Subscription-Token": self._api_key},
                        params=params,
                    )
                    payload = response.json()
                    for item in payload.get("web", {}).get("results", []):
                        results.append(
                            SearchResult(
                                url=item.get("url", ""),
                                title=item.get("title", ""),
                                snippet=item.get("description", ""),
                                provider=self.name,
                                raw_metadata={
                                    "page_age": item.get("page_age"),
                                    "query": query,
                                },
                            )
                        )
                except Exception:
                    logger.exception("Brave search failed for query: %s", query)
        return results


def _recency_to_freshness(recency_days: int | None) -> str | None:
    """Map recency_days to Brave's freshness parameter."""
    if recency_days is None:
        return None
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 31:
        return "pm"
    return "py"
