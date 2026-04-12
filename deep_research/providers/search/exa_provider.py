from __future__ import annotations

from datetime import UTC, datetime, timedelta

from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import RawToolResult
from deep_research.providers.search import failure_result
from deep_research.providers.search._http import (
    DEFAULT_RETRY_POLICY,
    build_client,
    request_with_retry,
)


class ExaSearchProvider:
    name = "exa"
    source_group = SourceGroup.WEB
    supported_source_kinds = (SourceKind.WEB,)
    _cost_per_query_usd = 0.02

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def is_available(self) -> bool:
        return bool(self._api_key)

    def estimate_cost_usd(self, query_count: int) -> float:
        return round(query_count * self._cost_per_query_usd, 6)

    def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[RawToolResult]:
        client = build_client()
        try:
            results: list[RawToolResult] = []
            start_published_date = None
            if recency_days is not None:
                start_published_date = (
                    datetime.now(UTC) - timedelta(days=recency_days)
                ).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            for query in queries:
                try:
                    response = request_with_retry(
                        client,
                        "POST",
                        "https://api.exa.ai/search",
                        retry_policy=DEFAULT_RETRY_POLICY,
                        headers={"x-api-key": self._api_key},
                        json={
                            "query": query,
                            "numResults": max_results_per_query,
                            **(
                                {"startPublishedDate": start_published_date}
                                if start_published_date
                                else {}
                            ),
                        },
                    )
                    payload = response.json()
                    items = [
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("text", ""),
                            "freshness_score": (
                                0.7 if item.get("publishedDate") else 0.3
                            ),
                        }
                        for item in payload.get("results", [])
                    ]
                    results.append(
                        RawToolResult(
                            tool_name="search",
                            provider=self.name,
                            payload={"source_kind": "web", "results": items},
                        )
                    )
                except Exception as exc:
                    results.append(failure_result(self.name, "web", exc))
            return results
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
