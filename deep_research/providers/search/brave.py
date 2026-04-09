from __future__ import annotations

from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import RawToolResult
from deep_research.providers.search import failure_result
from deep_research.providers.search._http import build_client


class BraveSearchProvider:
    name = "brave"
    source_group = SourceGroup.WEB
    supported_source_kinds = (SourceKind.WEB,)
    _cost_per_query_usd = 0.005

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
            freshness = None
            if recency_days is not None and recency_days <= 1:
                freshness = "pd"
            elif recency_days is not None and recency_days <= 7:
                freshness = "pw"
            elif recency_days is not None and recency_days <= 31:
                freshness = "pm"
            elif recency_days is not None:
                freshness = "py"

            for query in queries:
                try:
                    response = client.get(
                        "https://api.search.brave.com/res/v1/web/search",
                        headers={"X-Subscription-Token": self._api_key},
                        params={
                            "q": query,
                            "count": min(max_results_per_query, 20),
                            **({"freshness": freshness} if freshness else {}),
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    items = [
                        {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "freshness_score": 0.7 if item.get("page_age") else 0.3,
                        }
                        for item in payload.get("web", {}).get("results", [])
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
