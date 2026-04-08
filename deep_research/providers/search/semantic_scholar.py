from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from deep_research.enums import SourceKind
from deep_research.models import RawToolResult
from deep_research.providers.search import failure_result
from deep_research.providers.search._http import build_client


def published_on_or_after(
    publication_date: str | None, year: int | None, recency_days: int | None
) -> bool:
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
    name = "semantic_scholar"
    supported_source_kinds = (SourceKind.PAPER,)

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

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
        client = build_client()
        try:
            results: list[RawToolResult] = []
            headers = {"x-api-key": self._api_key} if self._api_key else {}
            for query in queries:
                try:
                    response = client.get(
                        "https://api.semanticscholar.org/graph/v1/paper/search",
                        headers=headers,
                        params={
                            "query": query,
                            "limit": max_results_per_query,
                            "fields": "title,url,abstract,year,citationCount,authors,externalIds,fieldsOfStudy,isOpenAccess,publicationDate",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    items = []
                    for paper in payload.get("data", []):
                        if not published_on_or_after(
                            paper.get("publicationDate"),
                            paper.get("year"),
                            recency_days,
                        ):
                            continue
                        external_ids = paper.get("externalIds") or {}
                        items.append(
                            {
                                "title": paper.get("title", ""),
                                "url": paper.get("url", ""),
                                "description": paper.get("abstract", ""),
                                "doi": external_ids.get("DOI"),
                                "arxiv_id": external_ids.get("ArXiv"),
                                "authority_score": (
                                    0.9 if paper.get("citationCount", 0) >= 100 else 0.5
                                ),
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
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
