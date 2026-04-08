from datetime import UTC, datetime

import pytest

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import SearchAction
from deep_research.providers.search import ProviderRegistry
from deep_research.providers.search.arxiv_provider import ArxivSearchProvider
from deep_research.providers.search.brave import BraveSearchProvider
from deep_research.providers.search.exa_provider import ExaSearchProvider
from deep_research.providers.search.semantic_scholar import SemanticScholarProvider


def test_registry_skips_keyed_providers_without_keys() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "enabled_providers": ["brave", "arxiv", "semantic_scholar", "exa"],
            "brave_api_key": "",
            "exa_api_key": "",
            "semantic_scholar_api_key": "",
        }
    )

    registry = ProviderRegistry(config)

    assert [provider.name for provider in registry.active_providers()] == [
        "arxiv",
        "semantic_scholar",
    ]


def test_registry_rejects_unknown_enabled_provider_names() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={"enabled_providers": ["arxiv", "semntic_scholar"]}
    )

    with pytest.raises(ValueError, match="Unknown search providers"):
        ProviderRegistry(config)


def test_registry_routes_to_requested_provider_subset() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD).model_copy(
        update={
            "enabled_providers": ["arxiv", "semantic_scholar"],
        }
    )
    registry = ProviderRegistry(config)
    action = SearchAction(
        query="rlhf alternatives",
        rationale="Need paper sources.",
        preferred_providers=["semantic_scholar"],
    )

    assert [provider.name for provider in registry.providers_for(action)] == [
        "semantic_scholar"
    ]


def test_brave_provider_maps_results_into_raw_tool_result(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "web": {
                    "results": [
                        {
                            "title": "Brave result",
                            "url": "https://example.com/brave",
                            "description": "Snippet",
                        }
                    ]
                }
            }

    class FakeClient:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "deep_research.providers.search.brave.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = BraveSearchProvider(api_key="secret")
    results = provider.search(["rlhf alternatives"], max_results_per_query=3)

    assert results[0].provider == "brave"
    assert results[0].payload["source_kind"] == "web"
    assert results[0].payload["results"][0]["title"] == "Brave result"


def test_arxiv_provider_maps_results_into_raw_tool_result(monkeypatch) -> None:
    class FakeAuthor:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakePaper:
        title = "Arxiv paper"
        entry_id = "https://arxiv.org/abs/2401.00001"
        summary = "Paper summary"
        published = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
        authors = [FakeAuthor("Author One")]

        def get_short_id(self) -> str:
            return "2401.00001"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def results(self, search):
            return [FakePaper()]

    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Client",
        FakeClient,
    )
    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Search",
        lambda query, max_results: {"query": query, "max_results": max_results},
    )

    provider = ArxivSearchProvider()
    results = provider.search(["alignment survey"], max_results_per_query=2)

    assert results[0].tool_name == "search"
    assert results[0].provider == "arxiv"
    assert results[0].payload["source_kind"] == "paper"
    assert results[0].payload["results"][0]["arxiv_id"] == "2401.00001"


def test_arxiv_provider_filters_stale_results_when_recency_days_is_set(
    monkeypatch,
) -> None:
    class FakeAuthor:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakePaper:
        def __init__(self, short_id: str, published: datetime) -> None:
            self.title = short_id
            self.entry_id = f"https://arxiv.org/abs/{short_id}"
            self.summary = f"Summary for {short_id}"
            self.published = published
            self.authors = [FakeAuthor("Author One")]

        def get_short_id(self) -> str:
            return self.title

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def results(self, search):
            return [
                FakePaper("recent-paper", datetime.now(UTC)),
                FakePaper("stale-paper", datetime(2020, 1, 1, tzinfo=UTC)),
            ]

    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Client",
        FakeClient,
    )
    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Search",
        lambda query, max_results: {"query": query, "max_results": max_results},
    )

    provider = ArxivSearchProvider()
    results = provider.search(["alignment survey"], recency_days=30)

    assert [item["arxiv_id"] for item in results[0].payload["results"]] == [
        "recent-paper"
    ]


def test_arxiv_provider_normalizes_library_failures(monkeypatch) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def results(self, search):
            raise RuntimeError("arxiv unavailable")

    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Client",
        FakeClient,
    )
    monkeypatch.setattr(
        "deep_research.providers.search.arxiv_provider.arxiv.Search",
        lambda query, max_results: {"query": query, "max_results": max_results},
    )

    provider = ArxivSearchProvider()
    results = provider.search(["alignment survey"])

    assert results[0].ok is False
    assert results[0].payload == {"source_kind": "paper", "results": []}
    assert "arxiv unavailable" in results[0].error


def test_semantic_scholar_provider_maps_results_into_raw_tool_result(
    monkeypatch,
) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {
                        "title": "Semantic Scholar paper",
                        "url": "https://example.com/paper",
                        "abstract": "Abstract",
                        "citationCount": 123,
                        "externalIds": {"DOI": "10.1000/test", "ArXiv": "2401.00001"},
                    }
                ]
            }

    class FakeClient:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "deep_research.providers.search.semantic_scholar.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = SemanticScholarProvider(api_key="")
    results = provider.search(["alignment survey"], max_results_per_query=2)

    assert results[0].tool_name == "search"
    assert results[0].provider == "semantic_scholar"
    assert results[0].payload["source_kind"] == "paper"
    assert results[0].payload["results"][0]["doi"] == "10.1000/test"


def test_semantic_scholar_provider_filters_stale_results_when_recency_days_is_set(
    monkeypatch,
) -> None:
    recent_date = datetime.now(UTC).strftime("%Y-%m-%d")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {
                        "title": "Recent paper",
                        "url": "https://example.com/recent",
                        "abstract": "Recent abstract",
                        "year": datetime.now(UTC).year,
                        "publicationDate": recent_date,
                        "citationCount": 10,
                        "externalIds": {},
                    },
                    {
                        "title": "Stale paper",
                        "url": "https://example.com/stale",
                        "abstract": "Stale abstract",
                        "year": 2020,
                        "publicationDate": "2020-01-01",
                        "citationCount": 10,
                        "externalIds": {},
                    },
                ]
            }

    class FakeClient:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        "deep_research.providers.search.semantic_scholar.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = SemanticScholarProvider(api_key="")
    results = provider.search(["alignment survey"], recency_days=30)

    assert [item["title"] for item in results[0].payload["results"]] == ["Recent paper"]


def test_semantic_scholar_provider_normalizes_http_failures(monkeypatch) -> None:
    class FakeClient:
        def get(self, *args, **kwargs):
            raise RuntimeError("semantic scholar unavailable")

    monkeypatch.setattr(
        "deep_research.providers.search.semantic_scholar.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = SemanticScholarProvider(api_key="")
    results = provider.search(["alignment survey"])

    assert results[0].ok is False
    assert results[0].payload == {"source_kind": "paper", "results": []}
    assert "semantic scholar unavailable" in results[0].error


def test_brave_provider_normalizes_http_failures(monkeypatch) -> None:
    class FakeClient:
        def get(self, *args, **kwargs):
            raise RuntimeError("brave unavailable")

    monkeypatch.setattr(
        "deep_research.providers.search.brave.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = BraveSearchProvider(api_key="secret")
    results = provider.search(["rlhf alternatives"])

    assert results[0].ok is False
    assert results[0].payload == {"source_kind": "web", "results": []}
    assert "brave unavailable" in results[0].error


def test_exa_provider_sends_iso_start_published_date(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"results": []}

    class FakeClient:
        def post(self, *args, **kwargs):
            captured["json"] = kwargs["json"]
            return FakeResponse()

    monkeypatch.setattr(
        "deep_research.providers.search.exa_provider.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = ExaSearchProvider(api_key="secret")
    results = provider.search(["recent rlhf alternatives"], recency_days=14)

    payload = captured["json"]

    assert isinstance(payload, dict)
    assert results[0].tool_name == "search"
    assert results[0].provider == "exa"
    assert results[0].payload["source_kind"] == "web"
    start_published_date = payload["startPublishedDate"]
    assert isinstance(start_published_date, str)
    parsed = datetime.fromisoformat(start_published_date.replace("Z", "+00:00"))
    assert parsed.tzinfo == UTC


def test_exa_provider_normalizes_http_failures(monkeypatch) -> None:
    class FakeClient:
        def post(self, *args, **kwargs):
            raise RuntimeError("exa unavailable")

    monkeypatch.setattr(
        "deep_research.providers.search.exa_provider.build_client",
        lambda timeout=20: FakeClient(),
    )

    provider = ExaSearchProvider(api_key="secret")
    results = provider.search(["recent rlhf alternatives"])

    assert results[0].ok is False
    assert results[0].payload == {"source_kind": "web", "results": []}
    assert "exa unavailable" in results[0].error
