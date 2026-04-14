"""Tests for V2 search providers: protocol, registry, and individual providers.

All tests are offline — no real HTTP calls. Provider search methods are
tested with mocked httpx responses or by verifying the protocol/registry
logic directly.

Note: pytest-asyncio is not available, so async tests use asyncio.run().
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from research.config import ResearchConfig, ResearchSettings
from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    _backoff_delay,
    _is_retryable_http_error,
    build_async_client,
    request_with_retry,
)
from research.providers.search import (
    ProviderRegistry,
    SearchProvider,
    SearchResult,
)


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_basic_construction(self):
        r = SearchResult(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            provider="test",
        )
        assert r.url == "https://example.com"
        assert r.title == "Example"
        assert r.snippet == "A snippet"
        assert r.provider == "test"
        assert r.raw_metadata == {}

    def test_with_raw_metadata(self):
        r = SearchResult(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            provider="test",
            raw_metadata={"doi": "10.1234/test"},
        )
        assert r.raw_metadata == {"doi": "10.1234/test"}

    def test_is_frozen(self):
        r = SearchResult(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            provider="test",
        )
        with pytest.raises(AttributeError):
            r.url = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self):
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.backoff_base_seconds == 0.5
        assert p.backoff_factor == 2.0
        assert 429 in p.retryable_status_codes

    def test_negative_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            RetryPolicy(max_retries=-1)

    def test_negative_backoff_raises(self):
        with pytest.raises(ValueError, match="backoff_base_seconds"):
            RetryPolicy(backoff_base_seconds=-0.1)

    def test_factor_below_one_raises(self):
        with pytest.raises(ValueError, match="backoff_factor"):
            RetryPolicy(backoff_factor=0.5)


class TestBackoffDelay:
    def test_first_attempt(self):
        p = RetryPolicy(backoff_base_seconds=1.0, backoff_factor=2.0)
        assert _backoff_delay(p, 0) == 1.0

    def test_second_attempt(self):
        p = RetryPolicy(backoff_base_seconds=1.0, backoff_factor=2.0)
        assert _backoff_delay(p, 1) == 2.0

    def test_third_attempt(self):
        p = RetryPolicy(backoff_base_seconds=0.5, backoff_factor=2.0)
        assert _backoff_delay(p, 2) == 2.0


class TestIsRetryableHttpError:
    def test_429_is_retryable(self):
        response = httpx.Response(429, request=httpx.Request("GET", "http://test"))
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY) is True

    def test_500_is_retryable(self):
        response = httpx.Response(500, request=httpx.Request("GET", "http://test"))
        exc = httpx.HTTPStatusError(
            "server error", request=response.request, response=response
        )
        assert _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY) is True

    def test_404_not_retryable(self):
        response = httpx.Response(404, request=httpx.Request("GET", "http://test"))
        exc = httpx.HTTPStatusError(
            "not found", request=response.request, response=response
        )
        assert _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY) is False

    def test_connection_error_is_retryable(self):
        exc = httpx.ConnectError("connection refused")
        assert _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY) is True

    def test_non_httpx_error_not_retryable(self):
        exc = ValueError("bad value")
        assert _is_retryable_http_error(exc, DEFAULT_RETRY_POLICY) is False


# ---------------------------------------------------------------------------
# build_async_client
# ---------------------------------------------------------------------------


class TestBuildAsyncClient:
    def test_returns_async_client(self):
        client = build_async_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_custom_timeout(self):
        client = build_async_client(timeout=60)
        assert client.timeout.connect == 60.0


# ---------------------------------------------------------------------------
# request_with_retry (async)
# ---------------------------------------------------------------------------


class TestRequestWithRetry:
    def test_successful_request(self):
        async def _run():
            mock_response = httpx.Response(
                200,
                json={"ok": True},
                request=httpx.Request("GET", "http://test"),
            )
            client = AsyncMock(spec=httpx.AsyncClient)
            client.request = AsyncMock(return_value=mock_response)

            result = await request_with_retry(client, "GET", "http://test")
            assert result.status_code == 200

        asyncio.run(_run())

    def test_retries_on_429(self):
        async def _run():
            error_response = httpx.Response(
                429,
                request=httpx.Request("GET", "http://test"),
            )
            ok_response = httpx.Response(
                200,
                json={"ok": True},
                request=httpx.Request("GET", "http://test"),
            )

            call_count = 0

            async def mock_request(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise httpx.HTTPStatusError(
                        "rate limited",
                        request=error_response.request,
                        response=error_response,
                    )
                return ok_response

            client = AsyncMock(spec=httpx.AsyncClient)
            client.request = mock_request

            fast_policy = RetryPolicy(max_retries=2, backoff_base_seconds=0.01)
            result = await request_with_retry(
                client, "GET", "http://test", retry_policy=fast_policy
            )
            assert result.status_code == 200
            assert call_count == 2

        asyncio.run(_run())

    def test_raises_after_exhausted_retries(self):
        async def _run():
            error_response = httpx.Response(
                500,
                request=httpx.Request("GET", "http://test"),
            )

            async def mock_request(*args, **kwargs):
                raise httpx.HTTPStatusError(
                    "server error",
                    request=error_response.request,
                    response=error_response,
                )

            client = AsyncMock(spec=httpx.AsyncClient)
            client.request = mock_request

            fast_policy = RetryPolicy(max_retries=1, backoff_base_seconds=0.01)
            with pytest.raises(httpx.HTTPStatusError):
                await request_with_retry(
                    client, "GET", "http://test", retry_policy=fast_policy
                )

        asyncio.run(_run())

    def test_does_not_retry_non_retryable(self):
        async def _run():
            error_response = httpx.Response(
                404,
                request=httpx.Request("GET", "http://test"),
            )

            call_count = 0

            async def mock_request(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise httpx.HTTPStatusError(
                    "not found",
                    request=error_response.request,
                    response=error_response,
                )

            client = AsyncMock(spec=httpx.AsyncClient)
            client.request = mock_request

            with pytest.raises(httpx.HTTPStatusError):
                await request_with_retry(client, "GET", "http://test")
            assert call_count == 1

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# ProviderRegistry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def _make_config(self, providers: list[str]) -> ResearchConfig:
        settings = ResearchSettings(enabled_providers=",".join(providers))
        return ResearchConfig.for_tier("quick", settings=settings)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown search providers.*bogus"):
            ProviderRegistry(self._make_config(["arxiv", "bogus"]))

    def test_registry_builds_enabled_providers(self):
        reg = ProviderRegistry(self._make_config(["arxiv", "semantic_scholar"]))
        names = list(reg.all_providers.keys())
        assert "arxiv" in names
        assert "semantic_scholar" in names
        assert "brave" not in names

    def test_active_providers_filters_unavailable(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        reg = ProviderRegistry(self._make_config(["brave", "arxiv"]))
        active = reg.active_providers()
        active_names = [p.name for p in active]
        assert "arxiv" in active_names
        # brave should be excluded (no API key)
        assert "brave" not in active_names

    def test_active_providers_includes_keyed_brave(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        reg = ProviderRegistry(self._make_config(["brave", "arxiv"]))
        active_names = [p.name for p in reg.active_providers()]
        assert "brave" in active_names
        assert "arxiv" in active_names

    def test_get_provider_returns_provider(self):
        reg = ProviderRegistry(self._make_config(["arxiv"]))
        p = reg.get_provider("arxiv")
        assert p is not None
        assert p.name == "arxiv"

    def test_get_provider_returns_none_for_missing(self):
        reg = ProviderRegistry(self._make_config(["arxiv"]))
        assert reg.get_provider("brave") is None

    def test_all_four_providers_can_be_registered(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "k")
        monkeypatch.setenv("EXA_API_KEY", "k")
        reg = ProviderRegistry(
            self._make_config(["brave", "exa", "arxiv", "semantic_scholar"])
        )
        assert len(reg.all_providers) == 4

    def test_empty_enabled_providers(self):
        settings = ResearchSettings(enabled_providers="")
        cfg = ResearchConfig.for_tier("quick", settings=settings)
        reg = ProviderRegistry(cfg)
        assert reg.active_providers() == []


# ---------------------------------------------------------------------------
# SearchProvider protocol
# ---------------------------------------------------------------------------


class TestSearchProviderProtocol:
    """Verify that concrete providers satisfy the SearchProvider protocol."""

    def test_brave_is_search_provider(self):
        from research.providers.brave import BraveSearchProvider

        assert isinstance(BraveSearchProvider(), SearchProvider)

    def test_arxiv_is_search_provider(self):
        from research.providers.arxiv_provider import ArxivSearchProvider

        assert isinstance(ArxivSearchProvider(), SearchProvider)

    def test_semantic_scholar_is_search_provider(self):
        from research.providers.semantic_scholar import SemanticScholarProvider

        assert isinstance(SemanticScholarProvider(), SearchProvider)

    def test_exa_is_search_provider(self):
        from research.providers.exa_provider import ExaSearchProvider

        assert isinstance(ExaSearchProvider(), SearchProvider)


# ---------------------------------------------------------------------------
# BraveSearchProvider
# ---------------------------------------------------------------------------


class TestBraveProvider:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        from research.providers.brave import BraveSearchProvider

        p = BraveSearchProvider()
        assert p.is_available() is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        from research.providers.brave import BraveSearchProvider

        p = BraveSearchProvider()
        assert p.is_available() is True

    def test_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        from research.providers.brave import BraveSearchProvider

        p = BraveSearchProvider()
        results = asyncio.run(p.search(["test query"]))
        assert results == []

    def test_parses_brave_response(self, monkeypatch):
        monkeypatch.setenv("BRAVE_API_KEY", "test-key")
        from research.providers.brave import BraveSearchProvider

        brave_payload = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "description": "Snippet 1",
                        "page_age": "2024-01-01",
                    },
                    {
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "description": "Snippet 2",
                    },
                ]
            }
        }

        mock_response = httpx.Response(
            200,
            json=brave_payload,
            request=httpx.Request(
                "GET", "https://api.search.brave.com/res/v1/web/search"
            ),
        )

        async def mock_retry(*args, **kwargs):
            return mock_response

        p = BraveSearchProvider()

        async def _run():
            with patch(
                "research.providers.brave.request_with_retry",
                side_effect=mock_retry,
            ):
                return await p.search(["test query"])

        results = asyncio.run(_run())

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert results[0].provider == "brave"

    def test_recency_to_freshness(self):
        from research.providers.brave import _recency_to_freshness

        assert _recency_to_freshness(None) is None
        assert _recency_to_freshness(1) == "pd"
        assert _recency_to_freshness(5) == "pw"
        assert _recency_to_freshness(7) == "pw"
        assert _recency_to_freshness(14) == "pm"
        assert _recency_to_freshness(31) == "pm"
        assert _recency_to_freshness(60) == "py"


# ---------------------------------------------------------------------------
# ArxivSearchProvider
# ---------------------------------------------------------------------------


class TestArxivProvider:
    def test_always_available(self):
        from research.providers.arxiv_provider import ArxivSearchProvider

        assert ArxivSearchProvider().is_available() is True

    def test_parses_arxiv_results(self):
        from datetime import UTC, datetime

        from research.providers.arxiv_provider import ArxivSearchProvider

        mock_paper = MagicMock()
        mock_paper.entry_id = "http://arxiv.org/abs/2301.12345v1"
        mock_paper.title = "Test Paper"
        mock_paper.summary = "A test paper about testing"
        mock_paper.published = datetime(2024, 1, 15, tzinfo=UTC)
        mock_paper.get_short_id.return_value = "2301.12345v1"
        mock_author = MagicMock()
        mock_author.name = "Test Author"
        mock_paper.authors = [mock_author]

        async def _run():
            with patch("research.providers.arxiv_provider.arxiv") as mock_arxiv:
                mock_client = MagicMock()
                mock_arxiv.Client.return_value = mock_client
                mock_client.results.return_value = [mock_paper]

                p = ArxivSearchProvider()
                return await p.search(["test query"])

        results = asyncio.run(_run())

        assert len(results) == 1
        assert results[0].title == "Test Paper"
        assert results[0].url == "http://arxiv.org/abs/2301.12345v1"
        assert results[0].provider == "arxiv"
        assert results[0].raw_metadata["arxiv_id"] == "2301.12345v1"

    def test_recency_filters_old_papers(self):
        from datetime import UTC, datetime

        from research.providers.arxiv_provider import ArxivSearchProvider

        old_paper = MagicMock()
        old_paper.entry_id = "http://arxiv.org/abs/2001.00001v1"
        old_paper.title = "Old Paper"
        old_paper.summary = "Old"
        old_paper.published = datetime(2020, 1, 1, tzinfo=UTC)
        old_paper.get_short_id.return_value = "2001.00001v1"
        old_paper.authors = []

        async def _run():
            with patch("research.providers.arxiv_provider.arxiv") as mock_arxiv:
                mock_client = MagicMock()
                mock_arxiv.Client.return_value = mock_client
                mock_client.results.return_value = [old_paper]

                p = ArxivSearchProvider()
                return await p.search(["test"], recency_days=30)

        results = asyncio.run(_run())
        assert len(results) == 0

    def test_handles_exception_gracefully(self):
        from research.providers.arxiv_provider import ArxivSearchProvider

        async def _run():
            with patch("research.providers.arxiv_provider.arxiv") as mock_arxiv:
                mock_client = MagicMock()
                mock_arxiv.Client.return_value = mock_client
                mock_client.results.side_effect = RuntimeError("network error")

                p = ArxivSearchProvider()
                return await p.search(["test"])

        results = asyncio.run(_run())
        assert results == []


# ---------------------------------------------------------------------------
# SemanticScholarProvider
# ---------------------------------------------------------------------------


class TestSemanticScholarProvider:
    def test_always_available(self):
        from research.providers.semantic_scholar import SemanticScholarProvider

        assert SemanticScholarProvider().is_available() is True

    def test_sends_api_key_header_when_set(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "test-s2-key")
        from research.providers.semantic_scholar import SemanticScholarProvider

        p = SemanticScholarProvider()
        assert p._api_key == "test-s2-key"

    def test_parses_s2_response(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        from research.providers.semantic_scholar import SemanticScholarProvider

        s2_payload = {
            "data": [
                {
                    "title": "Paper A",
                    "url": "https://semanticscholar.org/paper/A",
                    "abstract": "Abstract A",
                    "year": 2024,
                    "citationCount": 150,
                    "publicationDate": "2024-03-15",
                    "externalIds": {"DOI": "10.1234/a", "ArXiv": "2401.00001"},
                }
            ]
        }
        mock_response = httpx.Response(
            200,
            json=s2_payload,
            request=httpx.Request(
                "GET",
                "https://api.semanticscholar.org/graph/v1/paper/search",
            ),
        )

        async def mock_retry(*args, **kwargs):
            return mock_response

        p = SemanticScholarProvider()

        async def _run():
            with patch(
                "research.providers.semantic_scholar.request_with_retry",
                side_effect=mock_retry,
            ):
                return await p.search(["test"])

        results = asyncio.run(_run())

        assert len(results) == 1
        assert results[0].title == "Paper A"
        assert results[0].provider == "semantic_scholar"
        assert results[0].raw_metadata["doi"] == "10.1234/a"
        assert results[0].raw_metadata["arxiv_id"] == "2401.00001"

    def test_handles_exception_gracefully(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        from research.providers.semantic_scholar import SemanticScholarProvider

        async def mock_retry(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        p = SemanticScholarProvider()

        async def _run():
            with patch(
                "research.providers.semantic_scholar.request_with_retry",
                side_effect=mock_retry,
            ):
                return await p.search(["test"])

        results = asyncio.run(_run())
        assert results == []


# ---------------------------------------------------------------------------
# ExaSearchProvider
# ---------------------------------------------------------------------------


class TestExaProvider:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        from research.providers.exa_provider import ExaSearchProvider

        assert ExaSearchProvider().is_available() is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
        from research.providers.exa_provider import ExaSearchProvider

        assert ExaSearchProvider().is_available() is True

    def test_returns_empty_when_unavailable(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        from research.providers.exa_provider import ExaSearchProvider

        p = ExaSearchProvider()
        results = asyncio.run(p.search(["test query"]))
        assert results == []

    def test_parses_exa_response(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
        from research.providers.exa_provider import ExaSearchProvider

        exa_payload = {
            "results": [
                {
                    "title": "Exa Result",
                    "url": "https://example.com/exa",
                    "text": "Exa snippet text",
                    "publishedDate": "2024-01-15",
                },
            ]
        }
        mock_response = httpx.Response(
            200,
            json=exa_payload,
            request=httpx.Request("POST", "https://api.exa.ai/search"),
        )

        async def mock_retry(*args, **kwargs):
            return mock_response

        p = ExaSearchProvider()

        async def _run():
            with patch(
                "research.providers.exa_provider.request_with_retry",
                side_effect=mock_retry,
            ):
                return await p.search(["test"])

        results = asyncio.run(_run())

        assert len(results) == 1
        assert results[0].title == "Exa Result"
        assert results[0].snippet == "Exa snippet text"
        assert results[0].provider == "exa"

    def test_handles_exception_gracefully(self, monkeypatch):
        monkeypatch.setenv("EXA_API_KEY", "test-exa-key")
        from research.providers.exa_provider import ExaSearchProvider

        async def mock_retry(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        p = ExaSearchProvider()

        async def _run():
            with patch(
                "research.providers.exa_provider.request_with_retry",
                side_effect=mock_retry,
            ):
                return await p.search(["test"])

        results = asyncio.run(_run())
        assert results == []


# ---------------------------------------------------------------------------
# Re-exports from __init__
# ---------------------------------------------------------------------------


class TestProviderReExports:
    def test_search_result_importable_from_providers(self):
        from research.providers import SearchResult

        assert SearchResult is not None

    def test_registry_importable_from_providers(self):
        from research.providers import ProviderRegistry

        assert ProviderRegistry is not None

    def test_search_provider_importable_from_providers(self):
        from research.providers import SearchProvider

        assert SearchProvider is not None

    def test_retry_policy_importable_from_providers(self):
        from research.providers import RetryPolicy

        assert RetryPolicy is not None

    def test_build_async_client_importable(self):
        from research.providers import build_async_client

        assert callable(build_async_client)
