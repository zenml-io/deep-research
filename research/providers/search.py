"""SearchProvider Protocol, SearchResult dataclass, and ProviderRegistry.

This module defines the core search abstraction for V2:
- ``SearchResult``: typed result from any search provider
- ``SearchProvider``: Protocol every provider must satisfy
- ``ProviderRegistry``: resolves enabled providers from config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from research.config import ResearchConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result returned by a provider."""

    url: str
    title: str
    snippet: str
    provider: str
    raw_metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol every search-provider adapter must satisfy."""

    name: str

    def is_available(self) -> bool:
        """Return True if the provider is configured and usable."""
        ...

    async def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Run *queries* and return a flat list of results.

        Must not raise — failed individual queries should be logged
        and skipped, not bubble up.
        """
        ...


# ---------------------------------------------------------------------------
# Known provider names → lazy factory functions
# ---------------------------------------------------------------------------


def _build_brave() -> SearchProvider:
    from research.providers.brave import BraveSearchProvider

    return BraveSearchProvider()


def _build_arxiv() -> SearchProvider:
    from research.providers.arxiv_provider import ArxivSearchProvider

    return ArxivSearchProvider()


def _build_semantic_scholar() -> SearchProvider:
    from research.providers.semantic_scholar import SemanticScholarProvider

    return SemanticScholarProvider()


def _build_exa() -> SearchProvider:
    from research.providers.exa_provider import ExaSearchProvider

    return ExaSearchProvider()


_KNOWN_PROVIDERS: dict[str, Callable[[], SearchProvider]] = {
    "brave": _build_brave,
    "arxiv": _build_arxiv,
    "semantic_scholar": _build_semantic_scholar,
    "exa": _build_exa,
}


class ProviderRegistry:
    """Registry of search-provider adapters for a given ``ResearchConfig``."""

    def __init__(self, config: ResearchConfig) -> None:
        # Validate that all enabled providers are known.
        unknown = [
            name for name in config.enabled_providers if name not in _KNOWN_PROVIDERS
        ]
        if unknown:
            raise ValueError(
                f"Unknown search providers: {', '.join(sorted(set(unknown)))}"
            )

        # Eagerly build all enabled providers.
        self._providers: dict[str, SearchProvider] = {}
        for name in config.enabled_providers:
            self._providers[name] = _KNOWN_PROVIDERS[name]()

    def active_providers(self) -> list[SearchProvider]:
        """Return enabled providers that report themselves as available."""
        return [p for p in self._providers.values() if p.is_available()]

    def get_provider(self, name: str) -> SearchProvider | None:
        """Look up a specific provider by name, or None if not registered."""
        return self._providers.get(name)

    @property
    def all_providers(self) -> dict[str, SearchProvider]:
        """Return all registered providers (available or not)."""
        return dict(self._providers)
