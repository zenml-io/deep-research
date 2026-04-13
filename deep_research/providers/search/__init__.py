from __future__ import annotations

from typing import Protocol

from deep_research.config import ResearchConfig
from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import RawToolResult, SearchAction

_WEB_FIRST_RANK: dict[str, int] = {
    "exa": 0,
    "brave": 1,
    "semantic_scholar": 2,
    "arxiv": 3,
}

_PAPER_FIRST_RANK: dict[str, int] = {
    "semantic_scholar": 0,
    "arxiv": 1,
    "exa": 2,
    "brave": 3,
}

_ACADEMIC_MARKERS: frozenset[str] = frozenset(
    {
        "paper",
        "papers",
        "literature review",
        "survey",
        "survey paper",
        "arxiv",
        "semantic scholar",
        "citation",
        "citations",
        "peer reviewed",
        "peer-reviewed",
        "theorem",
        "proof",
    }
)


class SearchProvider(Protocol):
    """Protocol every search-provider adapter must satisfy."""

    name: str
    source_group: SourceGroup
    supported_source_kinds: tuple[SourceKind, ...]

    def is_available(self) -> bool: ...

    def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[RawToolResult]: ...

    def estimate_cost_usd(self, query_count: int) -> float: ...


def failure_result(provider: str, source_kind: str, error: Exception) -> RawToolResult:
    """Return a failed `RawToolResult` for use in provider error handlers."""
    return RawToolResult(
        tool_name="search",
        provider=provider,
        payload={"source_kind": source_kind, "results": []},
        ok=False,
        error=str(error),
    )


class ProviderRegistry:
    """Registry of all search-provider adapters for a given `ResearchConfig`."""

    def __init__(self, config: ResearchConfig) -> None:
        from deep_research.providers.search.arxiv_provider import ArxivSearchProvider
        from deep_research.providers.search.brave import BraveSearchProvider
        from deep_research.providers.search.exa_provider import ExaSearchProvider
        from deep_research.providers.search.semantic_scholar import (
            SemanticScholarProvider,
        )

        self._providers: dict[str, SearchProvider] = {
            "brave": BraveSearchProvider(api_key=config.brave_api_key),
            "arxiv": ArxivSearchProvider(),
            "semantic_scholar": SemanticScholarProvider(
                api_key=config.semantic_scholar_api_key
            ),
            "exa": ExaSearchProvider(api_key=config.exa_api_key),
        }
        unknown_names = [
            name for name in config.enabled_providers if name not in self._providers
        ]
        if unknown_names:
            unknown_display = ", ".join(sorted(set(unknown_names)))
            raise ValueError(f"Unknown search providers: {unknown_display}")
        self._enabled_names = [
            name for name in config.enabled_providers if name in self._providers
        ]

    def active_providers(self) -> list[SearchProvider]:
        """Return enabled providers that report themselves as available."""
        providers = [self._providers[name] for name in self._enabled_names]
        return [p for p in providers if p.is_available()]

    @classmethod
    def _looks_paper_first(
        cls,
        action: SearchAction,
        preferred_source_groups: list[SourceGroup] | None = None,
    ) -> bool:
        if action.preferred_source_kinds:
            return SourceKind.PAPER in action.preferred_source_kinds
        if preferred_source_groups and SourceGroup.PAPERS in preferred_source_groups:
            return True
        return any(marker in action.query.casefold() for marker in _ACADEMIC_MARKERS)

    @classmethod
    def _provider_sort_key(
        cls,
        provider: SearchProvider,
        action: SearchAction,
        preferred_source_groups: list[SourceGroup] | None = None,
        preferred_providers: list[str] | None = None,
    ) -> tuple[int, int, int, int, str]:
        preferred_provider_rank = (
            preferred_providers.index(provider.name)
            if preferred_providers and provider.name in preferred_providers
            else len(preferred_providers or []) + 1
        )

        if preferred_source_groups:
            try:
                preferred_group_rank = preferred_source_groups.index(
                    provider.source_group
                )
            except ValueError:
                preferred_group_rank = len(preferred_source_groups) + 1
        else:
            preferred_group_rank = 0

        rank_table = (
            _PAPER_FIRST_RANK
            if cls._looks_paper_first(action, preferred_source_groups)
            else _WEB_FIRST_RANK
        )
        provider_family_rank = rank_table.get(provider.name, 99)

        explicit_kind_rank = 0
        if action.preferred_source_kinds:
            explicit_kind_rank = (
                0
                if set(action.preferred_source_kinds).intersection(
                    provider.supported_source_kinds
                )
                else 1
            )

        return (
            preferred_provider_rank,
            explicit_kind_rank,
            preferred_group_rank,
            provider_family_rank,
            provider.name,
        )

    def providers_for(
        self,
        action: SearchAction,
        excluded_providers: list[str] | None = None,
        excluded_source_groups: list[SourceGroup] | None = None,
        preferred_source_groups: list[SourceGroup] | None = None,
        preferred_providers: list[str] | None = None,
    ) -> list[SearchProvider]:
        """Return active providers applicable to *action*, sorted by preference."""
        providers = self.active_providers()

        if excluded_providers:
            excluded_set = set(excluded_providers)
            providers = [p for p in providers if p.name not in excluded_set]
        if excluded_source_groups:
            excluded_group_set = set(excluded_source_groups)
            providers = [
                p for p in providers if p.source_group not in excluded_group_set
            ]

        if action.preferred_providers:
            preferred = set(action.preferred_providers)
            providers = [p for p in providers if p.name in preferred]
        if action.preferred_source_kinds:
            requested_kinds = set(action.preferred_source_kinds)
            providers = [
                p
                for p in providers
                if requested_kinds.intersection(p.supported_source_kinds)
            ]

        return sorted(
            providers,
            key=lambda p: self._provider_sort_key(
                p,
                action,
                preferred_source_groups=preferred_source_groups,
                preferred_providers=preferred_providers,
            ),
        )


__all__ = ["ProviderRegistry", "SearchProvider", "failure_result"]
