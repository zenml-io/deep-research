from __future__ import annotations

from typing import Protocol

from deep_research.config import ResearchConfig
from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import RawToolResult, SearchAction


class SearchProvider(Protocol):
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
    return RawToolResult(
        tool_name="search",
        provider=provider,
        payload={"source_kind": source_kind, "results": []},
        ok=False,
        error=str(error),
    )


class ProviderRegistry:
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
        providers = [self._providers[name] for name in self._enabled_names]
        return [provider for provider in providers if provider.is_available()]

    def providers_for(
        self,
        action: SearchAction,
        excluded_providers: list[str] | None = None,
        excluded_source_groups: list[SourceGroup] | None = None,
    ) -> list[SearchProvider]:
        providers = self.active_providers()

        # Hard exclusions -- deterministic, no LLM discretion
        if excluded_providers:
            excluded_set = set(excluded_providers)
            providers = [p for p in providers if p.name not in excluded_set]
        if excluded_source_groups:
            excluded_group_set = set(excluded_source_groups)
            providers = [
                p for p in providers if p.source_group not in excluded_group_set
            ]

        # Action-level preferences (from supervisor SearchAction)
        if action.preferred_providers:
            preferred = set(action.preferred_providers)
            providers = [
                provider for provider in providers if provider.name in preferred
            ]
        if action.preferred_source_kinds:
            requested_kinds = set(action.preferred_source_kinds)
            providers = [
                provider
                for provider in providers
                if requested_kinds.intersection(provider.supported_source_kinds)
            ]
        return providers


__all__ = ["ProviderRegistry", "SearchProvider", "failure_result"]
