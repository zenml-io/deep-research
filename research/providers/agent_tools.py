"""Inward MCP-style tool surface for subagent slots.

Builds a collection of tools (``search``, ``fetch``, and conditionally
``code_exec``) that subagents can call during their execution.

Only the **subagent** slot is bound to this surface. The supervisor,
generator, reviewer, finalizer, and judge have **no** tool access.  All
web requests and sandbox execution happen inside ``run_subagent``
checkpoints — this is a structural guard against supervisor-as-executor
creep.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from research.config import ResearchConfig
from research.contracts.package import (
    ProviderResolution,
    ToolProviderManifest,
    ToolResolution,
)
from research.providers.code_exec import (
    CodeExecResult,
    SandboxExecutor,
    SandboxNotAvailableError,
)
from research.providers.fetch import fetch_url_content
from research.providers.search import ProviderRegistry, SearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Wrapper around a tool invocation result for uniform error handling."""

    tool: str
    success: bool
    data: object = None
    error: str | None = None


class AgentToolSurface:
    """Tool surface exposed to the subagent slot only.

    Provides ``search``, ``fetch``, and (when sandbox is enabled)
    ``code_exec`` capabilities.

    Parameters
    ----------
    config:
        Frozen research config for the current run.
    registry:
        Provider registry with enabled search providers.
    """

    def __init__(
        self,
        config: ResearchConfig,
        registry: ProviderRegistry,
    ) -> None:
        self._config = config
        self._registry = registry

        # Set up sandbox executor only if enabled and backend is configured.
        self._sandbox: SandboxExecutor | None = None
        if config.sandbox_enabled and config.sandbox_backend:
            self._sandbox = SandboxExecutor(backend=config.sandbox_backend)

    # -----------------------------------------------------------------------
    # Tool discovery
    # -----------------------------------------------------------------------

    def available_tools(self) -> list[str]:
        """Return names of tools available in this surface.

        Always includes ``"search"`` and ``"fetch"``.
        Includes ``"code_exec"`` only when sandbox is enabled and configured.
        """
        tools = ["search", "fetch"]
        if self._sandbox is not None:
            tools.append("code_exec")
        return tools

    # -----------------------------------------------------------------------
    # search
    # -----------------------------------------------------------------------

    async def search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Run *queries* across all active search providers.

        Delegates to each provider in the registry and flattens the results.
        Individual provider failures are logged and skipped.
        """
        active = self._registry.active_providers()
        if not active:
            logger.warning("No active search providers available")
            return []

        all_results: list[SearchResult] = []
        for provider in active:
            try:
                results = await provider.search(
                    queries,
                    max_results_per_query=max_results_per_query,
                    recency_days=recency_days,
                )
                all_results.extend(results)
            except Exception:
                logger.exception(
                    "Search provider %s failed for queries %s",
                    provider.name,
                    queries,
                )
        return all_results

    # -----------------------------------------------------------------------
    # fetch
    # -----------------------------------------------------------------------

    async def fetch(
        self,
        url: str,
        *,
        timeout_sec: int = 15,
        max_chars: int = 50_000,
    ) -> str | None:
        """Fetch URL content and return extracted plain text.

        Returns ``None`` on failure (network error, non-text content, PDF, etc.).
        """
        try:
            return await fetch_url_content(
                url, timeout_sec=timeout_sec, max_chars=max_chars
            )
        except Exception:
            logger.exception("Fetch failed for %s", url)
            return None

    # -----------------------------------------------------------------------
    # PydanticAI tool export
    # -----------------------------------------------------------------------

    def as_pydantic_tools(self) -> list[Callable[..., Any]]:
        """Return plain async callables that PydanticAI can bind as agent tools.

        Each callable has proper type hints for PydanticAI's tool introspection.
        Only includes tools that are available (code_exec omitted when sandbox
        is not configured).
        """
        tools: list[Callable[..., Any]] = []

        async def search(
            queries: list[str],
            max_results_per_query: int = 10,
        ) -> list[dict]:
            """Search across all active providers. Returns list of result dicts."""
            results = await self.search(
                queries, max_results_per_query=max_results_per_query
            )
            return [
                {
                    "url": r.url,
                    "title": r.title,
                    "snippet": r.snippet,
                    "provider": r.provider,
                }
                for r in results
            ]

        async def fetch(url: str) -> str:
            """Fetch URL content and return extracted plain text."""
            content = await self.fetch(url)
            return content or ""

        tools.append(search)
        tools.append(fetch)

        if self._sandbox is not None:

            async def code_exec(
                code: str,
                language: str = "python",
            ) -> dict:
                """Execute code in a sandboxed environment."""
                result = await self.code_exec(code, language=language)
                if result is None:
                    return {"success": False, "error": "Sandbox not available"}
                return {
                    "success": True,
                    "stdout": getattr(result, "stdout", ""),
                    "stderr": getattr(result, "stderr", ""),
                }

            tools.append(code_exec)

        return tools

    # -----------------------------------------------------------------------
    # code_exec
    # -----------------------------------------------------------------------

    async def code_exec(
        self,
        code: str,
        *,
        language: str = "python",
        input_data: dict[str, str] | None = None,
    ) -> CodeExecResult | None:
        """Execute code in the sandbox.

        Returns ``None`` if sandbox is not enabled/configured.
        Raises ``SandboxNotAvailableError`` if the backend is a stub.
        """
        if self._sandbox is None:
            logger.debug("code_exec called but sandbox is not enabled/configured")
            return None

        return await self._sandbox.execute(
            code, language=language, input_data=input_data
        )


def build_tool_surface(
    config: ResearchConfig,
    registry: ProviderRegistry,
) -> AgentToolSurface:
    """Build an ``AgentToolSurface`` for the subagent slot."""
    return AgentToolSurface(config=config, registry=registry)


def _surface_tool_names(surface: AgentToolSurface | object | None) -> list[str]:
    """Best-effort tool-name extraction for a resolved surface."""
    if surface is None:
        return []

    if hasattr(surface, "available_tools") and callable(surface.available_tools):
        return list(surface.available_tools())

    if hasattr(surface, "as_pydantic_tools") and callable(surface.as_pydantic_tools):
        return [
            getattr(tool, "__name__", str(tool))
            for tool in surface.as_pydantic_tools()
        ]

    return []


def build_tool_provider_manifest(
    config: ResearchConfig,
    registry: ProviderRegistry | None = None,
    surface: AgentToolSurface | object | None = None,
    *,
    degradation_reasons: list[str] | None = None,
) -> ToolProviderManifest:
    """Build a durable manifest of provider/tool setup for the current run."""
    configured = list(config.enabled_providers)
    instantiated = sorted(registry.all_providers.keys()) if registry else []

    provider_resolutions: list[ProviderResolution] = []
    active_providers: list[str] = []
    if registry is not None:
        for name, provider in registry.all_providers.items():
            try:
                available = provider.is_available()
            except Exception as exc:
                available = False
                provider_resolutions.append(
                    ProviderResolution(
                        provider=name,
                        instantiated=True,
                        available=False,
                        reason=f"availability_check_failed: {exc}",
                    )
                )
                continue

            if available:
                active_providers.append(name)
                reason = None
            else:
                reason = "provider_reported_unavailable"
            provider_resolutions.append(
                ProviderResolution(
                    provider=name,
                    instantiated=True,
                    available=available,
                    reason=reason,
                )
            )

        for name, reason in registry.build_errors.items():
            provider_resolutions.append(
                ProviderResolution(
                    provider=name,
                    instantiated=False,
                    available=False,
                    reason=f"provider_build_failed: {reason}",
                )
            )
    else:
        provider_resolutions.extend(
            ProviderResolution(
                provider=name,
                instantiated=False,
                available=False,
                reason="provider_registry_not_built",
            )
            for name in configured
        )

    available_tools = _surface_tool_names(surface)
    tool_resolutions = [
        ToolResolution(
            tool="search",
            enabled="search" in available_tools,
            reason=None if "search" in available_tools else "tool_surface_unavailable",
        ),
        ToolResolution(
            tool="fetch",
            enabled="fetch" in available_tools,
            reason=None if "fetch" in available_tools else "tool_surface_unavailable",
        ),
        ToolResolution(
            tool="code_exec",
            enabled="code_exec" in available_tools,
            reason=(
                None
                if "code_exec" in available_tools
                else (
                    "sandbox_disabled"
                    if not config.sandbox_enabled
                    else "sandbox_backend_not_configured"
                    if not config.sandbox_backend
                    else "tool_surface_unavailable"
                )
            ),
        ),
    ]

    reasons = list(degradation_reasons or [])
    if registry is not None and not active_providers and configured:
        reasons.append("no_active_search_providers")

    return ToolProviderManifest(
        configured_providers=configured,
        instantiated_providers=instantiated,
        active_providers=sorted(active_providers),
        available_tools=available_tools,
        provider_resolutions=sorted(
            provider_resolutions,
            key=lambda resolution: resolution.provider,
        ),
        tool_resolutions=tool_resolutions,
        degradation_reasons=reasons,
    )
