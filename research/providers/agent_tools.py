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
from dataclasses import dataclass, field

from research.config import ResearchConfig
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
    """Build an ``AgentToolSurface`` for the subagent slot.

    This is the canonical factory for creating the tool surface.
    """
    return AgentToolSurface(config=config, registry=registry)
