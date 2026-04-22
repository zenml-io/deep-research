"""Tool-surface resolution checkpoint.

Wraps provider-registry construction and tool-surface build in a Kitaru
``@checkpoint(type="tool_call")`` so the environment probing (``is_available``
calls on each configured provider, API-key reads, HTTP client setup) runs
**once** per durable run instead of re-running on every replay.

NOTE: do NOT add ``from __future__ import annotations`` here. ZenML's
materializer registry inspects the concrete return type of a @checkpoint
at step-registration time and raises if annotations are strings.
"""

import logging
from typing import Optional

from kitaru import checkpoint

from research.config.settings import ResearchConfig
from research.contracts.package import (
    SubagentToolSpec,
    ToolSurfaceResolution,
)
from research.providers.agent_tools import (
    build_tool_provider_manifest,
    build_tool_surface,
)
from research.providers.search import ProviderRegistry

logger = logging.getLogger(__name__)


@checkpoint(type="tool_call")
def resolve_tool_surface(cfg: ResearchConfig) -> ToolSurfaceResolution:
    """Probe providers and resolve the replay-stable subagent tool spec.

    Returns both the ``SubagentToolSpec`` (threaded into ``run_subagent``)
    and the ``ToolProviderManifest`` (persisted in the investigation
    package for audit).  Provider instantiation failures degrade: the
    manifest records the reason and the spec is set to ``None`` so
    subagents run without tools rather than crashing the flow.
    """
    registry: Optional[ProviderRegistry] = None
    surface = None
    spec: Optional[SubagentToolSpec] = None
    degradation_reasons: list[str] = []

    try:
        registry = ProviderRegistry(cfg)
    except Exception as exc:
        logger.warning("Failed to build provider registry: %s", exc)
        degradation_reasons.append(f"provider_registry_failed: {exc}")
    else:
        try:
            surface = build_tool_surface(cfg, registry)
            spec = SubagentToolSpec(
                enabled_providers=list(cfg.enabled_providers),
                sandbox_enabled=cfg.sandbox_enabled,
                sandbox_backend=cfg.sandbox_backend,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build tool surface; subagents will run without tools: %s",
                exc,
            )
            degradation_reasons.append(f"tool_surface_build_failed: {exc}")

    manifest = build_tool_provider_manifest(
        cfg,
        registry,
        surface,
        degradation_reasons=degradation_reasons,
    )
    return ToolSurfaceResolution(spec=spec, manifest=manifest)
