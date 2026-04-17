"""Investigation and council package contracts.

These are the top-level output types — durable, versioned bundles
that capture everything produced during a research run.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from pydantic import Field

from research.contracts.base import StrictBase
from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.plan import ResearchPlan
from research.contracts.reports import (
    CritiqueReport,
    DraftReport,
    FinalReport,
    VerificationReport,
)

Tier = Literal["quick", "standard", "deep", "exhaustive"]
"""The four research tiers. Matches ``run_v2.py`` argparse choices and
``ResearchConfig.for_tier`` keys."""


class EvidenceStats(TypedDict):
    """Typed shape for evidence-ledger statistics.

    Returned by ``research.package.assembly.compute_evidence_stats()``.
    Replaces an anonymous ``dict[str, Any]`` with a statically-checkable
    contract so downstream consumers can rely on specific keys.
    """

    total_items: int
    """Number of evidence items in the ledger."""

    unique_domains: list[str]
    """Sorted list of unique hostnames extracted from evidence URLs."""

    providers: list[str]
    """Sorted list of search providers that contributed evidence."""

    items_with_doi: int
    """Count of evidence items that have a DOI."""

    items_with_arxiv_id: int
    """Count of evidence items that have an arXiv ID."""

    items_with_url: int
    """Count of evidence items that have a URL (canonical or raw)."""

    iterations_represented: list[int]
    """Sorted list of iteration indices that contributed evidence."""


class ProviderResolution(StrictBase):
    """Resolved state for a configured provider during tool-surface setup."""

    provider: str
    instantiated: bool = False
    available: bool = False
    reason: str | None = None


class ToolResolution(StrictBase):
    """Resolved state for a tool exposed to the subagent surface."""

    tool: str
    enabled: bool = False
    reason: str | None = None


class ToolProviderManifest(StrictBase):
    """Durable record of provider/tool setup for a run."""

    configured_providers: list[str] = []
    instantiated_providers: list[str] = []
    active_providers: list[str] = []
    available_tools: list[str] = []
    provider_resolutions: list[ProviderResolution] = []
    tool_resolutions: list[ToolResolution] = []
    degradation_reasons: list[str] = []


class SubagentToolSpec(StrictBase):
    """Serialisable spec for rebuilding a subagent's tool surface.

    Passed to ``run_subagent`` as a replay-stable alternative to
    ``list[Callable]``. Closures are not fingerprint-stable across
    processes; this spec is.
    """

    enabled_providers: list[str] = []
    sandbox_enabled: bool = False
    sandbox_backend: str | None = None


class ToolSurfaceResolution(StrictBase):
    """Serialisable output of the tool-surface resolution checkpoint.

    Packaging spec + manifest as a single return value lets the result
    flow through a Kitaru checkpoint without a tuple unwrap boilerplate.
    """

    spec: SubagentToolSpec | None = None
    manifest: ToolProviderManifest = ToolProviderManifest()


class RunMetadata(StrictBase):
    """Metadata for a single research run.

    Captures identifiers, timing, cost, and stop reason.
    """

    run_id: str
    """Unique identifier for this run."""

    tier: Tier
    """Research tier."""

    started_at: str
    """ISO-8601 timestamp when the run started."""

    completed_at: str | None = None
    """ISO-8601 timestamp when the run completed, if finished."""

    total_cost_usd: Annotated[float, Field(ge=0.0)] = 0.0
    """Total cost in USD for the entire run."""

    total_iterations: Annotated[int, Field(ge=0)] = 0
    """Number of research iterations completed."""

    stop_reason: str | None = None
    """Why the run terminated (e.g. 'converged', 'budget_exhausted')."""

    grounding_density: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    """Fraction of substantive sentences with valid citations (0.0–1.0).

    Computed during assembly. None when no report was produced or the
    ledger was empty (grounding check skipped).
    """

    export_path: str | None = None
    """Filesystem path where the durable package was exported, if any."""


class InvestigationPackage(StrictBase):
    """Complete output of a single research investigation.

    Preserves all intermediate artifacts for auditability:
    brief, plan, evidence ledger, iteration records, draft,
    critique, final report, and prompt hashes.
    """

    schema_version: str = "1.0"
    """Schema version for forward compatibility."""

    metadata: RunMetadata
    """Run-level metadata (timing, cost, stop reason)."""

    brief: ResearchBrief
    """The normalized research brief."""

    plan: ResearchPlan
    """The originally approved investigation plan."""

    revised_plan: ResearchPlan | None = None
    """Supplemental-loop revision of the plan, when one was produced."""

    ledger: EvidenceLedger
    """All evidence collected during the run."""

    iterations: list[IterationRecord] = []
    """Per-iteration records with decisions and metrics."""

    draft: DraftReport | None = None
    """Draft report, if produced."""

    critique: CritiqueReport | None = None
    """Critique report, if produced."""

    final_report: FinalReport | None = None
    """Final polished report, if produced."""

    verification: VerificationReport | None = None
    """Optional post-finalize verification report; None when disabled."""

    prompt_hashes: dict[str, str] = {}
    """Mapping of role -> sha256 hash of the prompt used."""

    tool_provider_manifest: ToolProviderManifest = ToolProviderManifest()
    """Durable record of resolved providers, tools, and degradation reasons."""


class CouncilComparison(StrictBase):
    """Comparison of outputs across multiple generators in council mode."""

    comparison: str
    """Textual comparison of generator outputs."""

    generator_scores: dict[str, float] = {}
    """Scores per generator (generator_name -> score)."""

    recommended_generator: str | None = None
    """Which generator produced the best output, if determined."""


class CouncilPackage(StrictBase):
    """Output of a council-mode run with multiple generators.

    Structurally distinct from InvestigationPackage because
    council is a different product mode — it wraps multiple
    investigation packages with cross-generator comparison.
    """

    schema_version: str = "1.0"
    """Schema version for forward compatibility."""

    canonical_generator: str | None = None
    """The generator selected as canonical, if any."""

    council_provider_compromise: bool = False
    """Whether a provider compromise was made during council."""

    comparison: CouncilComparison | None = None
    """Cross-generator comparison, if performed."""

    packages: dict[str, InvestigationPackage] = {}
    """Per-generator investigation packages (generator_name -> package)."""
