"""Investigation and council package contracts.

These are the top-level output types — durable, versioned bundles
that capture everything produced during a research run.
"""

from __future__ import annotations

from typing import TypedDict

from research.contracts.base import StrictBase
from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.plan import ResearchPlan
from research.contracts.reports import CritiqueReport, DraftReport, FinalReport


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


class RunMetadata(StrictBase):
    """Metadata for a single research run.

    Captures identifiers, timing, cost, and stop reason.
    """

    run_id: str
    """Unique identifier for this run."""

    tier: str
    """Research tier (e.g. 'quick', 'standard', 'deep')."""

    started_at: str
    """ISO-8601 timestamp when the run started."""

    completed_at: str | None = None
    """ISO-8601 timestamp when the run completed, if finished."""

    total_cost_usd: float = 0.0
    """Total cost in USD for the entire run."""

    total_iterations: int = 0
    """Number of research iterations completed."""

    stop_reason: str | None = None
    """Why the run terminated (e.g. 'converged', 'budget_exhausted')."""

    grounding_density: float | None = None
    """Fraction of substantive sentences with valid citations (0.0–1.0).

    Computed during assembly. None when no report was produced or the
    ledger was empty (grounding check skipped).
    """


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
    """The investigation plan."""

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

    prompt_hashes: dict[str, str] = {}
    """Mapping of role -> sha256 hash of the prompt used."""


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
