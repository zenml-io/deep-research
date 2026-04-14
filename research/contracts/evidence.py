"""Evidence contracts — items and ledger.

The ledger is an append-only deduplicated index of distilled findings.
It is NOT a memory system and NOT a scoring engine. Merge/dedup logic
lives in the ``ledger/`` module — these are pure data contracts.
"""

from __future__ import annotations

from research.contracts.base import StrictBase


class EvidenceItem(StrictBase):
    """A single piece of distilled evidence.

    Stores the synthesized value produced by a subagent, plus
    optional verbatim excerpts and canonical identifiers for dedup.
    """

    evidence_id: str
    """Unique identifier for this evidence item."""

    title: str
    """Title of the source or finding."""

    url: str | None = None
    """Web URL of the source, if available."""

    doi: str | None = None
    """Digital Object Identifier, if available."""

    arxiv_id: str | None = None
    """arXiv paper ID (e.g. '2301.12345'), if available."""

    canonical_url: str | None = None
    """Canonical URL after redirect resolution."""

    source_type: str | None = None
    """Type of source (e.g. 'journal', 'preprint', 'blog')."""

    synthesis: str
    """Subagent's distilled summary of this evidence."""

    excerpts: list[str] = []
    """Optional verbatim excerpts from the source."""

    confidence_notes: str | None = None
    """Notes on reliability or confidence in this evidence."""

    iteration_added: int
    """Which research iteration discovered this evidence."""

    provider: str | None = None
    """Search provider that surfaced this source (e.g. 'arxiv', 'brave')."""


class EvidenceLedger(StrictBase):
    """Append-only index of all evidence collected during a run.

    This is a data contract only. Merge and dedup logic will live
    in the ``ledger/`` module.
    """

    items: list[EvidenceItem] = []
    """Collected evidence items."""

    schema_version: str = "1.0"
    """Schema version for forward compatibility."""
