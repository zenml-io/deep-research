"""Report contracts — draft, critique, and final report."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field

from research.contracts.base import StrictBase

# Matches markdown headings: "## Heading" or "### Sub-heading" etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

CritiqueDimension = Literal["source_reliability", "completeness", "grounding"]
"""The three dimensions every critique must score. Pinned by
``research/prompts/reviewer.md`` and asserted by the critique merger."""


def _extract_sections(markdown: str) -> list[str]:
    """Extract section headings from a markdown string.

    Returns heading text in document order, stripped of leading ``#`` markers.
    """
    return [m.group(2).strip() for m in _HEADING_RE.finditer(markdown)]


class DraftReport(StrictBase):
    """Initial report draft with inline evidence citations.

    Content is markdown with inline ``[evidence_id]`` citations
    linking claims to evidence in the ledger.
    """

    content: str
    """Markdown report body with inline [evidence_id] citations."""

    sections: list[str] = []
    """Section headings present in the report."""

    @classmethod
    def from_markdown(cls, markdown: str) -> DraftReport:
        """Construct a DraftReport from raw markdown text.

        Extracts section headings automatically from ``#`` markers.
        """
        return cls(content=markdown, sections=_extract_sections(markdown))


class CritiqueDimensionScore(StrictBase):
    """Score for a single critique dimension.

    The three dimensions are ``source_reliability``, ``completeness``,
    and ``grounding``. Scores are in the 0.0-1.0 range per
    ``research/prompts/reviewer.md``.
    """

    dimension: CritiqueDimension
    """Name of the dimension."""

    score: Annotated[float, Field(ge=0.0, le=1.0)]
    """Numeric score for this dimension (0.0-1.0)."""

    explanation: str
    """Explanation of the score."""


class ReviewerDisagreement(StrictBase):
    """Per-dimension disagreement profile for dual-reviewer critique merges."""

    dimension: str
    """Shared critique dimension name."""

    reviewer_1_score: float
    """Score assigned by reviewer 1."""

    reviewer_2_score: float
    """Score assigned by reviewer 2."""

    delta: float
    """Absolute difference between the two reviewer scores."""


class CritiqueReport(StrictBase):
    """Structured critique of a draft report.

    Evaluates the draft across multiple dimensions and indicates
    whether further research is needed. On deep tier, tracks
    per-reviewer provenance for multi-reviewer merging.
    """

    dimensions: list[CritiqueDimensionScore]
    """Scores across critique dimensions (must include source_reliability, completeness, grounding)."""

    require_more_research: bool
    """Whether the critique recommends additional research iterations."""

    issues: list[str] = []
    """Specific issues identified in the draft."""

    reviewer_provenance: list[str] = []
    """Per-reviewer tracking for deep tier multi-reviewer merging."""

    reviewer_disagreements: list[ReviewerDisagreement] = []
    """Per-dimension disagreement profile for shared dual-reviewer dimensions."""


class FinalReport(StrictBase):
    """Final polished report after critique and revision.

    Content is markdown with inline ``[evidence_id]`` citations,
    inheriting citation discipline from :class:`DraftReport`.
    """

    content: str
    """Markdown report body with inline [evidence_id] citations."""

    sections: list[str] = []
    """Section headings present in the report."""

    stop_reason: str | None = None
    """Why the research loop terminated (e.g. 'converged', 'budget_exhausted')."""

    @classmethod
    def from_markdown(
        cls, markdown: str, *, stop_reason: str | None = None
    ) -> FinalReport:
        """Construct a FinalReport from raw markdown text.

        Extracts section headings automatically from ``#`` markers.
        """
        return cls(
            content=markdown,
            sections=_extract_sections(markdown),
            stop_reason=stop_reason,
        )


class VerificationIssue(StrictBase):
    """A single claim flagged during post-finalize verification."""

    claim_excerpt: str
    """Short excerpt of the claim being verified."""

    evidence_ids: list[str] = []
    """Citations the claim references, if any."""

    status: str = "unsupported"
    """One of: 'unsupported', 'partial', 'contradicted'. String for forward-compat."""

    reason: str | None = None
    """Why the verifier flagged the claim."""

    suggested_fix: str | None = None
    """Optional concrete edit suggestion; informational only in this PR."""


class VerificationReport(StrictBase):
    """Structured verification output over a report against the ledger."""

    issues: list[VerificationIssue] = []
    verified_claim_count: int = 0
    unsupported_claim_count: int = 0
    needs_revision: bool = False
    """Reviewer-style advisory flag; NOT consumed by the flow in PR 2."""
