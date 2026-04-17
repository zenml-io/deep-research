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
