"""Research brief — typed normalization of a user's research request."""

from __future__ import annotations

from research.contracts.base import StrictBase


class ResearchBrief(StrictBase):
    """Normalized representation of what the user wants investigated.

    Captures the original request verbatim alongside structured fields
    that downstream agents use to scope the investigation.
    """

    topic: str
    """Primary subject of the investigation."""

    audience: str | None = None
    """Intended audience (e.g. 'ML practitioners', 'executives')."""

    scope: str | None = None
    """Explicit scope constraint (e.g. 'last 2 years', 'only open-source')."""

    freshness_constraint: str | None = None
    """Temporal freshness requirement (e.g. '2024 onwards')."""

    recency_days: int | None = None
    """Optional normalized freshness window in days for downstream retrieval."""

    source_preferences: list[str] = []
    """Preferred source types (e.g. ['peer-reviewed', 'arxiv'])."""

    raw_request: str
    """Original user input, preserved verbatim."""
