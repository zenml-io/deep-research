"""Deterministic ledger projection with windowing and pinning.

Produces a compressed view of the evidence ledger for the supervisor's
context window.  The projection is a pure function — fully replayable:
same inputs always produce identical output.

Windowing rules (from V2 spec):
- Items added in the last ``window_iterations`` iterations: shown in FULL
- Older items: shown in COMPACT form (title + canonical ID + truncated synthesis)
- Pinned items (from ``SupervisorDecision.pinned_evidence_ids``): always FULL

Rationale: transformer LMs exhibit U-shaped attention over long contexts
(Liu et al., 2023).  Age-based compaction with explicit supervisor-owned
pinning keeps the view bounded and controllable.
"""

from __future__ import annotations

from dataclasses import dataclass

from research.contracts.evidence import EvidenceLedger
from research.ledger.canonical import extract_canonical_id

# Compact synthesis excerpts are truncated to this many characters.
_COMPACT_SYNTHESIS_LIMIT = 100


@dataclass(frozen=True)
class ProjectedItem:
    """A single item in the ledger projection.

    Attributes:
        evidence_id: Unique identifier for this evidence item.
        title: Title of the source or finding.
        source_type: Type of source (e.g. 'journal', 'preprint', 'blog').
        canonical_id: Canonical identifier string (DOI, arXiv ID, or URL).
        synthesis: Full or truncated synthesis text.
        is_compact: True if this item was compacted (older than window).
    """

    evidence_id: str
    title: str
    source_type: str | None
    canonical_id: str | None
    synthesis: str
    is_compact: bool


def project_ledger(
    ledger: EvidenceLedger,
    iteration_index: int,
    pinned_ids: list[str] | None = None,
    window_iterations: int = 3,
) -> list[ProjectedItem]:
    """Project the ledger into a windowed view for the supervisor.

    Pure function of (ledger state, iteration index, pinned IDs, window size).
    Deterministic: same inputs always produce identical output.

    Args:
        ledger: The evidence ledger to project.
        iteration_index: The current iteration index (0-based).
        pinned_ids: Evidence IDs to always show in full, regardless of age.
        window_iterations: Number of recent iterations to show in full.
            Items added in iterations where
            ``iteration_index - item.iteration_added < window_iterations``
            are shown in full.  Default is 3.

    Returns:
        List of ``ProjectedItem`` in the same order as ledger items.
    """
    pinned: set[str] = set(pinned_ids) if pinned_ids else set()
    result: list[ProjectedItem] = []

    for item in ledger.items:
        # Resolve canonical ID for display
        _id_type, canonical_id_str = extract_canonical_id(
            doi=item.doi,
            arxiv_id=item.arxiv_id,
            url=item.canonical_url or item.url,
        )
        canonical_id = canonical_id_str if canonical_id_str else None

        # Determine if item is within the recency window
        age = iteration_index - item.iteration_added
        in_window = age < window_iterations
        is_pinned = item.evidence_id in pinned

        if in_window or is_pinned:
            # Full view
            result.append(
                ProjectedItem(
                    evidence_id=item.evidence_id,
                    title=item.title,
                    source_type=item.source_type,
                    canonical_id=canonical_id,
                    synthesis=item.synthesis,
                    is_compact=False,
                )
            )
        else:
            # Compact view: truncate synthesis
            truncated = item.synthesis[:_COMPACT_SYNTHESIS_LIMIT]
            if len(item.synthesis) > _COMPACT_SYNTHESIS_LIMIT:
                truncated += "..."

            result.append(
                ProjectedItem(
                    evidence_id=item.evidence_id,
                    title=item.title,
                    source_type=item.source_type,
                    canonical_id=canonical_id,
                    synthesis=truncated,
                    is_compact=True,
                )
            )

    return result


def format_projection(items: list[ProjectedItem]) -> str:
    """Format projected items into text for the supervisor's context window.

    Full items show title, source type, canonical ID, and full synthesis.
    Compact items show title, canonical ID, and truncated synthesis.
    Sections are separated by ``---``.

    Args:
        items: List of projected items to format.

    Returns:
        Formatted text string.  Empty string for empty input.
    """
    if not items:
        return ""

    sections: list[str] = []

    for item in items:
        lines: list[str] = []

        if item.is_compact:
            lines.append(f"[COMPACT] {item.title}")
            if item.canonical_id:
                lines.append(f"  ID: {item.canonical_id}")
            lines.append(f"  {item.synthesis}")
        else:
            lines.append(f"[FULL] {item.title}")
            if item.source_type:
                lines.append(f"  Source: {item.source_type}")
            if item.canonical_id:
                lines.append(f"  ID: {item.canonical_id}")
            lines.append(f"  {item.synthesis}")

        sections.append("\n".join(lines))

    return "\n---\n".join(sections)
