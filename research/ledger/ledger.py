"""Managed evidence ledger — append-only with dedup.

Wraps the ``EvidenceLedger`` data contract and adds operational
behaviour: deduplication on append, convenience merge from
``SubagentFindings``, and lookup by ID.

This is NOT a scoring engine and NOT a memory system.
"""

from __future__ import annotations

import uuid

from research.contracts.decisions import SubagentFindings
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.ledger.dedup import compute_dedup_key, is_duplicate


class ManagedLedger:
    """Append-only, deduplicated evidence ledger.

    Items are deduplicated by exact match on DOI, arXiv ID, or
    canonical URL (in that precedence order).  Items with no stable
    identifier are always admitted (bias toward over-inclusion).
    """

    def __init__(self) -> None:
        self._ledger = EvidenceLedger()
        self._dedup_keys: set[str] = set()
        self._by_id: dict[str, EvidenceItem] = {}

    # -- public API ----------------------------------------------------------

    def append(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Add *items* to the ledger, skipping duplicates.

        Returns the list of actually-added items (excludes duplicates).
        Append-only: items can never be removed.
        """
        added: list[EvidenceItem] = []
        for item in items:
            if is_duplicate(self._dedup_keys, item):
                continue
            key = compute_dedup_key(item)
            if key is not None:
                self._dedup_keys.add(key)
            self._ledger.items.append(item)
            self._by_id[item.evidence_id] = item
            added.append(item)
        return added

    def merge_findings(
        self,
        findings: SubagentFindings,
        iteration: int,
    ) -> list[EvidenceItem]:
        """Create ``EvidenceItem``s from *findings* and append them.

        Convenience method that turns each finding string into an
        ``EvidenceItem`` with a generated ``evidence_id``.  Returns
        the list of actually-added items.
        """
        items: list[EvidenceItem] = []
        for idx, finding in enumerate(findings.findings):
            evidence_id = f"ev_{uuid.uuid4().hex[:12]}"
            # pair source references with findings by index if available
            url = (
                findings.source_references[idx]
                if idx < len(findings.source_references)
                else None
            )
            item = EvidenceItem(
                evidence_id=evidence_id,
                title=finding[:120],  # truncate long findings for title
                url=url,
                synthesis=finding,
                excerpts=findings.excerpts if findings.excerpts else [],
                confidence_notes=findings.confidence_notes,
                iteration_added=iteration,
            )
            items.append(item)
        return self.append(items)

    def get_by_id(self, evidence_id: str) -> EvidenceItem | None:
        """Return the item with *evidence_id*, or ``None``."""
        return self._by_id.get(evidence_id)

    # -- properties ----------------------------------------------------------

    @property
    def ledger(self) -> EvidenceLedger:
        """The underlying ``EvidenceLedger`` data contract."""
        return self._ledger

    @property
    def size(self) -> int:
        """Number of items in the ledger."""
        return len(self._ledger.items)
