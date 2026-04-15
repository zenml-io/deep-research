"""Managed evidence ledger — append-only with dedup.

Wraps the ``EvidenceLedger`` data contract and adds operational
behaviour: deduplication on append, convenience merge from
``SubagentFindings``, and lookup by ID.

This is NOT a scoring engine and NOT a memory system.
"""

from __future__ import annotations

import re
import uuid

from research.contracts.decisions import SubagentFindings
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.ledger.canonical import parse_source_reference
from research.ledger.dedup import compute_dedup_key, is_duplicate
from research.ledger.url import canonicalize_url

# Pattern for excerpt source prefix, e.g. [arxiv:2305.18290] or [doi:10.1234/foo]
_EXCERPT_SOURCE_RE = re.compile(r"^\[([^\]]+)\]")


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

        Parses structured identifiers (DOI, arXiv ID, URL) from the
        pipe-separated ``source_references`` so that dedup keys are
        populated on first merge, not just on re-encounter.

        Excerpts are matched to findings by source prefix (e.g.
        ``[arxiv:2305.18290]``). Unmatched excerpts are dropped to
        avoid noisy duplication across evidence items.

        Returns the list of actually-added items.
        """
        # Pre-parse source references for provenance fields
        parsed_refs = [
            parse_source_reference(ref) for ref in findings.source_references
        ]

        # Bucket excerpts by source prefix → finding index
        excerpt_buckets: dict[int, list[str]] = {}
        unmatched_excerpts: list[str] = []

        for excerpt in findings.excerpts or []:
            match = _EXCERPT_SOURCE_RE.match(excerpt)
            matched = False
            if match:
                prefix = match.group(1).lower()
                # Try to match prefix against parsed refs
                for ref_idx, pref in enumerate(parsed_refs):
                    if pref.arxiv_id and pref.arxiv_id.lower() in prefix:
                        excerpt_buckets.setdefault(ref_idx, []).append(excerpt)
                        matched = True
                        break
                    if pref.doi and pref.doi.lower() in prefix:
                        excerpt_buckets.setdefault(ref_idx, []).append(excerpt)
                        matched = True
                        break
            if not matched:
                unmatched_excerpts.append(excerpt)

        items: list[EvidenceItem] = []
        for idx, finding in enumerate(findings.findings):
            evidence_id = f"ev_{uuid.uuid4().hex[:12]}"

            # Extract provenance from paired source reference
            doi: str | None = None
            arxiv_id: str | None = None
            canonical_url_val: str | None = None
            url: str | None = None

            if idx < len(parsed_refs):
                pref = parsed_refs[idx]
                doi = pref.doi
                arxiv_id = pref.arxiv_id
                if pref.url:
                    url = pref.url
                    canonical_url_val = canonicalize_url(pref.url) or None
            elif idx < len(findings.source_references):
                # Fallback: raw reference string as URL if it looks like one
                raw_ref = findings.source_references[idx]
                if raw_ref and raw_ref.strip().startswith("http"):
                    url = raw_ref.strip()
                    canonical_url_val = canonicalize_url(url) or None

            # Collect only excerpts matched to this finding by source prefix
            item_excerpts = excerpt_buckets.get(idx, [])

            # Prefer source-derived title over naive finding truncation
            ref_title: str | None = None
            if idx < len(parsed_refs):
                ref_title = parsed_refs[idx].title

            item = EvidenceItem(
                evidence_id=evidence_id,
                title=ref_title or finding[:120],
                url=url,
                doi=doi,
                arxiv_id=arxiv_id,
                canonical_url=canonical_url_val,
                synthesis=finding,
                excerpts=item_excerpts,
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
