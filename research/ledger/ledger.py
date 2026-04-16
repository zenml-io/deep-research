"""Managed evidence ledger — append-only with dedup.

Wraps the ``EvidenceLedger`` data contract and adds operational
behaviour: deduplication on append, convenience merge from
``SubagentFindings``, and lookup by ID.

This is NOT a scoring engine and NOT a memory system.
"""

from __future__ import annotations

import hashlib
import json
import re

from research.contracts.decisions import SubagentFindings
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.ledger.canonical import parse_source_reference
from research.ledger.dedup import compute_dedup_key, is_duplicate
from research.ledger.url import canonicalize_url

# Pattern for excerpt source prefix, e.g. [arxiv:2305.18290] or [doi:10.1234/foo]
_EXCERPT_SOURCE_RE = re.compile(r"^\[([^\]]+)\]")


def _normalize_text(value: str | None) -> str:
    """Return a stable normalized representation for hashing."""
    return " ".join((value or "").split())


def _stable_evidence_id(
    *,
    iteration: int,
    index_in_findings: int,
    ordinal_in_run: int,
    title: str,
    synthesis: str,
    doi: str | None,
    arxiv_id: str | None,
    canonical_url: str | None,
    url: str | None,
    excerpts: list[str],
    confidence_notes: str | None,
) -> str:
    """Derive a replay-safe evidence ID from stable content and provenance."""
    payload = {
        "iteration": iteration,
        "index_in_findings": index_in_findings,
        "ordinal_in_run": ordinal_in_run,
        "title": _normalize_text(title),
        "synthesis": _normalize_text(synthesis),
        "doi": _normalize_text(doi),
        "arxiv_id": _normalize_text(arxiv_id),
        "canonical_url": _normalize_text(canonical_url),
        "url": _normalize_text(url),
        "excerpts": [_normalize_text(excerpt) for excerpt in excerpts],
        "confidence_notes": _normalize_text(confidence_notes),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"ev_{digest[:12]}"


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

        for excerpt in findings.excerpts:
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
        base_ordinal = self.size
        for idx, finding in enumerate(findings.findings):
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

            # Collect only excerpts matched to this finding by source prefix
            item_excerpts = excerpt_buckets.get(idx, [])

            # Prefer source-derived title over naive finding truncation
            ref_title: str | None = None
            if idx < len(parsed_refs):
                ref_title = parsed_refs[idx].title

            title = ref_title or finding[:120]
            evidence_id = _stable_evidence_id(
                iteration=iteration,
                index_in_findings=idx,
                ordinal_in_run=base_ordinal + len(items),
                title=title,
                synthesis=finding,
                doi=doi,
                arxiv_id=arxiv_id,
                canonical_url=canonical_url_val,
                url=url,
                excerpts=item_excerpts,
                confidence_notes=findings.confidence_notes,
            )

            item = EvidenceItem(
                evidence_id=evidence_id,
                title=title,
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
