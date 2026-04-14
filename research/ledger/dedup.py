"""Evidence deduplication — exact-match only.

Dedup precedence: DOI -> arXiv ID -> canonical URL.
Items with no stable identifier are NEVER considered duplicates
(bias toward over-inclusion).

No fuzzy matching, no title-based dedup.
"""

from __future__ import annotations

from research.contracts.evidence import EvidenceItem
from research.ledger.canonical import extract_canonical_id


def compute_dedup_key(item: EvidenceItem) -> str | None:
    """Extract a canonical dedup key from an evidence item.

    Uses ``extract_canonical_id`` with precedence DOI -> arXiv ID -> canonical URL.
    Returns the canonical ID string, or ``None`` if no stable identifier exists.
    """
    id_type, canonical_id = extract_canonical_id(
        doi=item.doi,
        arxiv_id=item.arxiv_id,
        url=item.canonical_url or item.url,
    )
    if id_type == "none":
        return None
    return canonical_id


def is_duplicate(existing_keys: set[str], item: EvidenceItem) -> bool:
    """Check whether *item* is a duplicate of something already seen.

    Items with no stable identifier (key is ``None``) are **never**
    considered duplicates — we bias toward over-inclusion.
    """
    key = compute_dedup_key(item)
    if key is None:
        return False
    return key in existing_keys
