"""Evidence ledger for research v2.

Re-exports canonical ID resolution, URL canonicalization,
dedup utilities, and the managed ledger.
"""

from research.ledger.canonical import extract_canonical_id
from research.ledger.dedup import compute_dedup_key, is_duplicate
from research.ledger.ledger import ManagedLedger
from research.ledger.url import canonicalize_url, strip_tracking_params

__all__ = [
    "ManagedLedger",
    "canonicalize_url",
    "compute_dedup_key",
    "extract_canonical_id",
    "is_duplicate",
    "strip_tracking_params",
]
