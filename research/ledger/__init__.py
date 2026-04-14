"""Evidence ledger for research v2.

Re-exports canonical ID resolution, URL canonicalization,
dedup utilities, the managed ledger, and ledger projection.
"""

from research.ledger.canonical import extract_canonical_id
from research.ledger.dedup import compute_dedup_key, is_duplicate
from research.ledger.ledger import ManagedLedger
from research.ledger.projection import (
    ProjectedItem,
    format_projection,
    project_ledger,
)
from research.ledger.url import canonicalize_url, strip_tracking_params

__all__ = [
    "ManagedLedger",
    "ProjectedItem",
    "canonicalize_url",
    "compute_dedup_key",
    "extract_canonical_id",
    "format_projection",
    "is_duplicate",
    "project_ledger",
    "strip_tracking_params",
]
