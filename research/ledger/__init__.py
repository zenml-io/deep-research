"""Evidence ledger for research v2.

Re-exports canonical ID resolution and URL canonicalization utilities.
"""

from research.ledger.canonical import extract_canonical_id
from research.ledger.url import canonicalize_url, strip_tracking_params

__all__ = [
    "canonicalize_url",
    "extract_canonical_id",
    "strip_tracking_params",
]
