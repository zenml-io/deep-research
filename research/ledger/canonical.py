"""Canonical ID resolution for evidence deduplication.

Produces a (id_type, canonical_id) tuple from available identifiers.
Precedence: DOI -> arXiv ID -> canonical URL.

Canonical IDs are produced at subagent synthesis time, not during ledger merge.
V2 intentionally biases dedupe toward over-inclusion rather than incorrect merging.
"""

from __future__ import annotations

import re

from research.ledger.url import canonicalize_url

# arXiv version suffix pattern: "v1", "v2", etc.
_ARXIV_VERSION_RE = re.compile(r"v\d+$")

# arXiv URL prefixes to strip
_ARXIV_URL_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
)

# DOI URL prefixes to strip
_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


def _normalize_doi(raw: str) -> str:
    """Normalize a DOI: strip URL prefix, lowercase."""
    stripped = raw.strip()
    for prefix in _DOI_URL_PREFIXES:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix) :]
            break
    return stripped.lower()


def _normalize_arxiv_id(raw: str) -> str:
    """Normalize an arXiv ID: strip URL prefix and version suffix."""
    stripped = raw.strip()
    for prefix in _ARXIV_URL_PREFIXES:
        if stripped.lower().startswith(prefix.lower()):
            stripped = stripped[len(prefix) :]
            break
    # Strip trailing version suffix (e.g., "v1", "v2")
    stripped = _ARXIV_VERSION_RE.sub("", stripped)
    return stripped


def extract_canonical_id(
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
) -> tuple[str, str]:
    """Extract a canonical identifier from available metadata.

    Returns (id_type, canonical_id) with precedence: DOI -> arXiv ID -> canonical URL.
    If no stable identifier exists, returns ("url", raw_url) or ("none", "").
    """
    # DOI takes highest precedence
    if doi and doi.strip():
        return ("doi", _normalize_doi(doi))

    # arXiv ID is next
    if arxiv_id and arxiv_id.strip():
        return ("arxiv", _normalize_arxiv_id(arxiv_id))

    # Fall back to canonical URL
    if url and url.strip():
        canonical = canonicalize_url(url)
        if canonical:
            return ("url", canonical)
        # If canonicalization fails, return the raw URL
        return ("url", url.strip())

    return ("none", "")
