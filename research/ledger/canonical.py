"""Canonical ID resolution for evidence deduplication.

Produces a (id_type, canonical_id) tuple from available identifiers.
Precedence: DOI -> arXiv ID -> canonical URL.

Also provides ``parse_source_reference`` to extract structured identifiers
from the pipe-separated format subagents produce in source_references.

Biases toward over-inclusion rather than incorrect merging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


# Patterns for extracting structured identifiers from pipe-separated refs.
_DOI_PATTERN = re.compile(r"(?:^|\||\s)doi:\s*(\S+)", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"(?:^|\||\s)arxiv:\s*(\S+)", re.IGNORECASE)
_URL_PATTERN = re.compile(r"(https?://\S+)")


@dataclass(frozen=True, slots=True)
class ParsedReference:
    """Structured identifiers extracted from a subagent source reference string."""

    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    title: str | None = None


def parse_source_reference(ref: str) -> ParsedReference:
    """Extract DOI, arXiv ID, and URL from a pipe-separated source reference.

    The subagent prompt produces references like::

        Rafailov et al. (2023) DPO | arxiv:2305.18290 | https://arxiv.org/abs/2305.18290

    This function extracts ``doi:``, ``arxiv:``, and any ``https?://`` URL
    from such strings.  Unrecognized segments are silently ignored.

    Returns a ``ParsedReference`` with whichever fields were found (others ``None``).
    """
    if not ref or not ref.strip():
        return ParsedReference()

    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    title: str | None = None

    # Extract human-readable title from the first pipe-segment
    first_segment = ref.split("|")[0].strip()
    if first_segment and not first_segment.startswith("http"):
        title = first_segment

    doi_match = _DOI_PATTERN.search(ref)
    if doi_match:
        doi = doi_match.group(1).strip().rstrip("|").strip()

    arxiv_match = _ARXIV_PATTERN.search(ref)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).strip().rstrip("|").strip()

    url_match = _URL_PATTERN.search(ref)
    if url_match:
        url = url_match.group(1).strip().rstrip("|").strip()

    return ParsedReference(doi=doi, arxiv_id=arxiv_id, url=url, title=title)
