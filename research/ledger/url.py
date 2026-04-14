"""URL canonicalization for evidence deduplication.

Strips tracking parameters, normalizes trailing slashes, lowercases scheme
and hostname (but NOT the path), and strips fragments.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

# Tracking parameters to strip
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "ref",
        "source",
    }
)


def strip_tracking_params(url: str) -> str:
    """Remove known tracking query parameters from a URL.

    Preserves all non-tracking query parameters and their order.
    """
    if not url or not url.strip():
        return url

    parts = urlsplit(url)
    if not parts.query:
        return url

    # Parse and filter query params, preserving order
    filtered = [
        (k, v)
        for k, vs in parse_qs(parts.query, keep_blank_values=True).items()
        if k.lower() not in _TRACKING_PARAMS
        for v in vs
    ]
    new_query = urlencode(filtered)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, new_query, parts.fragment)
    )


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for identity comparison.

    - Strips tracking parameters (utm_*, fbclid, gclid, ref, source)
    - Normalizes trailing slashes (strips unless URL is just a domain root)
    - Lowercases scheme and hostname (but NOT the path)
    - Strips fragment (#...)

    Returns empty string for empty/None input or URLs without scheme/netloc.
    """
    if not url or not url.strip():
        return ""

    raw = url.strip()
    parts = urlsplit(raw)

    # Reject URLs without scheme or netloc
    if not parts.scheme or not parts.netloc:
        return ""

    # Lowercase scheme and hostname
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()

    # Reconstruct netloc with lowercased hostname but original port
    try:
        port = parts.port
    except ValueError:
        return ""

    netloc = hostname
    if port is not None:
        # Strip default ports
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            pass
        else:
            netloc = f"{hostname}:{port}"

    # Preserve path case, normalize trailing slash
    path = parts.path or "/"

    # Strip trailing slash unless path is just "/"  (domain root)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Filter tracking params from query
    filtered_params = [
        (k, v)
        for k, vs in parse_qs(parts.query, keep_blank_values=True).items()
        if k.lower() not in _TRACKING_PARAMS
        for v in vs
    ]
    query = urlencode(filtered_params)

    # Strip fragment entirely
    fragment = ""

    return urlunsplit((scheme, netloc, path, query, fragment))
