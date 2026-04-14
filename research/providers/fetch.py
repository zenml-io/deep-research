"""Async content fetcher — fetch a URL, extract visible text from HTML.

Skips PDFs.  Strips ``<script>``, ``<style>``, and ``<noscript>`` tags.
Returns up to *max_chars* characters of plain text, or ``None`` on failure.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser

import httpx

from research.providers._http import (
    DEFAULT_RETRY_POLICY,
    build_async_client,
    request_with_retry,
)

logger = logging.getLogger(__name__)

# Defaults ------------------------------------------------------------------
DEFAULT_FETCH_TIMEOUT_SEC = 15
DEFAULT_FETCH_MAX_CHARS = 50_000


# ---------------------------------------------------------------------------
# HTML text extraction
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML→text converter.

    Ignores content inside ``<script>``, ``<style>``, and ``<noscript>`` tags.
    Collapses whitespace within each text fragment.
    """

    _IGNORED_TAGS = frozenset({"script", "style", "noscript"})

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._ignored_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self._chunks.append(cleaned)

    def text(self) -> str:
        """Return extracted text as a single string."""
        return " ".join(self._chunks)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_url_content(
    url: str,
    *,
    timeout_sec: int = DEFAULT_FETCH_TIMEOUT_SEC,
    max_chars: int = DEFAULT_FETCH_MAX_CHARS,
) -> str | None:
    """Fetch *url* and return extracted plain text, or ``None`` on failure.

    * Skips URLs ending in ``.pdf``.
    * Skips responses whose ``Content-Type`` is not HTML or plain text.
    * Strips ``<script>``/``<style>``/``<noscript>`` tags from HTML.
    * Returns at most *max_chars* characters.
    """
    if url.lower().endswith(".pdf"):
        logger.debug("Skipping PDF URL: %s", url)
        return None

    client = build_async_client(timeout=timeout_sec)
    try:
        try:
            response = await request_with_retry(
                client,
                "GET",
                url,
                retry_policy=DEFAULT_RETRY_POLICY,
            )
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.debug("Fetch failed for %s: %s", url, exc)
            return None

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            logger.debug("Skipping non-text content-type %r for %s", content_type, url)
            return None

        parser = _HTMLTextExtractor()
        parser.feed(response.text)
        text = parser.text().strip()
        if not text:
            return None
        return text[:max_chars]
    finally:
        await client.aclose()
