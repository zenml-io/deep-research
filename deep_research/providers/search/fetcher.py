from __future__ import annotations

from html.parser import HTMLParser

from deep_research.providers.search._http import (
    DEFAULT_RETRY_POLICY,
    build_client,
    request_with_retry,
)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        cleaned = " ".join(data.split())
        if cleaned:
            self._chunks.append(cleaned)

    def text(self) -> str:
        return " ".join(self._chunks)


def fetch_url_content(url: str, timeout_sec: int, max_chars: int) -> str | None:
    if url.lower().endswith(".pdf"):
        return None

    client = build_client(timeout=timeout_sec)
    try:
        response = request_with_retry(
            client,
            "GET",
            url,
            retry_policy=DEFAULT_RETRY_POLICY,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        return None

    parser = _HTMLTextExtractor()
    parser.feed(response.text)
    text = parser.text().strip()
    if not text:
        return None
    return text[:max_chars]
