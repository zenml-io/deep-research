from collections.abc import Mapping
from typing import Any

from deep_research.models import EvidenceCandidate, EvidenceSnippet


def normalize_tool_results(
    raw_results: list[Mapping[str, Any]], provider: str, source_kind: str
) -> list[EvidenceCandidate]:
    normalized: list[EvidenceCandidate] = []

    for idx, item in enumerate(raw_results):
        url = str(item.get("url") or "").strip()
        if not url:
            continue

        snippet_text = str(item.get("snippet") or item.get("description") or "").strip()
        source_locator = item.get("source_locator")
        snippets = []
        if snippet_text:
            snippets.append(
                EvidenceSnippet(
                    text=snippet_text,
                    source_locator=str(source_locator).strip() or None
                    if source_locator is not None
                    else None,
                )
            )

        normalized.append(
            EvidenceCandidate(
                key=f"{provider}-{idx}",
                title=str(item.get("title") or url or f"result-{idx}").strip(),
                url=url,
                snippets=snippets,
                provider=provider,
                source_kind=source_kind,
            )
        )

    return normalized
