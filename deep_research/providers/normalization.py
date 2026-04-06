from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from deep_research.models import EvidenceCandidate, EvidenceSnippet, RawToolResult


def _iter_result_items(
    raw_results: list[Mapping[str, Any] | RawToolResult],
) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []

    for raw_result in raw_results:
        if isinstance(raw_result, RawToolResult):
            payload = raw_result.payload
            nested = payload.get("results") or payload.get("items")
            if isinstance(nested, list):
                items.extend(item for item in nested if isinstance(item, Mapping))
                continue
            items.append(payload)
            continue

        items.append(raw_result)

    return items


def _candidate_key(provider: str, url: str) -> str:
    identity = f"{provider}:{url}".encode("utf-8")
    return f"{provider}-{sha256(identity).hexdigest()[:16]}"


def normalize_tool_results(
    raw_results: list[Mapping[str, Any] | RawToolResult],
    provider: str,
    source_kind: str,
) -> list[EvidenceCandidate]:
    normalized: list[EvidenceCandidate] = []

    for idx, item in enumerate(_iter_result_items(raw_results)):
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
                key=_candidate_key(provider, url),
                title=str(item.get("title") or url or f"result-{idx}").strip(),
                url=url,
                snippets=snippets,
                provider=provider,
                source_kind=source_kind,
            )
        )

    return normalized
