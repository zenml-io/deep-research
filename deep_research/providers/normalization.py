from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from deep_research.evidence.url import canonicalize_url
from deep_research.models import EvidenceCandidate, EvidenceSnippet, RawToolResult


def _clean_string(value: object) -> str | None:
    """Normalize a scalar raw-result field into a trimmed string or `None`.

    Container values are rejected here because downstream identity and metadata fields
    expect leaf scalar values, not nested mappings or lists with ambiguous meaning.
    """
    if value is None:
        return None
    if isinstance(value, Mapping) or (
        isinstance(value, Iterable) and not isinstance(value, str)
    ):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _as_score(value: object) -> float:
    """Coerce an optional raw score field into a float, defaulting missing values to zero.

    Normalized evidence candidates treat absent scores as `0.0` so providers can omit
    any subset of ranking signals without breaking candidate validation.
    """
    if value is None:
        return 0.0
    return float(value)


def _matched_subtopics(value: object) -> list[str]:
    """Normalize raw matched-subtopic data into a list of non-empty strings.

    Providers may emit a single string, a list-like container, or unusable values.
    This helper preserves only trimmed, non-empty subtopic labels in input order.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, Iterable):
        return []
    normalized: list[str] = []
    for subtopic in value:
        if subtopic is None:
            continue
        cleaned = str(subtopic).strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _candidate_identity(item: Mapping[str, Any], canonical_url: str) -> str:
    """Choose the strongest stable identity available for a raw result item.

    DOI wins over arXiv ID, and both win over canonical URL because those identifiers
    survive URL variants and provider-specific formatting differences more reliably.
    """
    doi = _clean_string(item.get("doi"))
    if doi is not None:
        return f"doi:{doi.lower()}"

    arxiv_id = _clean_string(item.get("arxiv_id"))
    if arxiv_id is not None:
        return f"arxiv:{arxiv_id.lower()}"

    return f"url:{canonical_url}"


def _candidate_key(item: Mapping[str, Any], canonical_url: str) -> str:
    """Generate a deterministic short key from the chosen semantic identity string.

    Hashing keeps keys compact while still remaining stable across providers that emit
    the same DOI, arXiv ID, or canonical URL for an evidence candidate.
    """
    identity = _candidate_identity(item, canonical_url).encode("utf-8")
    return f"evidence-{sha256(identity).hexdigest()[:16]}"


def _raw_metadata(item: Mapping[str, Any]) -> dict[str, object]:
    """Copy provider-specific fields that are not promoted into first-class candidate data.

    Core fields used for typed model attributes are excluded so `raw_metadata` remains a
    clean bucket for auxiliary provider payload details rather than duplicated state.
    """
    excluded = {
        "title",
        "url",
        "snippet",
        "description",
        "quality_score",
        "relevance_score",
        "authority_score",
        "freshness_score",
        "matched_subtopics",
        "doi",
        "arxiv_id",
    }
    return {
        str(key): value
        for key, value in item.items()
        if key not in excluded and value is not None
    }


def _iter_result_items(
    raw_results: list[Mapping[str, Any] | RawToolResult],
) -> list[Mapping[str, Any]]:
    """Flatten mixed raw-result payloads into a uniform list of item mappings.

    `RawToolResult` payloads may wrap items under `results` or `items`, while tests and
    other callers can pass mappings directly. This helper normalizes those entry shapes.
    """
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

        if isinstance(raw_result, Mapping):
            items.append(raw_result)

    return items


def normalize_tool_results(
    raw_results: list[Mapping[str, Any] | RawToolResult],
    provider: str,
    source_kind: str,
) -> list[EvidenceCandidate]:
    """Transform raw tool results into typed EvidenceCandidate instances."""
    normalized: list[EvidenceCandidate] = []

    for idx, item in enumerate(_iter_result_items(raw_results)):
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        canonical_url = canonicalize_url(url)
        if canonical_url is None:
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

        try:
            candidate = EvidenceCandidate(
                key=_candidate_key(item, canonical_url),
                title=str(item.get("title") or url or f"result-{idx}").strip(),
                url=url,
                snippets=snippets,
                provider=provider,
                source_kind=source_kind,
                quality_score=_as_score(item.get("quality_score")),
                relevance_score=_as_score(item.get("relevance_score")),
                authority_score=_as_score(item.get("authority_score")),
                freshness_score=_as_score(item.get("freshness_score")),
                matched_subtopics=_matched_subtopics(item.get("matched_subtopics")),
                doi=_clean_string(item.get("doi")),
                arxiv_id=_clean_string(item.get("arxiv_id")),
                raw_metadata=_raw_metadata(item),
            )
        except (ValidationError, ValueError, TypeError):
            continue
        normalized.append(candidate)

    return normalized
