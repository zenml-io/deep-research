from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from deep_research.enums import SourceGroup, SourceKind
from deep_research.evidence.scoring import score_candidate_quality
from deep_research.evidence.url import canonicalize_url
from deep_research.models import EvidenceCandidate, EvidenceSnippet, RawToolResult


# Fields promoted to typed EvidenceCandidate attributes; excluded from raw_metadata.
_RAW_METADATA_EXCLUDED: frozenset[str] = frozenset({
    "title", "url", "snippet", "description",
    "quality_score", "relevance_score", "authority_score", "freshness_score",
    "matched_subtopics", "doi", "arxiv_id",
})

# URL substring patterns mapped to SourceGroup, evaluated in priority order.
_URL_PATTERNS: tuple[tuple[tuple[str, ...], SourceGroup], ...] = (
    (("github.com", "gitlab.com"), SourceGroup.REPOS),
    (("reddit.com", "news.ycombinator.com", "x.com", "twitter.com"), SourceGroup.FORUMS),
    (("medium.com", "substack.com", "dev.to"), SourceGroup.BLOGS),
    (("techcrunch.com", "theverge.com", "wired.com", "bloomberg.com", ".news/"), SourceGroup.NEWS),
    (("/docs/", "docs.", "readthedocs.", "developer."), SourceGroup.DOCS),
)


def clean_string(value: object) -> str | None:
    """Normalize a scalar raw-result field into a trimmed string, or None for containers."""
    if value is None:
        return None
    if isinstance(value, Mapping) or (
        isinstance(value, Iterable) and not isinstance(value, str)
    ):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def as_score(value: object) -> float:
    """Coerce an optional raw score field to float, treating absent values as 0.0."""
    if value is None:
        return 0.0
    return float(value)


def matched_subtopics(value: object) -> list[str]:
    """Normalize raw matched-subtopic data into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif not isinstance(value, Iterable):
        return []
    return [s for subtopic in value if subtopic is not None and (s := str(subtopic).strip())]


def candidate_identity(item: Mapping[str, Any], canonical_url: str) -> str:
    """Return the strongest stable identity: DOI > arXiv ID > canonical URL."""
    doi = clean_string(item.get("doi"))
    if doi is not None:
        return f"doi:{doi.lower()}"
    arxiv_id = clean_string(item.get("arxiv_id"))
    if arxiv_id is not None:
        return f"arxiv:{arxiv_id.lower()}"
    return f"url:{canonical_url}"


def candidate_key(item: Mapping[str, Any], canonical_url: str) -> str:
    """Return a deterministic 16-hex-char key derived from the semantic identity."""
    identity = candidate_identity(item, canonical_url).encode("utf-8")
    return f"evidence-{sha256(identity).hexdigest()[:16]}"


def raw_metadata(item: Mapping[str, Any]) -> dict[str, object]:
    """Copy auxiliary provider fields not promoted to first-class candidate attributes."""
    return {
        str(key): value
        for key, value in item.items()
        if key not in _RAW_METADATA_EXCLUDED and value is not None
    }


def parse_source_kind(value: object, fallback: str) -> SourceKind:
    """Coerce a raw source-kind field into a SourceKind enum, falling back to WEB."""
    raw_value = str(value or fallback).strip().lower()
    try:
        return SourceKind(raw_value)
    except ValueError:
        return SourceKind.WEB


def infer_source_group(
    canonical_url: str,
    source_kind: SourceKind,
    item: Mapping[str, Any],
) -> SourceGroup:
    """Derive the best SourceGroup from an explicit field, URL patterns, or content heuristics."""
    raw_group = clean_string(item.get("source_group"))
    if raw_group is not None:
        try:
            return SourceGroup(raw_group)
        except ValueError:
            pass

    if source_kind == SourceKind.PAPER:
        return SourceGroup.PAPERS

    lower_url = canonical_url.casefold()
    for patterns, group in _URL_PATTERNS:
        if any(p in lower_url for p in patterns):
            return group

    # Content heuristics — only computed when URL patterns produce no match.
    title = clean_string(item.get("title")) or ""
    description = clean_string(item.get("description")) or ""
    combined = f"{title} {description}".casefold()
    if "benchmark" in combined or "leaderboard" in combined:
        return SourceGroup.BENCHMARKS

    return SourceGroup.WEB


def iter_result_items(
    raw_results: list[Mapping[str, Any] | RawToolResult],
) -> list[Mapping[str, Any]]:
    """Flatten mixed raw-result payloads into a uniform list of item mappings."""
    items: list[Mapping[str, Any]] = []
    for raw_result in raw_results:
        if isinstance(raw_result, RawToolResult):
            payload = raw_result.payload
            nested = payload.get("results") or payload.get("items")
            if isinstance(nested, list):
                items.extend(item for item in nested if isinstance(item, Mapping))
                continue
            items.append(payload)
        elif isinstance(raw_result, Mapping):
            items.append(raw_result)
    return items


def _build_candidate(
    item: Mapping[str, Any],
    idx: int,
    provider: str,
    source_kind: str,
) -> EvidenceCandidate | None:
    """Build a typed EvidenceCandidate from a raw item mapping; None on missing URL or validation error."""
    url = str(item.get("url") or "").strip()
    if not url:
        return None
    canonical_url = canonicalize_url(url)
    if canonical_url is None:
        return None

    normalized_source_kind = parse_source_kind(item.get("source_kind"), source_kind)
    source_group = infer_source_group(canonical_url, normalized_source_kind, item)

    snippet_text = str(item.get("snippet") or item.get("description") or "").strip()
    raw_locator = item.get("source_locator")
    locator = str(raw_locator).strip() or None if raw_locator is not None else None
    snippets = [EvidenceSnippet(text=snippet_text, source_locator=locator)] if snippet_text else []

    raw_quality = item.get("quality_score")
    try:
        candidate = EvidenceCandidate(
            key=candidate_key(item, canonical_url),
            title=str(item.get("title") or url or f"result-{idx}").strip(),
            url=url,
            snippets=snippets,
            provider=provider,
            source_kind=normalized_source_kind,
            source_group=source_group,
            canonical_url=canonical_url,
            quality_score=as_score(raw_quality),
            relevance_score=as_score(item.get("relevance_score")),
            authority_score=as_score(item.get("authority_score")),
            freshness_score=as_score(item.get("freshness_score")),
            matched_subtopics=matched_subtopics(item.get("matched_subtopics")),
            doi=clean_string(item.get("doi")),
            arxiv_id=clean_string(item.get("arxiv_id")),
            published_at=clean_string(
                item.get("published_at") or item.get("raw_published")
            ),
            raw_metadata=raw_metadata(item),
        )
    except (ValidationError, ValueError, TypeError):
        return None

    if raw_quality is None:
        candidate = candidate.model_copy(
            update={"quality_score": score_candidate_quality(candidate)}
        )
    return candidate


def normalize_tool_results(
    raw_results: list[Mapping[str, Any] | RawToolResult],
    provider: str,
    source_kind: str,
) -> list[EvidenceCandidate]:
    """Transform raw tool results into typed EvidenceCandidate instances."""
    candidates = []
    for idx, item in enumerate(iter_result_items(raw_results)):
        candidate = _build_candidate(item, idx, provider, source_kind)
        if candidate is not None:
            candidates.append(candidate)
    return candidates
