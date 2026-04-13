from __future__ import annotations

from collections import Counter
from math import exp
from urllib.parse import urlparse

from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import EvidenceCandidate, ResearchPlan

_QUALITY_BY_SOURCE_KIND: dict[SourceKind, float] = {
    SourceKind.PAPER: 0.9,
    SourceKind.DOCS: 0.8,
    SourceKind.REPOSITORY: 0.8,
    SourceKind.BENCHMARK: 0.8,
    SourceKind.BLOG: 0.7,
    SourceKind.WEB: 0.6,
    SourceKind.FORUM: 0.5,
    SourceKind.DATASET: 0.4,
}

_SOURCE_GROUP_BY_KIND: dict[SourceKind, SourceGroup] = {
    SourceKind.PAPER: SourceGroup.PAPERS,
    SourceKind.DOCS: SourceGroup.DOCS,
    SourceKind.REPOSITORY: SourceGroup.REPOS,
    SourceKind.BENCHMARK: SourceGroup.BENCHMARKS,
    SourceKind.BLOG: SourceGroup.BLOGS,
    SourceKind.FORUM: SourceGroup.FORUMS,
    SourceKind.WEB: SourceGroup.WEB,
    SourceKind.DATASET: SourceGroup.WEB,
}

_SOURCE_GROUP_PRIOR: dict[SourceGroup, float] = {
    SourceGroup.DOCS: 0.9,
    SourceGroup.REPOS: 0.88,
    SourceGroup.BENCHMARKS: 0.84,
    SourceGroup.BLOGS: 0.72,
    SourceGroup.WEB: 0.68,
    SourceGroup.PAPERS: 0.7,
    SourceGroup.FORUMS: 0.55,
    SourceGroup.NEWS: 0.5,
    SourceGroup.SOCIAL: 0.35,
}

_DEFAULT_QUALITY = 0.45
_DEFAULT_NOVELTY = 0.45
_ENGINEERING_WEB_GROUPS = {
    SourceGroup.DOCS,
    SourceGroup.REPOS,
    SourceGroup.BENCHMARKS,
    SourceGroup.BLOGS,
    SourceGroup.WEB,
}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _normalise_source_group(value: object) -> SourceGroup | None:
    if isinstance(value, SourceGroup):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    for source_group in SourceGroup:
        if normalized == source_group.value or normalized == source_group.name.lower():
            return source_group
    return None


def infer_source_group(candidate: EvidenceCandidate) -> SourceGroup:
    metadata_group = _normalise_source_group(candidate.raw_metadata.get("source_group"))
    if metadata_group is not None:
        return metadata_group
    return _SOURCE_GROUP_BY_KIND.get(candidate.source_kind, SourceGroup.WEB)


def candidate_domain(candidate: EvidenceCandidate) -> str:
    hostname = urlparse(str(candidate.url)).hostname or ""
    return hostname.lower().removeprefix("www.")


def score_candidate_quality(candidate: EvidenceCandidate) -> float:
    """Return a quality score based on source type and current authority signal."""
    base_score = _QUALITY_BY_SOURCE_KIND.get(candidate.source_kind, _DEFAULT_QUALITY)
    quality = max(base_score, candidate.authority_score)
    return _clamp(quality)


def score_candidate_novelty(
    candidate: EvidenceCandidate,
    *,
    domain_counts: Counter[str] | None = None,
    title_counts: Counter[str] | None = None,
) -> float:
    title = candidate.title.strip().lower()
    domain = candidate_domain(candidate)
    matched_subtopic_bonus = min(len(candidate.matched_subtopics), 3) * 0.1
    snippet_bonus = 0.1 if candidate.snippets else 0.0
    penalty = 0.0
    if domain_counts is not None and domain:
        penalty += max(domain_counts[domain] - 1, 0) * 0.12
    if title_counts is not None and title:
        penalty += max(title_counts[title] - 1, 0) * 0.18
    return _clamp(_DEFAULT_NOVELTY + matched_subtopic_bonus + snippet_bonus - penalty)


def score_source_group_prior(
    candidate: EvidenceCandidate,
    plan: ResearchPlan | None = None,
) -> float:
    source_group = infer_source_group(candidate)
    prior = _SOURCE_GROUP_PRIOR.get(source_group, 0.5)
    if plan and plan.allowed_source_groups:
        allowed = {value.strip().lower() for value in plan.allowed_source_groups}
        if source_group.value in allowed or source_group.name.lower() in allowed:
            prior = min(1.0, prior + 0.1)
    if plan and source_group in _ENGINEERING_WEB_GROUPS and plan.approval_status in {
        "approved",
        "not_requested",
        "pending",
    }:
        comparison_hint = any(
            token in (plan.goal + " " + " ".join(plan.sections)).lower()
            for token in ("compare", "benchmark", "framework", "harness", "agent")
        )
        if comparison_hint:
            prior = min(1.0, prior + 0.08)
    return _clamp(prior)


def score_candidate_recency(candidate: EvidenceCandidate) -> float:
    if candidate.freshness_score > 0:
        return _clamp(candidate.freshness_score)
    recency_days = candidate.raw_metadata.get("age_days")
    if recency_days is None:
        return 0.5
    try:
        age_days = max(float(recency_days), 0.0)
    except (TypeError, ValueError):
        return 0.5
    half_life_days = 365.0
    return _clamp(exp(-age_days / half_life_days))


def combined_candidate_score(
    candidate: EvidenceCandidate,
    *,
    plan: ResearchPlan | None = None,
    domain_counts: Counter[str] | None = None,
    title_counts: Counter[str] | None = None,
) -> float:
    relevance = candidate.relevance_score
    authority = max(candidate.authority_score, score_candidate_quality(candidate))
    recency = score_candidate_recency(candidate)
    novelty = score_candidate_novelty(
        candidate,
        domain_counts=domain_counts,
        title_counts=title_counts,
    )
    source_prior = score_source_group_prior(candidate, plan=plan)
    score = (
        relevance * 0.35
        + authority * 0.25
        + recency * 0.1
        + novelty * 0.15
        + source_prior * 0.15
    )
    return _clamp(score)
