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

_COMPARISON_TOKENS = {"compare", "benchmark", "framework", "harness", "agent"}


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _normalise_source_group(value: object) -> SourceGroup | None:
    """Parse a raw metadata value into a SourceGroup, tolerating strings and None."""
    if isinstance(value, SourceGroup):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    for group in SourceGroup:
        if normalized == group.value or normalized == group.name.lower():
            return group
    return None


def infer_source_group(candidate: EvidenceCandidate) -> SourceGroup:
    """Return the source group for a candidate, preferring raw_metadata over source_kind."""
    metadata_group = _normalise_source_group(candidate.raw_metadata.get("source_group"))
    if metadata_group is not None:
        return metadata_group
    return _SOURCE_GROUP_BY_KIND.get(candidate.source_kind, SourceGroup.WEB)


def _candidate_domain(candidate: EvidenceCandidate) -> str:
    hostname = urlparse(str(candidate.url)).hostname or ""
    return hostname.lower().removeprefix("www.")


def score_candidate_quality(candidate: EvidenceCandidate) -> float:
    """Return a quality score based on source type and current authority signal."""
    base_score = _QUALITY_BY_SOURCE_KIND.get(candidate.source_kind, _DEFAULT_QUALITY)
    return _clamp(max(base_score, candidate.authority_score))


def score_candidate_novelty(
    candidate: EvidenceCandidate,
    *,
    domain_counts: Counter[str] | None = None,
    title_counts: Counter[str] | None = None,
) -> float:
    """Return a novelty score penalised for repeated domains or duplicate titles."""
    title = candidate.title.strip().lower()
    domain = _candidate_domain(candidate)
    bonus = min(len(candidate.matched_subtopics), 3) * 0.1 + (0.1 if candidate.snippets else 0.0)
    penalty = 0.0
    if domain_counts is not None and domain:
        penalty += max(domain_counts[domain] - 1, 0) * 0.12
    if title_counts is not None and title:
        penalty += max(title_counts[title] - 1, 0) * 0.18
    return _clamp(_DEFAULT_NOVELTY + bonus - penalty)


def _score_source_group_prior(
    candidate: EvidenceCandidate,
    plan: ResearchPlan | None = None,
) -> float:
    """Return a base prior for the candidate's source group, boosted by plan signals."""
    source_group = infer_source_group(candidate)
    prior = _SOURCE_GROUP_PRIOR.get(source_group, 0.5)
    if plan:
        if plan.allowed_source_groups:
            allowed = {v.strip().lower() for v in plan.allowed_source_groups}
            if source_group.value in allowed or source_group.name.lower() in allowed:
                prior = min(1.0, prior + 0.1)
        if source_group in _ENGINEERING_WEB_GROUPS and plan.approval_status != "rejected":
            plan_text = (plan.goal + " " + " ".join(plan.sections)).lower()
            if any(token in plan_text for token in _COMPARISON_TOKENS):
                prior = min(1.0, prior + 0.08)
    return _clamp(prior)


def _score_candidate_recency(candidate: EvidenceCandidate) -> float:
    """Return a recency score decaying exponentially with age, using a 365-day half-life."""
    if candidate.freshness_score > 0:
        return _clamp(candidate.freshness_score)
    recency_days = candidate.raw_metadata.get("age_days")
    if recency_days is None:
        return 0.5
    try:
        age_days = max(float(recency_days), 0.0)
    except (TypeError, ValueError):
        return 0.5
    return _clamp(exp(-age_days / 365.0))


def combined_candidate_score(
    candidate: EvidenceCandidate,
    *,
    plan: ResearchPlan | None = None,
    domain_counts: Counter[str] | None = None,
    title_counts: Counter[str] | None = None,
) -> float:
    """Return a weighted composite score across relevance, authority, recency, novelty, and source prior."""
    score = (
        candidate.relevance_score * 0.35
        + score_candidate_quality(candidate) * 0.25
        + _score_candidate_recency(candidate) * 0.1
        + score_candidate_novelty(candidate, domain_counts=domain_counts, title_counts=title_counts) * 0.15
        + _score_source_group_prior(candidate, plan=plan) * 0.15
    )
    return _clamp(score)
