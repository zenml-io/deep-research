from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.evidence.ledger import truncate_ledger_for_context
from deep_research.evidence.resolution import resolve_coverage_entries
from deep_research.evidence.scoring import infer_source_group
from deep_research.models import (
    CoverageScore,
    EvidenceCandidate,
    EvidenceLedger,
    ResearchPlan,
)
from deep_research.observability import bootstrap_logfire, span


def _entry_haystack(entry: EvidenceCandidate, *, include_matched: bool = True) -> str:
    """Lowercased text blob used for substring-matching a subtopic against an entry."""
    parts = [entry.title.lower()]
    parts.extend(snippet.text.lower() for snippet in entry.snippets[:2])
    if include_matched:
        parts.extend(matched.lower() for matched in entry.matched_subtopics)
    return " ".join(parts)


def _supporting_entries(
    entries: Iterable[EvidenceCandidate], subtopic: str
) -> list[EvidenceCandidate]:
    normalized = subtopic.strip().lower()
    if not normalized:
        return []
    return [entry for entry in entries if normalized in _entry_haystack(entry)]


def _recompute_total(score: CoverageScore) -> CoverageScore:
    total = round(
        score.subtopic_coverage * 0.45
        + score.source_diversity * 0.25
        + score.evidence_density * 0.2
        + score.plan_fidelity * 0.1,
        4,
    )
    return score.model_copy(update={"total": total})


def _coverage_by_subtopic(
    entries: list[EvidenceCandidate], plan: ResearchPlan
) -> dict[str, float]:
    return {
        subtopic: round(min(len(_supporting_entries(entries, subtopic)) / 2, 1.0), 4)
        for subtopic in plan.subtopics
    }


def _source_group_diversity(entries: Iterable[EvidenceCandidate]) -> dict[str, int]:
    counts = Counter(infer_source_group(entry).value for entry in entries)
    return dict(sorted(counts.items()))


def _weak_claims(entries: list[EvidenceCandidate], plan: ResearchPlan) -> list[str]:
    claims: list[str] = []
    for subtopic in plan.subtopics:
        supporting = _supporting_entries(entries, subtopic)
        if len(supporting) == 1:
            claims.append(
                f"{subtopic}: only one supporting source ({supporting[0].title})"
            )
    for question in plan.key_questions:
        terms = {token for token in question.lower().split() if len(token) > 4}
        if not terms:
            continue
        covered = any(
            any(term in _entry_haystack(entry, include_matched=False) for term in terms)
            for entry in entries
        )
        if not covered:
            claims.append(f"{question}: no direct evidence match")
    return claims


def _marginal_info_gain(
    entries: list[EvidenceCandidate], score: CoverageScore
) -> float:
    if not entries:
        return 0.0
    supported_subtopics = len(
        {
            subtopic.strip().lower()
            for entry in entries
            for subtopic in entry.matched_subtopics
            if subtopic.strip()
        }
    )
    penalty = len(score.unanswered_questions) * 0.05
    gain = score.subtopic_coverage * 0.6 + min(supported_subtopics / len(entries), 0.4)
    return round(max(0.0, min(1.0, gain - penalty)), 4)


def _with_augmented_diagnostics(
    score: CoverageScore,
    *,
    entries: list[EvidenceCandidate],
    plan: ResearchPlan,
) -> CoverageScore:
    return score.model_copy(
        update={
            "coverage_by_subtopic": _coverage_by_subtopic(entries, plan),
            "source_group_diversity": _source_group_diversity(entries),
            "marginal_info_gain": _marginal_info_gain(entries, score),
            "weak_claims": _weak_claims(entries, plan),
        }
    )


def _empty_score(plan: ResearchPlan) -> CoverageScore:
    return _recompute_total(
        CoverageScore(
            subtopic_coverage=0.0,
            plan_fidelity=0.0,
            source_diversity=0.0,
            evidence_density=0.0,
            total=0.0,
            uncovered_subtopics=list(plan.subtopics),
            unanswered_questions=list(plan.key_questions),
        )
    )


def _build_prompt_payload(
    entries: list[EvidenceCandidate], plan: ResearchPlan
) -> dict:
    evidence_summary = [
        {
            "key": entry.key,
            "title": entry.title,
            "provider": entry.provider,
            "source_kind": entry.source_kind.value,
            "source_group": infer_source_group(entry).value,
            "matched_subtopics": entry.matched_subtopics,
            "snippet_preview": [s.text for s in entry.snippets[:3]],
        }
        for entry in entries
    ]
    return {
        "plan": {
            "subtopics": plan.subtopics,
            "key_questions": plan.key_questions,
        },
        "evidence_count": len(entries),
        "provider_diversity": list({e.provider for e in entries}),
        "source_group_diversity": _source_group_diversity(entries),
        "evidence": evidence_summary,
    }


@checkpoint(type="llm_call")
def score_coverage(
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig | None = None,
) -> CoverageScore:
    """Checkpoint: use an LLM agent to evaluate subtopic coverage."""
    from deep_research.agents.coverage_scorer import build_coverage_scorer_agent

    bootstrap_logfire()
    if config is None:
        from deep_research.enums import Tier

        config = ResearchConfig.for_tier(Tier.STANDARD)

    ledger = truncate_ledger_for_context(
        ledger,
        max_chars=config.coverage_context_budget_chars,
        role="coverage",
        snippet_budget_chars=config.context_snippet_budget_chars,
    )
    entries = resolve_coverage_entries(ledger)
    if not entries:
        return _empty_score(plan)

    agent = build_coverage_scorer_agent(config.coverage_scorer_model)
    with span(
        "coverage_scoring",
        evidence_count=len(entries),
        subtopic_count=len(plan.subtopics),
    ):
        result = agent.run_sync(
            serialize_prompt_payload(
                _build_prompt_payload(entries, plan),
                label="coverage scorer prompt",
            )
        )
    return _with_augmented_diagnostics(result.output, entries=entries, plan=plan)
