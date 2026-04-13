from __future__ import annotations

from collections import Counter

from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.evidence.ledger import truncate_ledger_for_context
from deep_research.evidence.resolution import resolve_coverage_entries
from deep_research.evidence.scoring import infer_source_group
from deep_research.models import CoverageScore, EvidenceLedger, ResearchPlan
from deep_research.observability import bootstrap_logfire, span


def _with_recomputed_total(score: CoverageScore) -> CoverageScore:
    total = round(
        (
            score.subtopic_coverage * 0.45
            + score.source_diversity * 0.25
            + score.evidence_density * 0.2
            + score.plan_fidelity * 0.1
        ),
        4,
    )
    if "total" not in type(score).model_fields:
        return score
    return score.model_copy(update={"total": total})


def _coverage_by_subtopic(entries, plan: ResearchPlan) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for subtopic in plan.subtopics:
        normalized = subtopic.strip().lower()
        supporting_entries = 0
        for entry in entries:
            entry_text = " ".join(
                [
                    entry.title.lower(),
                    *(snippet.text.lower() for snippet in entry.snippets[:2]),
                    *[matched.lower() for matched in entry.matched_subtopics],
                ]
            )
            if normalized and normalized in entry_text:
                supporting_entries += 1
        coverage[subtopic] = round(min(supporting_entries / 2, 1.0), 4)
    return coverage


def _source_group_diversity(entries) -> dict[str, int]:
    counts = Counter(infer_source_group(entry).value for entry in entries)
    return dict(sorted(counts.items()))


def _weak_claims(entries, plan: ResearchPlan) -> list[str]:
    weak_claims: list[str] = []
    for subtopic in plan.subtopics:
        normalized = subtopic.strip().lower()
        supporting_entries = [
            entry
            for entry in entries
            if normalized in " ".join(
                [
                    entry.title.lower(),
                    *(snippet.text.lower() for snippet in entry.snippets[:2]),
                    *[matched.lower() for matched in entry.matched_subtopics],
                ]
            )
        ]
        if len(supporting_entries) == 1:
            weak_claims.append(
                f"{subtopic}: only one supporting source ({supporting_entries[0].title})"
            )
    for question in plan.key_questions:
        question_terms = {token for token in question.lower().split() if len(token) > 4}
        if not question_terms:
            continue
        covered = False
        for entry in entries:
            entry_text = " ".join(
                [entry.title.lower(), *(snippet.text.lower() for snippet in entry.snippets[:2])]
            )
            if any(term in entry_text for term in question_terms):
                covered = True
                break
        if not covered:
            weak_claims.append(f"{question}: no direct evidence match")
    return weak_claims


def _marginal_info_gain(entries, score: CoverageScore) -> float:
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
    unanswered_penalty = len(score.unanswered_questions) * 0.05
    gain = (score.subtopic_coverage * 0.6) + min(supported_subtopics / max(len(entries), 1), 0.4)
    return round(max(0.0, min(1.0, gain - unanswered_penalty)), 4)


def _with_augmented_diagnostics(
    score: CoverageScore,
    *,
    entries,
    plan: ResearchPlan,
) -> CoverageScore:
    update = {}
    coverage_by_subtopic = _coverage_by_subtopic(entries, plan)
    source_group_diversity = _source_group_diversity(entries)
    weak_claims = _weak_claims(entries, plan)
    marginal_info_gain = _marginal_info_gain(entries, score)

    score_fields = type(score).model_fields
    if "coverage_by_subtopic" in score_fields:
        update["coverage_by_subtopic"] = coverage_by_subtopic
    if "source_group_diversity" in score_fields:
        update["source_group_diversity"] = source_group_diversity
    if "marginal_info_gain" in score_fields:
        update["marginal_info_gain"] = marginal_info_gain
    if "weak_claims" in score_fields:
        update["weak_claims"] = weak_claims
    if update:
        score = score.model_copy(update=update)
    return score


def _empty_score(plan: ResearchPlan) -> CoverageScore:
    score = CoverageScore(
        subtopic_coverage=0.0,
        plan_fidelity=0.0,
        source_diversity=0.0,
        evidence_density=0.0,
        total=0.0,
        uncovered_subtopics=list(plan.subtopics),
        unanswered_questions=list(plan.key_questions),
    )
    return _with_recomputed_total(score)


def _normalize_result(score: CoverageScore, *, entries, plan: ResearchPlan) -> CoverageScore:
    return _with_augmented_diagnostics(score, entries=entries, plan=plan)


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

    evidence_summary = []
    for entry in entries:
        snippet_texts = [s.text for s in entry.snippets]
        evidence_summary.append(
            {
                "key": entry.key,
                "title": entry.title,
                "provider": entry.provider,
                "source_kind": entry.source_kind.value,
                "source_group": infer_source_group(entry).value,
                "matched_subtopics": entry.matched_subtopics,
                "snippet_preview": snippet_texts[:3],
            }
        )

    prompt_payload = {
        "plan": {
            "subtopics": plan.subtopics,
            "key_questions": plan.key_questions,
        },
        "evidence_count": len(entries),
        "provider_diversity": list({e.provider for e in entries}),
        "source_group_diversity": _source_group_diversity(entries),
        "evidence": evidence_summary,
    }

    agent = build_coverage_scorer_agent(config.coverage_scorer_model)
    with span(
        "coverage_scoring",
        evidence_count=len(entries),
        subtopic_count=len(plan.subtopics),
    ):
        result = agent.run_sync(
            serialize_prompt_payload(prompt_payload, label="coverage scorer prompt")
        )
    return _normalize_result(result.output, entries=entries, plan=plan)
