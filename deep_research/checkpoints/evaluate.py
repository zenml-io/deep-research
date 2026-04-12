from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.evidence.ledger import truncate_ledger_for_context
from deep_research.evidence.resolution import resolve_coverage_entries
from deep_research.models import CoverageScore, EvidenceLedger, ResearchPlan
from deep_research.observability import bootstrap_logfire, span


def _with_recomputed_total(score: CoverageScore) -> CoverageScore:
    if "plan_fidelity" not in score.model_fields_set:
        return score
    total = round(
        (
            score.subtopic_coverage
            + score.plan_fidelity
            + score.source_diversity
            + score.evidence_density
        )
        / 4,
        4,
    )
    return score.model_copy(update={"total": total})


@checkpoint(type="llm_call")
def score_coverage(
    ledger: EvidenceLedger, plan: ResearchPlan, config: ResearchConfig
) -> CoverageScore:
    """Checkpoint: use an LLM agent to evaluate subtopic coverage."""
    from deep_research.agents.coverage_scorer import build_coverage_scorer_agent

    bootstrap_logfire()
    ledger = truncate_ledger_for_context(
        ledger,
        max_chars=config.coverage_context_budget_chars,
        role="coverage",
        snippet_budget_chars=config.context_snippet_budget_chars,
    )
    entries = resolve_coverage_entries(ledger)
    if not entries:
        return _with_recomputed_total(
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

    evidence_summary = []
    for entry in entries:
        snippet_texts = [s.text for s in entry.snippets]
        evidence_summary.append(
            {
                "key": entry.key,
                "title": entry.title,
                "provider": entry.provider,
                "source_kind": entry.source_kind.value,
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
    return _with_recomputed_total(result.output)
