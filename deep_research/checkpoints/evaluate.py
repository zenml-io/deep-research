from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.evidence.resolution import resolve_coverage_entries
from deep_research.models import CoverageScore, EvidenceLedger, ResearchPlan
from deep_research.observability import bootstrap_logfire, span


@checkpoint(type="llm_call")
def score_coverage(
    ledger: EvidenceLedger, plan: ResearchPlan, config: ResearchConfig
) -> CoverageScore:
    """Checkpoint: use an LLM agent to evaluate subtopic coverage."""
    from deep_research.agents.coverage_scorer import build_coverage_scorer_agent

    bootstrap_logfire()
    entries = resolve_coverage_entries(ledger)
    if not entries:
        return CoverageScore(
            subtopic_coverage=0.0,
            source_diversity=0.0,
            evidence_density=0.0,
            total=0.0,
            uncovered_subtopics=list(plan.subtopics),
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
    return result.output
