from __future__ import annotations

from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.agents.claim_extractor import build_claim_extractor_agent
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.evidence.ledger import truncate_ledger_for_context
from deep_research.evidence.scoring import infer_source_group
from deep_research.flow.costing import budget_from_agent_result
from deep_research.models import (
    ClaimExtractionResult,
    ClaimInventory,
    EvidenceLedger,
    RenderPayload,
    ResearchPlan,
)
from deep_research.observability import span


def _build_prompt_payload(
    renders: list[RenderPayload],
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> dict:
    report_text = "\n\n---\n\n".join(
        render.content_markdown for render in renders if render.content_markdown
    )
    ledger_summary = [
        {
            "key": entry.key,
            "title": entry.title,
            "provider": entry.provider,
            "source_kind": entry.source_kind.value,
            "source_group": infer_source_group(entry).value,
            "snippet_preview": [s.text for s in entry.snippets[:3]],
        }
        for entry in ledger.considered
    ]
    return {
        "report_text": report_text,
        "ledger": ledger_summary,
        "plan_subtopics": plan.subtopics,
        "plan_key_questions": plan.key_questions,
    }


@checkpoint(type="llm_call")
def extract_claims(
    renders: list[RenderPayload],
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig,
) -> ClaimExtractionResult:
    """Checkpoint: extract atomic claims from the rendered report and ground against ledger."""
    if not renders or not ledger.considered:
        return ClaimExtractionResult(inventory=ClaimInventory())

    truncated_ledger = truncate_ledger_for_context(
        ledger,
        max_chars=config.coverage_context_budget_chars,
        role="coverage",
        snippet_budget_chars=config.context_snippet_budget_chars,
    )

    agent = build_claim_extractor_agent(config.claim_extractor_model)
    with span(
        "extract_claims",
        render_count=len(renders),
        ledger_size=len(truncated_ledger.considered),
    ):
        result = agent.run_sync(
            serialize_prompt_payload(
                _build_prompt_payload(renders, truncated_ledger, plan),
                label="claim extractor prompt",
            )
        )
    return ClaimExtractionResult(
        inventory=result.output,
        budget=budget_from_agent_result(
            result,
            ModelPricing.model_validate(config.claim_extractor_pricing),
        ),
    )
