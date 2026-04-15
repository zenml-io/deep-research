"""Draft checkpoint — generator agent produces a DraftReport from evidence."""

import json

from kitaru import checkpoint

from research.agents.generator import build_generator_agent
from research.contracts.brief import ResearchBrief
from research.contracts.evidence import EvidenceLedger
from research.contracts.plan import ResearchPlan
from research.contracts.reports import DraftReport


@checkpoint(type="llm_call")
def run_draft(
    brief: ResearchBrief,
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    model_name: str,
) -> DraftReport:
    """Checkpoint: generate a draft report from the evidence ledger.

    The generator agent returns plain markdown text (``output_type=str``).
    This checkpoint wraps the output in a :class:`DraftReport` by extracting
    section headings from the markdown.

    Args:
        brief: The normalized research brief.
        plan: The research plan.
        ledger: The evidence ledger with all collected items.
        model_name: PydanticAI model string for the generator agent.

    Returns:
        A DraftReport with markdown content and section headings.
    """
    agent = build_generator_agent(model_name)
    prompt = json.dumps(
        {
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "ledger": ledger.model_dump(mode="json"),
        },
        indent=2,
    )
    markdown: str = agent.run_sync(prompt).output
    return DraftReport.from_markdown(markdown)
