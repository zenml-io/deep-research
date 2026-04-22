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

    The generator returns plain markdown text (``output_type=str``); this
    checkpoint wraps it in a :class:`DraftReport` by extracting section headings.
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
