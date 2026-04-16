"""Plan-revision checkpoint — revises a ResearchPlan from critique feedback."""

import json

from kitaru import checkpoint

from research.agents.replanner import build_replanner_agent
from research.contracts.brief import ResearchBrief
from research.contracts.plan import ResearchPlan
from research.contracts.reports import CritiqueReport


@checkpoint(type="llm_call")
def run_plan_revision(
    brief: ResearchBrief,
    plan: ResearchPlan,
    critique: CritiqueReport,
    ledger_projection: str,
    model_name: str,
) -> ResearchPlan:
    """Checkpoint: revise a research plan for one supplemental loop."""
    agent = build_replanner_agent(model_name)
    prompt = json.dumps(
        {
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "critique": critique.model_dump(mode="json"),
            "ledger_projection": ledger_projection,
        },
        indent=2,
    )
    return agent.run_sync(prompt).output
