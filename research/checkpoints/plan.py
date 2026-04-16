"""Plan checkpoint — runs the planner agent to produce a ResearchPlan."""

import json

from kitaru import checkpoint

from research.agents.planner import build_planner_agent
from research.contracts.brief import ResearchBrief
from research.contracts.plan import ResearchPlan


@checkpoint(type="llm_call")
def run_plan(brief: ResearchBrief, model_name: str) -> ResearchPlan:
    """Checkpoint: generate a structured research plan from a brief."""
    agent = build_planner_agent(model_name)
    prompt = json.dumps(brief.model_dump(mode="json"), indent=2)
    return agent.run_sync(prompt).output
