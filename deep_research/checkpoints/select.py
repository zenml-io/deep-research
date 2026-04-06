from kitaru import checkpoint

from deep_research.agents.curator import build_curator_agent
from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import EvidenceLedger, ResearchPlan, SelectionGraph


@checkpoint(type="llm_call")
def build_selection_graph(ledger: EvidenceLedger, plan: ResearchPlan) -> SelectionGraph:
    model_name = ResearchConfig.for_tier(Tier.STANDARD).curator_model
    agent = build_curator_agent(model_name)
    prompt = {
        "plan": plan.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
    }
    return agent.run_sync(prompt).output
