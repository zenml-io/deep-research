from kitaru import checkpoint

from deep_research.agents.supervisor import build_supervisor_agent
from deep_research.config import ResearchConfig
from deep_research.models import (
    EvidenceLedger,
    ResearchPlan,
    SupervisorCheckpointResult,
)


def _execute_supervisor_turn(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    config: ResearchConfig,
    model_name: str | None = None,
) -> SupervisorCheckpointResult:
    agent = build_supervisor_agent(
        model_name or config.supervisor_model,
        toolsets=[],
        tools=[],
    )
    prompt = {
        "plan": plan.model_dump(mode="json"),
        "ledger": ledger.model_dump(mode="json"),
        "iteration": iteration,
        "tier": config.tier.value,
    }
    return agent.run_sync(prompt).output


@checkpoint(type="llm_call")
def run_supervisor(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    config: ResearchConfig,
) -> SupervisorCheckpointResult:
    return _execute_supervisor_turn(plan, ledger, iteration, config)
