from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ResearchConfig
from deep_research.models import (
    CoverageScore,
    IterationRecord,
    ReplanDecision,
    ResearchPlan,
)
from deep_research.observability import bootstrap_logfire, span


@checkpoint(type="llm_call")
def evaluate_replan(
    plan: ResearchPlan,
    coverage: CoverageScore,
    iteration_history: list[IterationRecord],
    config: ResearchConfig,
) -> ReplanDecision:
    """Checkpoint: ask the LLM whether mid-run replanning is warranted."""
    from deep_research.agents.replanner import build_replanner_agent

    bootstrap_logfire()
    prompt_payload = {
        "plan": plan.model_dump(mode="json"),
        "coverage": coverage.model_dump(mode="json"),
        "iterations_completed": len(iteration_history),
        "recent_coverage_deltas": [r.coverage_delta for r in iteration_history[-3:]],
        "uncovered_subtopics": coverage.uncovered_subtopics,
    }
    agent = build_replanner_agent(config.planner_model)
    with span("replan_evaluation", iteration_count=len(iteration_history)):
        result = agent.run_sync(
            serialize_prompt_payload(prompt_payload, label="replanner prompt")
        )
    return result.output
