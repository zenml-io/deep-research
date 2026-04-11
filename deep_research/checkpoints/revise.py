from kitaru import checkpoint

from deep_research.agent_io import serialize_prompt_payload
from deep_research.config import ModelPricing, ResearchConfig
from deep_research.flow.costing import budget_from_agent_result, merge_usage
from deep_research.models import (
    CritiqueResult,
    IterationBudget,
    RenderPayload,
    RenderProse,
    ResearchPlan,
    RevisionCheckpointResult,
)
from deep_research.observability import bootstrap_logfire, span


@checkpoint(type="llm_call")
def apply_revisions(
    renders: list[RenderPayload],
    critique: CritiqueResult,
    plan: ResearchPlan,
    config: ResearchConfig,
) -> RevisionCheckpointResult:
    """Checkpoint: regenerate prose for each render based on critique feedback."""
    from deep_research.agents.writer import build_writer_agent

    bootstrap_logfire()
    agent = build_writer_agent(config.writer_model)
    pricing = ModelPricing.model_validate(config.writer_pricing)
    revised: list[RenderPayload] = []
    total_budget = IterationBudget()

    for render in renders:
        prompt_payload = {
            "original_markdown": render.content_markdown,
            "critique_summary": critique.summary,
            "revision_suggestions": critique.revision_suggestions,
            "plan_goal": plan.goal,
            "plan_sections": plan.sections,
            "render_label": render.name,
        }
        with span("revise_render", render_name=render.name):
            result = agent.run_sync(
                serialize_prompt_payload(prompt_payload, label="revision prompt")
            )
        total_budget = merge_usage(
            total_budget, budget_from_agent_result(result, pricing)
        )
        revised.append(
            render.model_copy(
                update={
                    "content_markdown": result.output.content_markdown,
                    "structured_content": {
                        **(render.structured_content or {}),
                        "critique_summary": critique.summary,
                        "revision_applied": True,
                    },
                }
            )
        )
    return RevisionCheckpointResult(renders=revised, budget=total_budget)
