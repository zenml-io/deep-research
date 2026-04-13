import json

from kitaru import checkpoint

from deep_research.agents.planner import build_planner_agent
from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import (
    RequestClassification,
    ResearchPlan,
)
from deep_research.observability import span


@checkpoint(type="llm_call")
def build_plan(
    brief: str,
    classification: RequestClassification,
    tier: Tier,
    seeded_entities: dict[str, list[str]] | None = None,
) -> ResearchPlan:
    """Checkpoint: generate a structured research plan from the brief, classification, and preferences."""
    with span("build_plan", tier=tier):
        model_name = ResearchConfig.for_tier(tier).planner_model
        preferences = classification.preferences
        prompt_parts = {
            "brief": brief,
            "classification": {
                "audience_mode": classification.audience_mode,
                "freshness_mode": classification.freshness_mode,
                "recommended_tier": classification.recommended_tier.value,
            },
            "preferences": preferences.model_dump(mode="json"),
        }
        if seeded_entities:
            prompt_parts["seeded_entities"] = seeded_entities
        if not preferences.preferred_source_groups and (
            preferences.comparison_targets
            or preferences.deliverable_mode.value == "comparison_memo"
        ):
            prompt_parts["planning_guidance"] = {
                "web_first_default": True,
                "exact_name_queries_first": True,
            }
        prompt = json.dumps(prompt_parts, indent=2)
        return build_planner_agent(model_name).run_sync(prompt).output
