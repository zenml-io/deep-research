import json

from kitaru import checkpoint

from deep_research.agents.classifier import build_classifier_agent
from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import RequestClassification
from deep_research.observability import span


@checkpoint(type="llm_call")
def classify_request(
    brief: str,
    config: ResearchConfig | None = None,
    seeded_entities: dict[str, list[str]] | None = None,
) -> RequestClassification:
    """Checkpoint: classify a research brief into audience, freshness, and tier."""
    with span("classify_request"):
        model_name = (
            config.classifier_model
            if config is not None
            else ResearchConfig.for_tier(Tier.STANDARD).classifier_model
        )
        payload = (
            json.dumps({"brief": brief, "seeded_entities": seeded_entities}, indent=2)
            if seeded_entities
            else brief
        )
        return build_classifier_agent(model_name).run_sync(payload).output
