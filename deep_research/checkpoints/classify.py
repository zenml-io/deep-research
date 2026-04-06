from kitaru import checkpoint

from deep_research.agents.classifier import build_classifier_agent
from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.models import RequestClassification


@checkpoint(type="llm_call")
def classify_request(
    brief: str, config: ResearchConfig | None = None
) -> RequestClassification:
    model_name = (
        config.classifier_model
        if config is not None
        else ResearchConfig.for_tier(Tier.STANDARD).classifier_model
    )
    return build_classifier_agent(model_name).run_sync(brief).output
