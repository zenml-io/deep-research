from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import RequestClassification
from deep_research.prompts.loader import load_prompt


def build_classifier_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent for classifying research briefs."""
    return kp.wrap(
        Agent(
            model_name,
            name="classifier",
            output_type=RequestClassification,
            system_prompt=load_prompt("classifier"),
        )
    )
