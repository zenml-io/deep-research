from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import RequestClassification
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_classifier_agent(model_name: str) -> Agent[None, RequestClassification]:
    """Create a Kitaru-wrapped PydanticAI agent for classifying research briefs."""
    return wrap_agent(
        Agent(
            model_name,
            name="classifier",
            output_type=RequestClassification,
            instructions=load_prompt("classifier"),
        )
    )
