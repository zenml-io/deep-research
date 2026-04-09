from __future__ import annotations

from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import RelevanceScorerOutput
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_relevance_scorer_agent(model_name: str) -> Agent[None, RelevanceScorerOutput]:
    """Create a Kitaru-wrapped PydanticAI agent for scoring evidence relevance."""
    return wrap_agent(
        Agent(
            model_name,
            name="relevance_scorer",
            output_type=RelevanceScorerOutput,
            instructions=load_prompt("relevance_scorer"),
        )
    )
