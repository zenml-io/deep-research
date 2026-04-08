from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import CoherenceResult, GroundingResult
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=8)
def build_grounding_judge_agent(model_name: str) -> Agent[None, GroundingResult]:
    """Create a Kitaru-wrapped grounding judge agent."""
    return wrap_agent(
        Agent(
            model_name,
            name="grounding_judge",
            output_type=GroundingResult,
            instructions=load_prompt("judge_grounding"),
        )
    )


@lru_cache(maxsize=8)
def build_coherence_judge_agent(model_name: str) -> Agent[None, CoherenceResult]:
    """Create a Kitaru-wrapped coherence judge agent."""
    return wrap_agent(
        Agent(
            model_name,
            name="coherence_judge",
            output_type=CoherenceResult,
            instructions=load_prompt("judge_coherence"),
        )
    )
