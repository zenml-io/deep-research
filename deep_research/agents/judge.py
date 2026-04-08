from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import CoherenceResult, GroundingResult
from deep_research.prompts.loader import load_prompt


def build_grounding_judge_agent(model_name: str):
    """Create a Kitaru-wrapped grounding judge agent."""
    return kp.wrap(
        Agent(
            model_name,
            name="grounding_judge",
            output_type=GroundingResult,
            instructions=load_prompt("judge_grounding"),
        )
    )


def build_coherence_judge_agent(model_name: str):
    """Create a Kitaru-wrapped coherence judge agent."""
    return kp.wrap(
        Agent(
            model_name,
            name="coherence_judge",
            output_type=CoherenceResult,
            instructions=load_prompt("judge_coherence"),
        )
    )
