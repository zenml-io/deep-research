from functools import lru_cache

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import CoverageScore
from deep_research.prompts.loader import load_prompt


@lru_cache(maxsize=4)
def build_coverage_scorer_agent(model_name: str):
    """Create a Kitaru-wrapped PydanticAI agent that scores evidence coverage."""
    return wrap_agent(
        Agent(
            model_name,
            name="coverage_scorer",
            output_type=CoverageScore,
            instructions=load_prompt("coverage_scorer"),
        ),
    )
