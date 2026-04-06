from kitaru.adapters import pydantic_ai as kp
from pydantic_ai import Agent

from deep_research.models import RenderPayload
from deep_research.prompts.loader import load_prompt


def build_writer_agent(model_name: str):
    return kp.wrap(
        Agent(
            model_name,
            name="writer",
            result_type=RenderPayload,
            system_prompt=load_prompt("writer"),
        )
    )
