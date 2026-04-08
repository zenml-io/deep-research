from typing import Any

from pydantic_ai import Agent

from deep_research.agents._kitaru import wrap_agent
from deep_research.models import SupervisorDecision
from deep_research.prompts.loader import load_prompt


def build_supervisor_agent(model_name: str, toolsets: list[Any], tools: list[Any]):
    """Create a Kitaru-wrapped PydanticAI supervisor agent with tool capture."""
    return wrap_agent(
        Agent(
            model_name,
            name="supervisor",
            output_type=SupervisorDecision,
            instructions=load_prompt("supervisor"),
            toolsets=toolsets,
            tools=tools,
        ),
        tool_capture_config={"mode": "full"},
    )
