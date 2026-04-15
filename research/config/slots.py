"""Model slot definitions for V2 config.

ModelSlot is a StrEnum naming the logical roles an LLM fills.
ModelSlotConfig holds the provider/model string and per-token costs.
"""

from enum import StrEnum

from pydantic import BaseModel


class ModelSlot(StrEnum):
    """Logical roles an LLM fills during a research run."""

    generator = "generator"
    subagent = "subagent"
    reviewer = "reviewer"
    judge = "judge"


class ModelSlotConfig(BaseModel):
    """Configuration for a single model slot.

    provider + model together form the PydanticAI model string,
    e.g. provider="anthropic", model="claude-sonnet-4-6"
    → "anthropic:claude-sonnet-4-6".

    model_settings is an optional dict of provider-specific PydanticAI
    model settings (e.g. ``{"google_thinking_config": {"thinking_level": "high"}}``
    for Gemini models with thinking enabled).
    """

    provider: str
    model: str
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    model_settings: dict[str, object] | None = None

    @property
    def model_string(self) -> str:
        """PydanticAI-style provider:model string."""
        return f"{self.provider}:{self.model}"
