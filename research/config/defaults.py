"""Tier default configurations for quick, standard, and deep runs."""

from pydantic import BaseModel

from research.config.slots import ModelSlot, ModelSlotConfig


class TierDefaults(BaseModel):
    """Default configuration for a research tier.

    slots maps each ModelSlot to its ModelSlotConfig (or omits it if
    the slot is unused for that tier).

    second_reviewer is only populated for the deep tier, which uses
    two independent reviewers for cross-validation.

    scope_override, when set, overrides the generator model for the
    scope agent only — allows deep tier to use a more capable model
    (e.g. Opus) for the brief without upgrading all generator uses.
    """

    slots: dict[ModelSlot, ModelSlotConfig]
    max_iterations: int
    max_parallel_subagents: int = 3
    second_reviewer: ModelSlotConfig | None = None
    scope_override: ModelSlotConfig | None = None


TIER_DEFAULTS: dict[str, TierDefaults] = {
    "quick": TierDefaults(
        slots={
            ModelSlot.generator: ModelSlotConfig(
                provider="anthropic", model="claude-sonnet-4-6"
            ),
            ModelSlot.subagent: ModelSlotConfig(
                provider="google-gla", model="gemini-3.1-flash-lite-preview"
            ),
            ModelSlot.reviewer: ModelSlotConfig(
                provider="openai", model="gpt-5.4-mini"
            ),
            # judge is not used in quick tier
        },
        max_iterations=2,
    ),
    "standard": TierDefaults(
        slots={
            ModelSlot.generator: ModelSlotConfig(
                provider="anthropic", model="claude-sonnet-4-6"
            ),
            ModelSlot.subagent: ModelSlotConfig(
                provider="google-gla", model="gemini-3.1-flash-lite-preview"
            ),
            ModelSlot.reviewer: ModelSlotConfig(
                provider="openai", model="gpt-5.4-mini"
            ),
            ModelSlot.judge: ModelSlotConfig(
                provider="google-gla",
                model="gemini-3.1-pro-preview",
                model_settings={"google_thinking_config": {"thinking_level": "high"}},
            ),
        },
        max_iterations=5,
    ),
    "deep": TierDefaults(
        slots={
            ModelSlot.generator: ModelSlotConfig(
                provider="anthropic", model="claude-sonnet-4-6"
            ),
            ModelSlot.subagent: ModelSlotConfig(
                provider="google-gla", model="gemini-3.1-flash-lite-preview"
            ),
            ModelSlot.reviewer: ModelSlotConfig(
                provider="openai", model="gpt-5.4-mini"
            ),
            ModelSlot.judge: ModelSlotConfig(
                provider="google-gla",
                model="gemini-3.1-pro-preview",
                model_settings={"google_thinking_config": {"thinking_level": "high"}},
            ),
        },
        max_iterations=10,
        second_reviewer=ModelSlotConfig(
            provider="google-gla", model="gemini-3.1-pro-preview"
        ),
        scope_override=ModelSlotConfig(provider="anthropic", model="claude-opus-4-6"),
    ),
}
