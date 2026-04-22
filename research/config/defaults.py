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
    breadth_first: bool = False
    respect_supervisor_done: bool = True
    default_budget_usd: float | None = None


# Shared model slot configs — single change point when upgrading a model.
# All four tiers use the same generator/subagent/reviewer baseline; deeper
# tiers add judge, second_reviewer, and scope_override on top.
_GENERATOR_SLOT = ModelSlotConfig(provider="anthropic", model="claude-sonnet-4-6")
_SUBAGENT_SLOT = ModelSlotConfig(
    provider="google-gla", model="gemini-3.1-flash-lite-preview"
)
_REVIEWER_SLOT = ModelSlotConfig(provider="openai", model="gpt-5.4-mini")
_JUDGE_SLOT = ModelSlotConfig(
    provider="google-gla",
    model="gemini-3.1-pro-preview",
    model_settings={"google_thinking_config": {"thinking_level": "high"}},
)
_SECOND_REVIEWER_SLOT = ModelSlotConfig(
    provider="google-gla", model="gemini-3.1-pro-preview"
)
_SCOPE_OVERRIDE_SLOT = ModelSlotConfig(provider="anthropic", model="claude-opus-4-6")


TIER_DEFAULTS: dict[str, TierDefaults] = {
    "quick": TierDefaults(
        slots={
            ModelSlot.generator: _GENERATOR_SLOT,
            ModelSlot.subagent: _SUBAGENT_SLOT,
            ModelSlot.reviewer: _REVIEWER_SLOT,
            # judge is not used in quick tier
        },
        max_iterations=2,
    ),
    "standard": TierDefaults(
        slots={
            ModelSlot.generator: _GENERATOR_SLOT,
            ModelSlot.subagent: _SUBAGENT_SLOT,
            ModelSlot.reviewer: _REVIEWER_SLOT,
            ModelSlot.judge: _JUDGE_SLOT,
        },
        max_iterations=5,
    ),
    "deep": TierDefaults(
        slots={
            ModelSlot.generator: _GENERATOR_SLOT,
            ModelSlot.subagent: _SUBAGENT_SLOT,
            ModelSlot.reviewer: _REVIEWER_SLOT,
            ModelSlot.judge: _JUDGE_SLOT,
        },
        max_iterations=10,
        second_reviewer=_SECOND_REVIEWER_SLOT,
        scope_override=_SCOPE_OVERRIDE_SLOT,
    ),
    "exhaustive": TierDefaults(
        slots={
            ModelSlot.generator: _GENERATOR_SLOT,
            ModelSlot.subagent: _SUBAGENT_SLOT,
            ModelSlot.reviewer: _REVIEWER_SLOT,
            ModelSlot.judge: _JUDGE_SLOT,
        },
        max_iterations=20,
        max_parallel_subagents=10,
        second_reviewer=_SECOND_REVIEWER_SLOT,
        scope_override=_SCOPE_OVERRIDE_SLOT,
        breadth_first=True,
        respect_supervisor_done=False,
        default_budget_usd=3.00,
    ),
}
