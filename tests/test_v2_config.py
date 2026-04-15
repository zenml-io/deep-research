"""Tests for V2 config system: budget, model slots, tier defaults, settings."""

import pytest
from pydantic import ValidationError

from research.config import (
    TIER_DEFAULTS,
    BudgetConfig,
    ModelSlot,
    ModelSlotConfig,
    ResearchConfig,
    ResearchSettings,
    TierDefaults,
)


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_is_exceeded_returns_true_when_at_soft_limit(self):
        b = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.10)
        assert b.is_exceeded() is True

    def test_is_exceeded_returns_true_when_above_soft_limit(self):
        b = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.15)
        assert b.is_exceeded() is True

    def test_is_exceeded_returns_false_when_below_soft_limit(self):
        b = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.05)
        assert b.is_exceeded() is False

    def test_is_hard_exceeded_returns_false_when_no_hard_limit(self):
        b = BudgetConfig(spent_usd=100.0)
        assert b.is_hard_exceeded() is False

    def test_is_hard_exceeded_returns_true_when_at_hard_limit(self):
        b = BudgetConfig(hard_budget_usd=1.0, spent_usd=1.0)
        assert b.is_hard_exceeded() is True

    def test_is_hard_exceeded_returns_true_when_above_hard_limit(self):
        b = BudgetConfig(hard_budget_usd=1.0, spent_usd=1.5)
        assert b.is_hard_exceeded() is True

    def test_is_hard_exceeded_returns_false_when_below_hard_limit(self):
        b = BudgetConfig(hard_budget_usd=1.0, spent_usd=0.5)
        assert b.is_hard_exceeded() is False

    def test_budget_is_mutable(self):
        """BudgetConfig must NOT be frozen — spent_usd updates during a run."""
        b = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.0)
        b.spent_usd = 0.05
        assert b.spent_usd == 0.05


# ---------------------------------------------------------------------------
# ModelSlot enum
# ---------------------------------------------------------------------------


class TestModelSlot:
    def test_has_all_four_values(self):
        expected = {"generator", "subagent", "reviewer", "judge"}
        actual = {s.value for s in ModelSlot}
        assert actual == expected

    def test_is_str_enum(self):
        assert isinstance(ModelSlot.generator, str)
        assert ModelSlot.generator == "generator"


# ---------------------------------------------------------------------------
# ModelSlotConfig
# ---------------------------------------------------------------------------


class TestModelSlotConfig:
    def test_model_string_property(self):
        cfg = ModelSlotConfig(provider="anthropic", model="claude-sonnet-4-6")
        assert cfg.model_string == "anthropic:claude-sonnet-4-6"

    def test_default_costs_are_zero(self):
        cfg = ModelSlotConfig(provider="openai", model="gpt-5.4-mini")
        assert cfg.input_cost_per_token == 0.0
        assert cfg.output_cost_per_token == 0.0


# ---------------------------------------------------------------------------
# TierDefaults / TIER_DEFAULTS
# ---------------------------------------------------------------------------


class TestTierDefaults:
    def test_quick_tier_exists(self):
        assert "quick" in TIER_DEFAULTS

    def test_standard_tier_exists(self):
        assert "standard" in TIER_DEFAULTS

    def test_deep_tier_exists(self):
        assert "deep" in TIER_DEFAULTS

    def test_quick_max_iterations(self):
        assert TIER_DEFAULTS["quick"].max_iterations == 2

    def test_standard_max_iterations(self):
        assert TIER_DEFAULTS["standard"].max_iterations == 5

    def test_deep_max_iterations(self):
        assert TIER_DEFAULTS["deep"].max_iterations == 10

    def test_quick_generator_is_anthropic_sonnet(self):
        gen = TIER_DEFAULTS["quick"].slots[ModelSlot.generator]
        assert gen.provider == "anthropic"
        assert gen.model == "claude-sonnet-4-6"

    def test_quick_subagent_is_gemini_flash(self):
        sub = TIER_DEFAULTS["quick"].slots[ModelSlot.subagent]
        assert sub.provider == "google-gla"
        assert sub.model == "gemini-3.1-flash-lite-preview"

    def test_quick_reviewer_is_gpt5_4_mini(self):
        rev = TIER_DEFAULTS["quick"].slots[ModelSlot.reviewer]
        assert rev.provider == "openai"
        assert rev.model == "gpt-5.4-mini"

    def test_quick_has_no_judge(self):
        assert ModelSlot.judge not in TIER_DEFAULTS["quick"].slots

    def test_standard_judge_is_gemini_pro(self):
        judge = TIER_DEFAULTS["standard"].slots[ModelSlot.judge]
        assert judge.provider == "google-gla"
        assert judge.model == "gemini-3.1-pro-preview"

    def test_standard_reviewer_is_gpt5_4_mini(self):
        rev = TIER_DEFAULTS["standard"].slots[ModelSlot.reviewer]
        assert rev.provider == "openai"
        assert rev.model == "gpt-5.4-mini"

    def test_deep_has_second_reviewer(self):
        assert TIER_DEFAULTS["deep"].second_reviewer is not None
        second = TIER_DEFAULTS["deep"].second_reviewer
        assert second.provider == "google-gla"
        assert second.model == "gemini-3.1-pro-preview"

    def test_quick_has_no_second_reviewer(self):
        assert TIER_DEFAULTS["quick"].second_reviewer is None

    def test_standard_has_no_second_reviewer(self):
        assert TIER_DEFAULTS["standard"].second_reviewer is None

    def test_quick_has_no_scope_override(self):
        assert TIER_DEFAULTS["quick"].scope_override is None

    def test_standard_has_no_scope_override(self):
        assert TIER_DEFAULTS["standard"].scope_override is None

    def test_deep_has_scope_override(self):
        override = TIER_DEFAULTS["deep"].scope_override
        assert override is not None
        assert override.provider == "anthropic"
        assert override.model == "claude-opus-4-6"

    def test_judge_has_thinking_config(self):
        """Standard and deep judge slots should have thinking enabled."""
        for tier_name in ("standard", "deep"):
            judge = TIER_DEFAULTS[tier_name].slots[ModelSlot.judge]
            assert judge.model_settings is not None
            assert "google_thinking_config" in judge.model_settings
            thinking_cfg = judge.model_settings["google_thinking_config"]
            assert thinking_cfg == {"thinking_level": "high"}

    def test_model_settings_default_is_none(self):
        """Slots without explicit model_settings default to None."""
        gen = TIER_DEFAULTS["quick"].slots[ModelSlot.generator]
        assert gen.model_settings is None

    def test_default_parallel_subagents(self):
        for tier in TIER_DEFAULTS.values():
            assert tier.max_parallel_subagents == 3


# ---------------------------------------------------------------------------
# ResearchSettings (env var loading)
# ---------------------------------------------------------------------------


class TestResearchSettings:
    def test_defaults(self):
        s = ResearchSettings()
        assert s.default_tier == "standard"
        assert s.default_cost_budget_usd == 0.10
        assert s.daily_cost_limit_usd == 10.0

    def test_loads_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_DEFAULT_TIER", "quick")
        monkeypatch.setenv("RESEARCH_DEFAULT_COST_BUDGET_USD", "0.50")
        monkeypatch.setenv("RESEARCH_DAILY_COST_LIMIT_USD", "5.0")
        monkeypatch.setenv("RESEARCH_MAX_PARALLEL_SUBAGENTS", "6")
        monkeypatch.setenv("RESEARCH_ENABLED_PROVIDERS", "arxiv,exa")

        s = ResearchSettings()
        assert s.default_tier == "quick"
        assert s.default_cost_budget_usd == 0.50
        assert s.daily_cost_limit_usd == 5.0
        assert s.max_parallel_subagents == 6
        assert s.enabled_providers == "arxiv,exa"

    def test_enabled_providers_default(self):
        s = ResearchSettings()
        assert s.enabled_providers == "brave,exa,arxiv,semantic_scholar"

    def test_sandbox_defaults(self):
        s = ResearchSettings()
        assert s.sandbox_enabled is False
        assert s.sandbox_backend is None

    def test_sandbox_from_env(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_SANDBOX_ENABLED", "true")
        monkeypatch.setenv("RESEARCH_SANDBOX_BACKEND", "docker")
        s = ResearchSettings()
        assert s.sandbox_enabled is True
        assert s.sandbox_backend == "docker"


# ---------------------------------------------------------------------------
# ResearchConfig
# ---------------------------------------------------------------------------


class TestResearchConfig:
    def test_for_tier_quick(self):
        cfg = ResearchConfig.for_tier("quick")
        assert cfg.tier == "quick"
        assert cfg.max_iterations == 2
        assert "generator" in cfg.slots
        assert "subagent" in cfg.slots
        assert "reviewer" in cfg.slots
        assert "judge" not in cfg.slots
        assert cfg.second_reviewer is None
        assert cfg.scope_override is None

    def test_for_tier_standard(self):
        cfg = ResearchConfig.for_tier("standard")
        assert cfg.tier == "standard"
        assert cfg.max_iterations == 5
        assert "judge" in cfg.slots
        assert cfg.second_reviewer is None
        assert cfg.scope_override is None

    def test_for_tier_deep(self):
        cfg = ResearchConfig.for_tier("deep")
        assert cfg.tier == "deep"
        assert cfg.max_iterations == 10
        assert cfg.second_reviewer is not None
        assert cfg.second_reviewer.model == "gemini-3.1-pro-preview"
        assert cfg.scope_override is not None
        assert cfg.scope_override.model_string == "anthropic:claude-opus-4-6"

    def test_for_tier_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tier"):
            ResearchConfig.for_tier("nonexistent")

    def test_config_is_frozen(self):
        cfg = ResearchConfig.for_tier("standard")
        with pytest.raises(ValidationError):
            cfg.tier = "quick"  # type: ignore[misc]

    def test_budget_uses_settings_cost(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_DEFAULT_COST_BUDGET_USD", "0.50")
        s = ResearchSettings()
        cfg = ResearchConfig.for_tier("standard", settings=s)
        assert cfg.budget.soft_budget_usd == 0.50

    def test_enabled_providers_parsed_from_comma_string(self):
        s = ResearchSettings()
        cfg = ResearchConfig.for_tier("standard", settings=s)
        assert cfg.enabled_providers == ["brave", "exa", "arxiv", "semantic_scholar"]

    def test_enabled_providers_custom(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_ENABLED_PROVIDERS", "arxiv,exa")
        s = ResearchSettings()
        cfg = ResearchConfig.for_tier("quick", settings=s)
        assert cfg.enabled_providers == ["arxiv", "exa"]

    def test_enabled_providers_handles_whitespace(self):
        s = ResearchSettings(enabled_providers=" arxiv , exa , brave ")
        cfg = ResearchConfig.for_tier("quick", settings=s)
        assert cfg.enabled_providers == ["arxiv", "exa", "brave"]

    def test_settings_overrides_applied(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LEDGER_WINDOW_ITERATIONS", "5")
        monkeypatch.setenv("RESEARCH_GROUNDING_MIN_RATIO", "0.8")
        monkeypatch.setenv("RESEARCH_MAX_SUPPLEMENTAL_LOOPS", "2")
        s = ResearchSettings()
        cfg = ResearchConfig.for_tier("standard", settings=s)
        assert cfg.ledger_window_iterations == 5
        assert cfg.grounding_min_ratio == 0.8
        assert cfg.max_supplemental_loops == 2

    def test_max_parallel_subagents_from_settings(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_MAX_PARALLEL_SUBAGENTS", "8")
        s = ResearchSettings()
        cfg = ResearchConfig.for_tier("quick", settings=s)
        assert cfg.max_parallel_subagents == 8

    def test_deep_tier_slot_models_match_spec(self):
        cfg = ResearchConfig.for_tier("deep")
        assert cfg.slots["generator"].model_string == "anthropic:claude-sonnet-4-6"
        assert (
            cfg.slots["subagent"].model_string
            == "google-gla:gemini-3.1-flash-lite-preview"
        )
        assert cfg.slots["reviewer"].model_string == "openai:gpt-5.4-mini"
        assert cfg.slots["judge"].model_string == "google-gla:gemini-3.1-pro-preview"

    def test_budget_not_frozen_inside_frozen_config(self):
        """BudgetConfig inside ResearchConfig should still be mutable.

        ResearchConfig is frozen, but BudgetConfig is a mutable sub-model.
        Pydantic's frozen=True prevents reassigning the field itself,
        but the BudgetConfig object's own fields remain mutable.
        """
        cfg = ResearchConfig.for_tier("standard")
        # The budget object's fields are mutable
        cfg.budget.spent_usd = 0.05
        assert cfg.budget.spent_usd == 0.05
