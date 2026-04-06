import pytest
from pydantic import ValidationError

from deep_research.config import ModelPricing, ResearchConfig, ResearchSettings
from deep_research.enums import Tier


def test_settings_reads_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_DEFAULT_MAX_ITERATIONS", "4")
    settings = ResearchSettings()
    assert settings.default_max_iterations == 4


def test_research_config_builds_standard_tier_defaults() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD)
    assert config.max_iterations == 3
    assert config.cost_budget_usd == 0.10
    assert config.time_box_seconds == 600
    assert config.critique_enabled is False
    assert config.judge_enabled is False
    assert config.council_size == 1


def test_research_config_builds_deep_tier_flags_and_council() -> None:
    settings = ResearchSettings(council_size=5)
    config = ResearchConfig.for_tier(Tier.DEEP, settings=settings)

    assert config.critique_enabled is True
    assert config.judge_enabled is True
    assert config.council_size == 5


def test_invalid_settings_values_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        ResearchSettings(default_max_iterations=0)

    with pytest.raises(ValidationError):
        ResearchSettings(default_cost_budget_usd=-0.01)

    with pytest.raises(ValidationError):
        ResearchSettings(daily_cost_limit_usd=-1.0)

    with pytest.raises(ValidationError):
        ResearchSettings(convergence_epsilon=1.1)

    with pytest.raises(ValidationError):
        ResearchSettings(convergence_min_coverage=-0.1)

    with pytest.raises(ValidationError):
        ResearchSettings(max_tool_calls_per_cycle=0)

    with pytest.raises(ValidationError):
        ResearchSettings(tool_timeout_sec=0)

    with pytest.raises(ValidationError):
        ResearchSettings(source_quality_floor=1.1)

    with pytest.raises(ValidationError):
        ResearchSettings(council_size=0)

    with pytest.raises(ValidationError):
        ResearchSettings(council_cost_budget_usd=-1.0)

    with pytest.raises(ValidationError):
        ModelPricing(input_per_million_usd=-1.0)

    with pytest.raises(ValidationError):
        ModelPricing(output_per_million_usd=-1.0)

    with pytest.raises(ValidationError):
        ResearchConfig.for_tier(
            Tier.STANDARD,
            settings=ResearchSettings(default_max_iterations=0),
        )


def test_model_pricing_estimates_cost() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
    input_cost = pricing.input_per_million_usd * 1000 / 1_000_000
    output_cost = pricing.output_per_million_usd * 500 / 1_000_000
    assert round(input_cost + output_cost, 6) == 0.002
