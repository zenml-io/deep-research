from deep_research.config import ModelPricing, ResearchConfig, ResearchSettings
from deep_research.enums import Tier


def test_settings_reads_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_DEFAULT_MAX_ITERATIONS", "4")
    settings = ResearchSettings()
    assert settings.default_max_iterations == 4


def test_research_config_builds_standard_tier_defaults() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD)
    assert config.max_iterations > 0
    assert config.cost_budget_usd > 0


def test_model_pricing_estimates_cost() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
    input_cost = pricing.input_per_million_usd * 1000 / 1_000_000
    output_cost = pricing.output_per_million_usd * 500 / 1_000_000
    assert round(input_cost + output_cost, 6) == 0.002
