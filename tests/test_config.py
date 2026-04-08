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


def test_research_config_carries_source_quality_floor() -> None:
    settings = ResearchSettings(source_quality_floor=0.55)

    config = ResearchConfig.for_tier(Tier.STANDARD, settings=settings)

    assert config.source_quality_floor == 0.55


def test_research_config_carries_review_and_judge_models() -> None:
    settings = ResearchSettings(
        review_model="anthropic/reviewer",
        judge_model="openai/judge",
    )

    config = ResearchConfig.for_tier(Tier.DEEP, settings=settings)

    assert config.review_model == "anthropic/reviewer"
    assert config.judge_model == "openai/judge"


def test_research_config_carries_supervisor_tool_settings() -> None:
    settings = ResearchSettings(max_tool_calls_per_cycle=7, tool_timeout_sec=45)

    config = ResearchConfig.for_tier(Tier.STANDARD, settings=settings)

    assert config.max_tool_calls_per_cycle == 7
    assert config.tool_timeout_sec == 45


def test_research_config_carries_provider_and_fetch_settings() -> None:
    settings = ResearchSettings(
        enabled_providers=["arxiv", "semantic_scholar"],
        max_results_per_query=12,
        max_fetch_candidates_per_iteration=4,
        max_fetched_chars_per_candidate=3500,
    )

    config = ResearchConfig.for_tier(Tier.STANDARD, settings=settings)

    assert config.enabled_providers == ["arxiv", "semantic_scholar"]
    assert config.max_results_per_query == 12
    assert config.max_fetch_candidates_per_iteration == 4
    assert config.max_fetched_chars_per_candidate == 3500


def test_research_config_normalizes_enabled_providers() -> None:
    settings = ResearchSettings(
        enabled_providers=[" arxiv ", "semantic_scholar", "", "arxiv", "  "],
    )

    config = ResearchConfig.for_tier(Tier.STANDARD, settings=settings)

    assert config.enabled_providers == ["arxiv", "semantic_scholar"]


def test_settings_accept_provider_api_keys_from_prefixed_or_naked_env_vars(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RESEARCH_BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_EXA_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCH_SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "brave-secret")
    monkeypatch.setenv("EXA_API_KEY", "exa-secret")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "semantic-secret")

    settings = ResearchSettings()

    assert settings.brave_api_key == "brave-secret"
    assert settings.exa_api_key == "exa-secret"
    assert settings.semantic_scholar_api_key == "semantic-secret"


def test_research_config_uses_default_supervisor_tool_settings() -> None:
    settings = ResearchSettings()
    config = ResearchConfig.for_tier(Tier.STANDARD, settings=settings)

    assert config.max_tool_calls_per_cycle == settings.max_tool_calls_per_cycle
    assert config.tool_timeout_sec == settings.tool_timeout_sec


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
