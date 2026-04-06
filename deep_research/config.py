from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.enums import Tier


class ModelPricing(BaseModel):
    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0


class TierConfig(BaseModel):
    max_iterations: int
    cost_budget_usd: float
    time_box_seconds: int
    critique_enabled: bool = False
    judge_enabled: bool = False
    allows_council: bool = False
    requires_plan_approval: bool = True


class ResearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_", extra="ignore")

    default_tier: Tier = Tier.STANDARD
    default_max_iterations: int = 3
    default_cost_budget_usd: float = 0.10
    daily_cost_limit_usd: float = 10.0
    convergence_epsilon: float = 0.05
    convergence_min_coverage: float = 0.60
    max_tool_calls_per_cycle: int = 5
    tool_timeout_sec: int = 20
    source_quality_floor: float = 0.30
    council_size: int = 3
    council_cost_budget_usd: float = 2.0
    classifier_model: str = "gemini/gemini-2.0-flash-lite"
    planner_model: str = "gemini/gemini-2.5-flash"
    supervisor_model: str = "gemini/gemini-2.5-flash"
    relevance_scorer_model: str = "gemini/gemini-2.5-flash"
    curator_model: str = "gemini/gemini-2.0-flash-lite"
    writer_model: str = "gemini/gemini-2.5-flash"
    aggregator_model: str = "openai/gpt-4o-mini"


class ResearchConfig(BaseModel):
    tier: Tier
    max_iterations: int
    cost_budget_usd: float
    time_box_seconds: int
    council_mode: bool = False
    council_size: int = 1
    require_plan_approval: bool = True
    classifier_model: str
    planner_model: str
    supervisor_model: str
    relevance_scorer_model: str
    curator_model: str
    writer_model: str
    aggregator_model: str

    @classmethod
    def for_tier(
        cls, tier: Tier, settings: ResearchSettings | None = None
    ) -> "ResearchConfig":
        settings = settings or ResearchSettings()
        mapping = {
            Tier.QUICK: TierConfig(
                max_iterations=2,
                cost_budget_usd=0.05,
                time_box_seconds=120,
            ),
            Tier.STANDARD: TierConfig(
                max_iterations=settings.default_max_iterations,
                cost_budget_usd=settings.default_cost_budget_usd,
                time_box_seconds=600,
            ),
            Tier.DEEP: TierConfig(
                max_iterations=6,
                cost_budget_usd=1.0,
                time_box_seconds=1800,
                critique_enabled=True,
                judge_enabled=True,
                allows_council=True,
            ),
            Tier.CUSTOM: TierConfig(
                max_iterations=settings.default_max_iterations,
                cost_budget_usd=settings.default_cost_budget_usd,
                time_box_seconds=600,
                allows_council=True,
            ),
        }
        base = mapping[tier]
        return cls(
            tier=tier,
            max_iterations=base.max_iterations,
            cost_budget_usd=base.cost_budget_usd,
            time_box_seconds=base.time_box_seconds,
            council_mode=False,
            council_size=settings.council_size if base.allows_council else 1,
            require_plan_approval=base.requires_plan_approval,
            classifier_model=settings.classifier_model,
            planner_model=settings.planner_model,
            supervisor_model=settings.supervisor_model,
            relevance_scorer_model=settings.relevance_scorer_model,
            curator_model=settings.curator_model,
            writer_model=settings.writer_model,
            aggregator_model=settings.aggregator_model,
        )
