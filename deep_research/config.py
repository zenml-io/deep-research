from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_research.enums import Tier


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_million_usd: float = Field(default=0.0, ge=0.0)
    output_per_million_usd: float = Field(default=0.0, ge=0.0)


class TierConfig(BaseModel):
    max_iterations: int = Field(gt=0)
    cost_budget_usd: float = Field(ge=0.0)
    time_box_seconds: int = Field(gt=0)
    critique_enabled: bool = False
    judge_enabled: bool = False
    allows_council: bool = False
    requires_plan_approval: bool = True
    max_replans: int = 0


class ResearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RESEARCH_", extra="ignore")

    default_tier: Tier = Tier.STANDARD
    default_max_iterations: int = Field(default=3, gt=0)
    default_cost_budget_usd: float = Field(default=0.10, ge=0.0)
    daily_cost_limit_usd: float = Field(default=10.0, ge=0.0)
    convergence_epsilon: float = Field(default=0.05, ge=0.0, le=1.0)
    convergence_min_coverage: float = Field(default=0.60, ge=0.0, le=1.0)
    max_tool_calls_per_cycle: int = Field(default=5, gt=0)
    tool_timeout_sec: int = Field(default=20, gt=0)
    allow_supervisor_bash: bool = False
    source_quality_floor: float = Field(default=0.30, ge=0.0, le=1.0)
    council_size: int = Field(default=3, gt=0)
    council_cost_budget_usd: float = Field(default=2.0, ge=0.0)
    classifier_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    planner_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    supervisor_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    relevance_scorer_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    curator_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    writer_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    aggregator_model: str = "google-gla:gemini-3.1-flash-lite-preview"
    review_model: str = "anthropic:claude-sonnet-4-20250514"
    judge_model: str = "openai:gpt-4o-mini"
    coverage_scorer_model: str = "openai:gpt-4o-mini"
    use_curator_for_selection: bool = False
    supervisor_context_budget_chars: int = Field(default=18000, gt=0)
    relevance_context_budget_chars: int = Field(default=16000, gt=0)
    coverage_context_budget_chars: int = Field(default=12000, gt=0)
    writer_context_budget_chars: int = Field(default=40000, gt=0)
    context_snippet_budget_chars: int = Field(default=600, gt=0)
    brave_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RESEARCH_BRAVE_API_KEY", "BRAVE_API_KEY"),
    )
    exa_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("RESEARCH_EXA_API_KEY", "EXA_API_KEY"),
    )
    semantic_scholar_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "RESEARCH_SEMANTIC_SCHOLAR_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY",
        ),
    )
    enabled_providers_csv: str = Field(
        default="arxiv,exa",
        alias="enabled_providers",
        validation_alias=AliasChoices("RESEARCH_ENABLED_PROVIDERS", "enabled_providers"),
    )
    max_results_per_query: int = Field(default=10, gt=0)
    max_fetch_candidates_per_iteration: int = Field(default=5, gt=0)
    max_fetched_chars_per_candidate: int = Field(default=4000, gt=0)

    @property
    def enabled_providers(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in self.enabled_providers_csv.split(","):
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                result.append(cleaned)
                seen.add(cleaned)
        return result


class ResearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: Tier
    max_iterations: int
    cost_budget_usd: float
    time_box_seconds: int
    critique_enabled: bool = False
    judge_enabled: bool = False
    council_mode: bool = False
    council_size: int = 1
    require_plan_approval: bool = True
    convergence_epsilon: float = 0.05
    convergence_min_coverage: float = 0.60
    source_quality_floor: float = 0.30
    max_tool_calls_per_cycle: int
    tool_timeout_sec: int
    allow_supervisor_bash: bool = False
    classifier_model: str
    planner_model: str
    supervisor_model: str
    relevance_scorer_model: str
    curator_model: str
    writer_model: str
    aggregator_model: str
    review_model: str
    judge_model: str
    coverage_scorer_model: str
    use_curator_for_selection: bool = False
    supervisor_context_budget_chars: int = 18000
    relevance_context_budget_chars: int = 16000
    coverage_context_budget_chars: int = 12000
    writer_context_budget_chars: int = 40000
    context_snippet_budget_chars: int = 600
    supervisor_pricing: ModelPricing = Field(default_factory=ModelPricing)
    relevance_scorer_pricing: ModelPricing = Field(default_factory=ModelPricing)
    writer_pricing: ModelPricing = Field(default_factory=ModelPricing)
    review_pricing: ModelPricing = Field(default_factory=ModelPricing)
    judge_pricing: ModelPricing = Field(default_factory=ModelPricing)
    brave_api_key: str = ""
    exa_api_key: str = ""
    semantic_scholar_api_key: str = ""
    enabled_providers: list[str] = Field(default_factory=list)
    max_results_per_query: int = 10
    max_fetch_candidates_per_iteration: int = 5
    max_fetched_chars_per_candidate: int = 4000
    max_replans: int = 0

    @classmethod
    def for_tier(
        cls, tier: Tier, settings: ResearchSettings | None = None
    ) -> "ResearchConfig":
        """Build a ResearchConfig with defaults appropriate for the given tier."""
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
                max_replans=1,
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
            critique_enabled=base.critique_enabled,
            judge_enabled=base.judge_enabled,
            council_mode=False,
            council_size=settings.council_size if base.allows_council else 1,
            require_plan_approval=base.requires_plan_approval,
            convergence_epsilon=settings.convergence_epsilon,
            convergence_min_coverage=settings.convergence_min_coverage,
            source_quality_floor=settings.source_quality_floor,
            max_tool_calls_per_cycle=settings.max_tool_calls_per_cycle,
            tool_timeout_sec=settings.tool_timeout_sec,
            allow_supervisor_bash=settings.allow_supervisor_bash,
            classifier_model=settings.classifier_model,
            planner_model=settings.planner_model,
            supervisor_model=settings.supervisor_model,
            relevance_scorer_model=settings.relevance_scorer_model,
            curator_model=settings.curator_model,
            writer_model=settings.writer_model,
            aggregator_model=settings.aggregator_model,
            review_model=settings.review_model,
            judge_model=settings.judge_model,
            coverage_scorer_model=settings.coverage_scorer_model,
            use_curator_for_selection=(
                True if tier == Tier.DEEP else settings.use_curator_for_selection
            ),
            supervisor_context_budget_chars=settings.supervisor_context_budget_chars,
            relevance_context_budget_chars=settings.relevance_context_budget_chars,
            coverage_context_budget_chars=settings.coverage_context_budget_chars,
            writer_context_budget_chars=settings.writer_context_budget_chars,
            context_snippet_budget_chars=settings.context_snippet_budget_chars,
            supervisor_pricing=ModelPricing(),
            relevance_scorer_pricing=ModelPricing(),
            writer_pricing=ModelPricing(),
            review_pricing=ModelPricing(),
            judge_pricing=ModelPricing(),
            brave_api_key=settings.brave_api_key,
            exa_api_key=settings.exa_api_key,
            semantic_scholar_api_key=settings.semantic_scholar_api_key,
            enabled_providers=list(settings.enabled_providers),
            max_results_per_query=settings.max_results_per_query,
            max_fetch_candidates_per_iteration=settings.max_fetch_candidates_per_iteration,
            max_fetched_chars_per_candidate=settings.max_fetched_chars_per_candidate,
            max_replans=base.max_replans,
        )
