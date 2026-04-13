from typing import Union

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


class ConvergenceConfig(BaseModel):
    token_budget: int = Field(default=200_000, ge=0)
    coverage_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    strong_coverage_shortcut: float = Field(default=0.92, ge=0.0, le=1.0)
    marginal_gain_threshold: float = Field(default=0.05, ge=0.0)
    max_stall_count: int = Field(default=3, ge=0)
    target_considered_resources: int = Field(default=30, ge=0)
    min_considered_floor: int = Field(default=15, ge=0)
    min_selected_for_stop: int = Field(default=12, ge=0)


class SelectionPolicyConfig(BaseModel):
    relevance_weight: float = Field(default=0.40, ge=0.0)
    authority_weight: float = Field(default=0.25, ge=0.0)
    recency_weight: float = Field(default=0.20, ge=0.0)
    novelty_weight: float = Field(default=0.15, ge=0.0)
    mmr_relevance_lambda: float = Field(default=0.50, ge=0.0)
    mmr_diversity_lambda: float = Field(default=0.35, ge=0.0)
    mmr_source_type_lambda: float = Field(default=0.15, ge=0.0)
    max_paper_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    # Wave 4.5: synthesis-to-unsupported-claims feedback loop. Hard cap on the
    # number of feedback iterations per run (deep tier only). 0 disables.
    feedback_loop_max_iterations: int = Field(default=2, ge=0)


class WebSearchPolicyConfig(BaseModel):
    default_source_group_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "web": 1.0,
            "docs": 1.0,
            "repos": 0.95,
            "benchmarks": 0.9,
            "blogs": 0.8,
            "papers": 0.45,
        }
    )
    domain_recency_half_life_days: dict[str, int] = Field(
        default_factory=lambda: {
            "tooling": 90,
            "architecture": 365,
            "algorithm": 1825,
            "current_events": 14,
        }
    )
    max_results_per_query: int = Field(default=10, gt=0)
    max_fetch_candidates_per_iteration: int = Field(default=5, gt=0)
    max_fetched_chars_per_candidate: int = Field(default=4000, gt=0)


class ActiveTimeConfig(BaseModel):
    quick_limit_seconds: int = Field(default=60, gt=0)
    standard_limit_seconds: int = Field(default=300, gt=0)
    deep_limit_seconds: int = Field(default=900, gt=0)
    custom_limit_seconds: int = Field(default=300, gt=0)


class ClaimExtractionConfig(BaseModel):
    enabled_for_quick: bool = False
    enabled_for_standard: bool = False
    enabled_for_deep: bool = True
    enabled_for_custom: bool = False


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
    claim_extractor_model: str = "openai:gpt-4o-mini"
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
    # Web-first default: brave/exa lead (engineering prompts favor live web/docs),
    # with arxiv/semantic_scholar as academic corroboration. Brave and Exa require
    # API keys and no-op gracefully via provider.is_available() when keys are absent
    # (see deep_research.providers.search.ProviderRegistry.active_providers).
    enabled_providers_raw: Union[str, list[str]] = Field(
        default="brave,exa,arxiv,semantic_scholar",
        alias="enabled_providers",
        validation_alias=AliasChoices("RESEARCH_ENABLED_PROVIDERS", "enabled_providers"),
    )
    max_results_per_query: int = Field(default=10, gt=0)
    max_fetch_candidates_per_iteration: int = Field(default=5, gt=0)
    max_fetched_chars_per_candidate: int = Field(default=4000, gt=0)
    convergence_token_budget: int = Field(default=200_000, ge=0)
    convergence_strong_coverage_shortcut: float = Field(
        default=0.92, ge=0.0, le=1.0
    )
    convergence_marginal_gain_threshold: float = Field(default=0.05, ge=0.0)
    convergence_max_stall_count: int = Field(default=3, ge=0)
    convergence_target_considered_resources: int = Field(default=200, ge=0)
    convergence_min_considered_floor: int = Field(default=100, ge=0)
    convergence_min_selected_for_stop: int = Field(default=12, ge=0)
    selection_relevance_weight: float = Field(default=0.40, ge=0.0)
    selection_authority_weight: float = Field(default=0.25, ge=0.0)
    selection_recency_weight: float = Field(default=0.20, ge=0.0)
    selection_novelty_weight: float = Field(default=0.15, ge=0.0)
    selection_mmr_relevance_lambda: float = Field(default=0.50, ge=0.0)
    selection_mmr_diversity_lambda: float = Field(default=0.35, ge=0.0)
    selection_mmr_source_type_lambda: float = Field(default=0.15, ge=0.0)
    selection_max_paper_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    active_time_quick_limit_seconds: int = Field(default=60, gt=0)
    active_time_standard_limit_seconds: int = Field(default=300, gt=0)
    active_time_deep_limit_seconds: int = Field(default=900, gt=0)
    active_time_custom_limit_seconds: int = Field(default=300, gt=0)
    claim_extraction_enabled_for_quick: bool = False
    claim_extraction_enabled_for_standard: bool = False
    claim_extraction_enabled_for_deep: bool = True
    claim_extraction_enabled_for_custom: bool = False

    @property
    def enabled_providers(self) -> list[str]:
        raw = self.enabled_providers_raw
        if isinstance(raw, str):
            items = raw.split(",")
        else:
            items = list(raw)
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            cleaned = item.strip() if isinstance(item, str) else str(item).strip()
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
    claim_extractor_model: str
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
    claim_extractor_pricing: ModelPricing = Field(default_factory=ModelPricing)
    brave_api_key: str = ""
    exa_api_key: str = ""
    semantic_scholar_api_key: str = ""
    enabled_providers: list[str] = Field(default_factory=list)
    max_results_per_query: int = 10
    max_fetch_candidates_per_iteration: int = 5
    max_fetched_chars_per_candidate: int = 4000
    max_replans: int = 0
    convergence: ConvergenceConfig = Field(default_factory=ConvergenceConfig)
    selection_policy: SelectionPolicyConfig = Field(
        default_factory=SelectionPolicyConfig
    )
    web_search_policy: WebSearchPolicyConfig = Field(
        default_factory=WebSearchPolicyConfig
    )
    active_time: ActiveTimeConfig = Field(default_factory=ActiveTimeConfig)
    claim_extraction: ClaimExtractionConfig = Field(
        default_factory=ClaimExtractionConfig
    )
    claim_extraction_enabled: bool = False

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
        convergence = ConvergenceConfig(
            token_budget=settings.convergence_token_budget,
            coverage_threshold=settings.convergence_min_coverage,
            strong_coverage_shortcut=settings.convergence_strong_coverage_shortcut,
            marginal_gain_threshold=settings.convergence_marginal_gain_threshold,
            max_stall_count=settings.convergence_max_stall_count,
            target_considered_resources=settings.convergence_target_considered_resources,
            min_considered_floor=settings.convergence_min_considered_floor,
            min_selected_for_stop=settings.convergence_min_selected_for_stop,
        )
        selection_policy = SelectionPolicyConfig(
            relevance_weight=settings.selection_relevance_weight,
            authority_weight=settings.selection_authority_weight,
            recency_weight=settings.selection_recency_weight,
            novelty_weight=settings.selection_novelty_weight,
            mmr_relevance_lambda=settings.selection_mmr_relevance_lambda,
            mmr_diversity_lambda=settings.selection_mmr_diversity_lambda,
            mmr_source_type_lambda=settings.selection_mmr_source_type_lambda,
            max_paper_ratio=settings.selection_max_paper_ratio,
        )
        web_search_policy = WebSearchPolicyConfig(
            max_results_per_query=settings.max_results_per_query,
            max_fetch_candidates_per_iteration=settings.max_fetch_candidates_per_iteration,
            max_fetched_chars_per_candidate=settings.max_fetched_chars_per_candidate,
        )
        active_time = ActiveTimeConfig(
            quick_limit_seconds=settings.active_time_quick_limit_seconds,
            standard_limit_seconds=settings.active_time_standard_limit_seconds,
            deep_limit_seconds=settings.active_time_deep_limit_seconds,
            custom_limit_seconds=settings.active_time_custom_limit_seconds,
        )
        claim_extraction = ClaimExtractionConfig(
            enabled_for_quick=settings.claim_extraction_enabled_for_quick,
            enabled_for_standard=settings.claim_extraction_enabled_for_standard,
            enabled_for_deep=settings.claim_extraction_enabled_for_deep,
            enabled_for_custom=settings.claim_extraction_enabled_for_custom,
        )
        claim_extraction_enabled = {
            Tier.QUICK: claim_extraction.enabled_for_quick,
            Tier.STANDARD: claim_extraction.enabled_for_standard,
            Tier.DEEP: claim_extraction.enabled_for_deep,
            Tier.CUSTOM: claim_extraction.enabled_for_custom,
        }[tier]
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
            claim_extractor_model=settings.claim_extractor_model,
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
            claim_extractor_pricing=ModelPricing(),
            brave_api_key=settings.brave_api_key,
            exa_api_key=settings.exa_api_key,
            semantic_scholar_api_key=settings.semantic_scholar_api_key,
            enabled_providers=list(settings.enabled_providers),
            max_results_per_query=settings.max_results_per_query,
            max_fetch_candidates_per_iteration=settings.max_fetch_candidates_per_iteration,
            max_fetched_chars_per_candidate=settings.max_fetched_chars_per_candidate,
            max_replans=base.max_replans,
            convergence=convergence,
            selection_policy=selection_policy,
            web_search_policy=web_search_policy,
            active_time=active_time,
            claim_extraction=claim_extraction,
            claim_extraction_enabled=claim_extraction_enabled,
        )
