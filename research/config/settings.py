"""Runtime settings and frozen research config for V2.

ResearchSettings: env-var-driven knobs (pydantic-settings).
ResearchConfig: frozen, assembled config for a single run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from research.config.budget import BudgetConfig
from research.config.defaults import TIER_DEFAULTS, TierDefaults
from research.config.slots import ModelSlot, ModelSlotConfig


class ResearchSettings(BaseSettings):
    """Operator-tunable knobs, loaded from RESEARCH_* env vars."""

    model_config = SettingsConfigDict(env_prefix="RESEARCH_", extra="ignore")

    default_tier: str = "standard"
    default_cost_budget_usd: float = 0.10
    daily_cost_limit_usd: float = 10.0
    ledger_window_iterations: int = 3
    grounding_min_ratio: float = 0.7
    max_supplemental_loops: int = 1
    wait_timeout_seconds: int = 3600
    allow_unfinalized_package: bool = False
    strict_unknown_model_cost: bool = False
    sandbox_enabled: bool = False
    sandbox_backend: str | None = None
    max_parallel_subagents: int = 3
    enabled_providers: str = "brave,exa,tavily,arxiv,semantic_scholar"


class ResearchConfig(BaseModel):
    """Frozen, fully-resolved config for a single research run.

    Created via the `for_tier()` classmethod which merges tier defaults
    with operator settings.
    """

    model_config = ConfigDict(frozen=True)

    tier: str
    budget: BudgetConfig
    slots: dict[str, ModelSlotConfig]
    second_reviewer: ModelSlotConfig | None = None
    scope_override: ModelSlotConfig | None = None
    max_iterations: int
    max_parallel_subagents: int
    ledger_window_iterations: int
    grounding_min_ratio: float
    max_supplemental_loops: int
    wait_timeout_seconds: int
    allow_unfinalized_package: bool
    strict_unknown_model_cost: bool
    sandbox_enabled: bool
    sandbox_backend: str | None
    enabled_providers: list[str]

    @classmethod
    def for_tier(
        cls,
        tier: str,
        settings: ResearchSettings | None = None,
    ) -> ResearchConfig:
        """Build a frozen ResearchConfig from tier defaults + settings.

        This is the canonical factory — every run should go through here.
        """
        settings = settings or ResearchSettings()

        if tier not in TIER_DEFAULTS:
            raise ValueError(
                f"Unknown tier {tier!r}. Valid tiers: {sorted(TIER_DEFAULTS)}"
            )

        defaults: TierDefaults = TIER_DEFAULTS[tier]

        # Convert ModelSlot enum keys to string keys for the frozen dict
        slots: dict[str, ModelSlotConfig] = {
            slot.value: config for slot, config in defaults.slots.items()
        }

        # Parse comma-separated providers into a list
        enabled_providers = [
            p.strip() for p in settings.enabled_providers.split(",") if p.strip()
        ]

        budget = BudgetConfig(
            soft_budget_usd=settings.default_cost_budget_usd,
        )

        return cls(
            tier=tier,
            budget=budget,
            slots=slots,
            second_reviewer=defaults.second_reviewer,
            scope_override=defaults.scope_override,
            max_iterations=defaults.max_iterations,
            max_parallel_subagents=settings.max_parallel_subagents,
            ledger_window_iterations=settings.ledger_window_iterations,
            grounding_min_ratio=settings.grounding_min_ratio,
            max_supplemental_loops=settings.max_supplemental_loops,
            wait_timeout_seconds=settings.wait_timeout_seconds,
            allow_unfinalized_package=settings.allow_unfinalized_package,
            strict_unknown_model_cost=settings.strict_unknown_model_cost,
            sandbox_enabled=settings.sandbox_enabled,
            sandbox_backend=settings.sandbox_backend,
            enabled_providers=enabled_providers,
        )
