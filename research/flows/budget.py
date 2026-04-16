"""Budget accounting at the PydanticAI-to-Kitaru adapter boundary.

Centralised cost tracking: the wrapped agent adapter calls
``BudgetTracker.record_usage()`` after every model response.  The tracker
looks up per-model pricing, increments ``BudgetConfig.spent_usd``, and
optionally enforces a hard ceiling.

Design decisions
~~~~~~~~~~~~~~~~
* Budget is checked **between** iterations, not mid-checkpoint.
  Draft/critique/finalize/assemble run unconditionally so the engine
  always produces a deliverable (some overshoot accepted).
* ``hard_budget_usd`` (null by default) is the exception — it causes
  the adapter to fail on *any* call that pushes past the ceiling.
* Unknown models are priced at $0 with a loud warning.  If
  ``strict_unknown_model_cost=True``, a hard error is raised instead.
"""

from __future__ import annotations

import logging
import warnings
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from research.config.budget import BudgetConfig

logger = logging.getLogger(__name__)

# Run-scoped active tracker. Set by the flow before running checkpoints so
# budget-aware agents can record usage without callers threading a tracker
# through every call site.
_active_tracker: ContextVar[BudgetTracker | None] = ContextVar(
    "research_active_budget_tracker",
    default=None,
)


def set_active_tracker(tracker: BudgetTracker | None) -> Token[BudgetTracker | None]:
    """Set (or clear) the active budget tracker for the current run context."""
    return _active_tracker.set(tracker)


def reset_active_tracker(token: Token[BudgetTracker | None]) -> None:
    """Restore the previously-active tracker for the current run context."""
    _active_tracker.reset(token)


def get_active_tracker() -> BudgetTracker | None:
    """Return the currently-active tracker for this run context, or ``None``."""
    return _active_tracker.get()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownModelCostError(Exception):
    """Raised when ``strict_unknown_model_cost=True`` and the model has no
    pricing entry."""


class HardBudgetExceededError(Exception):
    """Raised when ``hard_budget_usd`` is set and spending exceeds it."""


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """Per-model rate card (USD per million tokens)."""

    input_per_million_usd: float
    output_per_million_usd: float


# Approximate 2025 rates.  Keys are PydanticAI model strings.
# Prefix matching is used so ``gateway/openai:gpt-4o-mini`` resolves to
# the ``openai:gpt-4o-mini`` entry.
DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    # ── Active tier models (defaults.py) ─────────────────────────────
    "anthropic:claude-sonnet-4-6": ModelPricing(
        input_per_million_usd=3.00,
        output_per_million_usd=15.00,
    ),
    "anthropic:claude-opus-4-6": ModelPricing(
        input_per_million_usd=15.00,
        output_per_million_usd=75.00,
    ),
    "google-gla:gemini-3.1-flash-lite-preview": ModelPricing(
        input_per_million_usd=0.15,
        output_per_million_usd=0.60,
    ),
    "google-gla:gemini-3.1-pro-preview": ModelPricing(
        input_per_million_usd=1.25,
        output_per_million_usd=10.00,
    ),
    "openai:gpt-5.4-mini": ModelPricing(
        input_per_million_usd=0.15,
        output_per_million_usd=0.60,
    ),
    # ── Legacy / fallback models ─────────────────────────────────────
    "google-gla:gemini-2.5-flash": ModelPricing(
        input_per_million_usd=0.15,
        output_per_million_usd=0.60,
    ),
    "google-gla:gemini-2.5-pro": ModelPricing(
        input_per_million_usd=1.25,
        output_per_million_usd=10.00,
    ),
    "openai:gpt-4o-mini": ModelPricing(
        input_per_million_usd=0.15,
        output_per_million_usd=0.60,
    ),
    "openai:gpt-4o": ModelPricing(
        input_per_million_usd=2.50,
        output_per_million_usd=10.00,
    ),
    "anthropic:claude-sonnet-4-20250514": ModelPricing(
        input_per_million_usd=3.00,
        output_per_million_usd=15.00,
    ),
    "anthropic:claude-haiku-4-20250514": ModelPricing(
        input_per_million_usd=0.80,
        output_per_million_usd=4.00,
    ),
}


def lookup_pricing(
    model_name: str,
    pricing_table: dict[str, ModelPricing] | None = None,
) -> ModelPricing | None:
    """Look up pricing for *model_name*, supporting prefix matching.

    Exact match is tried first.  If that fails, each key is tested as a
    suffix of *model_name* (e.g. ``gateway/openai:gpt-4o-mini`` ends with
    ``openai:gpt-4o-mini``).  Returns ``None`` when no match is found.
    """
    table = pricing_table or DEFAULT_MODEL_PRICING

    # 1. Exact match
    if model_name in table:
        return table[model_name]

    # 2. Suffix / prefix match — model_name may have a gateway prefix
    for key, pricing in table.items():
        if model_name.endswith(key):
            return pricing

    return None


# ---------------------------------------------------------------------------
# Usage record (audit trail entry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageRecord:
    """Single model-call cost record kept in the audit trail."""

    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    pricing_known: bool


# ---------------------------------------------------------------------------
# BudgetTracker
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Accumulates token costs against a ``BudgetConfig``.

    Instantiated once per run.  The wrapped PydanticAI adapter calls
    ``record_usage()`` after every model response.

    Parameters
    ----------
    budget:
        Mutable budget config for the current run.
    strict_unknown_model_cost:
        When ``True``, raise ``UnknownModelCostError`` for models not
        found in the pricing table instead of pricing at $0.
    pricing_table:
        Custom pricing table.  Defaults to ``DEFAULT_MODEL_PRICING``.
    """

    budget: BudgetConfig
    strict_unknown_model_cost: bool = False
    pricing_table: dict[str, ModelPricing] = field(
        default_factory=lambda: dict(DEFAULT_MODEL_PRICING)
    )
    audit_trail: list[UsageRecord] = field(default_factory=list)

    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> UsageRecord:
        """Record token usage from a model call and update spent_usd.

        Returns the ``UsageRecord`` for the call.

        Raises
        ------
        UnknownModelCostError
            If the model is not in the pricing table and
            ``strict_unknown_model_cost`` is ``True``.
        HardBudgetExceededError
            If ``hard_budget_usd`` is set and the cumulative spend
            exceeds it after this call.
        """
        pricing = lookup_pricing(model_name, self.pricing_table)

        if pricing is None:
            if self.strict_unknown_model_cost:
                raise UnknownModelCostError(
                    f"No pricing entry for model {model_name!r} and "
                    f"strict_unknown_model_cost is enabled."
                )
            # Price at $0 but warn loudly
            msg = (
                f"Unknown model {model_name!r} — pricing at $0.  "
                f"Add an entry to the pricing table to track cost accurately."
            )
            warnings.warn(msg, stacklevel=2)
            logger.warning(msg)
            cost_usd = 0.0
            pricing_known = False
        else:
            cost_usd = (
                pricing.input_per_million_usd * input_tokens / 1_000_000
                + pricing.output_per_million_usd * output_tokens / 1_000_000
            )
            pricing_known = True

        record = UsageRecord(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            pricing_known=pricing_known,
        )
        self.audit_trail.append(record)
        self.budget.spent_usd += cost_usd

        # Hard budget enforcement — checked after every call
        if self.budget.is_hard_exceeded():
            raise HardBudgetExceededError(
                f"Hard budget exceeded: spent ${self.budget.spent_usd:.6f} "
                f">= hard limit ${self.budget.hard_budget_usd:.6f}"
            )

        return record


# ---------------------------------------------------------------------------
# Iteration-boundary budget check
# ---------------------------------------------------------------------------


def check_iteration_budget(budget: BudgetConfig) -> tuple[bool, str]:
    """Check whether the soft budget is exceeded between iterations.

    Called by the flow between iteration loops.  Does **not** raise —
    returns a ``(exceeded, reason)`` tuple so the flow can decide whether
    to stop iterating.

    Returns
    -------
    tuple[bool, str]
        ``(True, reason)`` if the soft budget is exceeded,
        ``(False, "")`` otherwise.
    """
    if budget.is_exceeded():
        return (
            True,
            f"Soft budget exceeded: spent ${budget.spent_usd:.6f} "
            f">= limit ${budget.soft_budget_usd:.6f}",
        )
    return (False, "")
