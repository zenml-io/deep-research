"""Tests for budget accounting at the adapter boundary.

Pure Python — no Kitaru or PydanticAI imports needed.
"""

from __future__ import annotations

import warnings

import pytest

from research.config.budget import BudgetConfig
from research.flows.budget import (
    DEFAULT_MODEL_PRICING,
    BudgetTracker,
    HardBudgetExceededError,
    ModelPricing,
    UnknownModelCostError,
    UsageRecord,
    check_iteration_budget,
    lookup_pricing,
)


# ---------------------------------------------------------------------------
# ModelPricing / lookup
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_frozen(self):
        p = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)
        with pytest.raises(AttributeError):
            p.input_per_million_usd = 99.0  # type: ignore[misc]

    def test_default_table_has_expected_models(self):
        expected = {
            "google-gla:gemini-2.5-flash",
            "google-gla:gemini-2.5-pro",
            "openai:gpt-4o-mini",
            "openai:gpt-4o",
            "anthropic:claude-sonnet-4-20250514",
            "anthropic:claude-haiku-4-20250514",
        }
        assert set(DEFAULT_MODEL_PRICING.keys()) == expected


class TestLookupPricing:
    def test_exact_match(self):
        result = lookup_pricing("openai:gpt-4o-mini")
        assert result is not None
        assert result.input_per_million_usd == 0.15
        assert result.output_per_million_usd == 0.60

    def test_gateway_prefix_match(self):
        result = lookup_pricing("gateway/openai:gpt-4o-mini")
        assert result is not None
        assert result.input_per_million_usd == 0.15

    def test_gateway_anthropic_prefix_match(self):
        result = lookup_pricing("gateway/anthropic:claude-sonnet-4-20250514")
        assert result is not None
        assert result.input_per_million_usd == 3.00

    def test_unknown_model_returns_none(self):
        assert lookup_pricing("unknown:model-v1") is None

    def test_custom_pricing_table(self):
        custom = {"custom:my-model": ModelPricing(0.50, 1.00)}
        result = lookup_pricing("custom:my-model", pricing_table=custom)
        assert result is not None
        assert result.input_per_million_usd == 0.50

    def test_custom_table_does_not_fallback_to_defaults(self):
        custom = {"custom:my-model": ModelPricing(0.50, 1.00)}
        assert lookup_pricing("openai:gpt-4o-mini", pricing_table=custom) is None


# ---------------------------------------------------------------------------
# BudgetTracker — basic recording
# ---------------------------------------------------------------------------


class TestBudgetTrackerRecording:
    def test_record_usage_increments_spent(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget)

        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )

        # 1M input tokens * $0.15/M = $0.15
        assert budget.spent_usd == pytest.approx(0.15)

    def test_record_usage_adds_audit_record(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget)

        record = tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1000, output_tokens=500
        )

        assert isinstance(record, UsageRecord)
        assert record.model_name == "openai:gpt-4o-mini"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.pricing_known is True
        assert record.cost_usd > 0
        assert len(tracker.audit_trail) == 1
        assert tracker.audit_trail[0] is record

    def test_multiple_calls_accumulate(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )
        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )

        assert budget.spent_usd == pytest.approx(0.30)
        assert len(tracker.audit_trail) == 2

    def test_cost_calculation_both_input_and_output(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        # 1M input @ $0.15 + 1M output @ $0.60 = $0.75
        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000
        )

        assert budget.spent_usd == pytest.approx(0.75)

    def test_cost_calculation_small_token_counts(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        # 100 input @ $2.50/M + 200 output @ $10.00/M
        # = 0.00025 + 0.002 = 0.00225
        tracker.record_usage("openai:gpt-4o", input_tokens=100, output_tokens=200)

        assert budget.spent_usd == pytest.approx(0.00225)

    def test_gateway_model_priced_correctly(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        tracker.record_usage(
            "gateway/openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )

        assert budget.spent_usd == pytest.approx(0.15)

    def test_zero_tokens_zero_cost(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        record = tracker.record_usage("openai:gpt-4o", input_tokens=0, output_tokens=0)

        assert record.cost_usd == 0.0
        assert budget.spent_usd == 0.0


# ---------------------------------------------------------------------------
# BudgetTracker — unknown model handling
# ---------------------------------------------------------------------------


class TestBudgetTrackerUnknownModel:
    def test_unknown_model_priced_at_zero_with_warning(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget, strict_unknown_model_cost=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = tracker.record_usage(
                "unknown:model-v1", input_tokens=10_000, output_tokens=5_000
            )

        assert record.cost_usd == 0.0
        assert record.pricing_known is False
        assert budget.spent_usd == 0.0
        # Warning was emitted
        assert len(w) == 1
        assert "Unknown model" in str(w[0].message)
        assert "unknown:model-v1" in str(w[0].message)

    def test_unknown_model_still_recorded_in_audit(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget, strict_unknown_model_cost=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            tracker.record_usage(
                "unknown:model-v1", input_tokens=10_000, output_tokens=5_000
            )

        assert len(tracker.audit_trail) == 1
        assert tracker.audit_trail[0].model_name == "unknown:model-v1"
        assert tracker.audit_trail[0].input_tokens == 10_000
        assert tracker.audit_trail[0].output_tokens == 5_000

    def test_strict_unknown_model_raises(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget, strict_unknown_model_cost=True)

        with pytest.raises(UnknownModelCostError, match="unknown:model-v1"):
            tracker.record_usage(
                "unknown:model-v1", input_tokens=1000, output_tokens=500
            )

    def test_strict_unknown_does_not_increment_budget(self):
        budget = BudgetConfig(soft_budget_usd=1.0)
        tracker = BudgetTracker(budget=budget, strict_unknown_model_cost=True)

        with pytest.raises(UnknownModelCostError):
            tracker.record_usage(
                "unknown:model-v1", input_tokens=1000, output_tokens=500
            )

        assert budget.spent_usd == 0.0
        assert len(tracker.audit_trail) == 0


# ---------------------------------------------------------------------------
# BudgetTracker — hard budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetTrackerHardBudget:
    def test_hard_budget_raises_on_overshoot(self):
        budget = BudgetConfig(
            soft_budget_usd=10.0,
            hard_budget_usd=0.20,
        )
        tracker = BudgetTracker(budget=budget)

        # First call: 1M input @ $0.15 — under hard limit
        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )
        assert budget.spent_usd == pytest.approx(0.15)

        # Second call: another 1M input @ $0.15 → total $0.30, exceeds $0.20
        with pytest.raises(HardBudgetExceededError, match="Hard budget exceeded"):
            tracker.record_usage(
                "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
            )

    def test_hard_budget_cost_is_still_recorded_before_raising(self):
        budget = BudgetConfig(
            soft_budget_usd=10.0,
            hard_budget_usd=0.10,
        )
        tracker = BudgetTracker(budget=budget)

        # This single call costs $0.15, exceeding hard limit of $0.10
        with pytest.raises(HardBudgetExceededError):
            tracker.record_usage(
                "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
            )

        # Cost was recorded before the error
        assert budget.spent_usd == pytest.approx(0.15)
        assert len(tracker.audit_trail) == 1

    def test_no_hard_budget_never_raises(self):
        budget = BudgetConfig(
            soft_budget_usd=0.01,
            hard_budget_usd=None,
        )
        tracker = BudgetTracker(budget=budget)

        # Even large spend doesn't raise when hard_budget_usd is None
        for _ in range(100):
            tracker.record_usage(
                "openai:gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000
            )

        # Just verify it didn't raise
        assert budget.spent_usd > 0

    def test_hard_budget_exact_boundary(self):
        """Exact equality triggers hard budget exceeded."""
        budget = BudgetConfig(
            soft_budget_usd=10.0,
            hard_budget_usd=0.15,
        )
        tracker = BudgetTracker(budget=budget)

        # Exactly $0.15 — at the boundary, should raise
        with pytest.raises(HardBudgetExceededError):
            tracker.record_usage(
                "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
            )


# ---------------------------------------------------------------------------
# check_iteration_budget
# ---------------------------------------------------------------------------


class TestCheckIterationBudget:
    def test_not_exceeded(self):
        budget = BudgetConfig(soft_budget_usd=1.0, spent_usd=0.05)
        exceeded, reason = check_iteration_budget(budget)
        assert exceeded is False
        assert reason == ""

    def test_exceeded_at_limit(self):
        budget = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.10)
        exceeded, reason = check_iteration_budget(budget)
        assert exceeded is True
        assert "Soft budget exceeded" in reason
        assert "$0.10" in reason

    def test_exceeded_above_limit(self):
        budget = BudgetConfig(soft_budget_usd=0.10, spent_usd=0.25)
        exceeded, reason = check_iteration_budget(budget)
        assert exceeded is True
        assert "Soft budget exceeded" in reason

    def test_zero_budget_immediately_exceeded(self):
        budget = BudgetConfig(soft_budget_usd=0.0, spent_usd=0.0)
        exceeded, reason = check_iteration_budget(budget)
        assert exceeded is True


# ---------------------------------------------------------------------------
# Integration-style: tracker + iteration check together
# ---------------------------------------------------------------------------


class TestBudgetTrackerWithIterationCheck:
    def test_tracker_accumulates_then_iteration_check_catches(self):
        """Simulates the flow pattern: record usage in checkpoints,
        then check budget between iterations."""
        budget = BudgetConfig(soft_budget_usd=1.00)
        tracker = BudgetTracker(budget=budget)

        # Iteration 1: two cheap model calls
        # gpt-4o-mini: 100K in × $0.15/M + 50K out × $0.60/M = $0.015 + $0.03 = $0.045
        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=100_000, output_tokens=50_000
        )
        # gemini-flash: 100K in × $0.15/M + 50K out × $0.60/M = $0.015 + $0.03 = $0.045
        tracker.record_usage(
            "google-gla:gemini-2.5-flash", input_tokens=100_000, output_tokens=50_000
        )

        # Total after iteration 1: $0.09 — well under $1.00
        exceeded, _ = check_iteration_budget(budget)
        assert exceeded is False

        # Iteration 2: expensive call
        # gpt-4o: 1M in × $2.50/M + 500K out × $10.00/M = $2.50 + $5.00 = $7.50
        tracker.record_usage(
            "openai:gpt-4o", input_tokens=1_000_000, output_tokens=500_000
        )

        # Total: $0.09 + $7.50 = $7.59 — exceeds $1.00
        exceeded, reason = check_iteration_budget(budget)
        assert exceeded is True
        assert "Soft budget exceeded" in reason

    def test_audit_trail_tracks_all_calls(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget)

        models = [
            "openai:gpt-4o-mini",
            "google-gla:gemini-2.5-flash",
            "anthropic:claude-sonnet-4-20250514",
        ]
        for model in models:
            tracker.record_usage(model, input_tokens=1000, output_tokens=500)

        assert len(tracker.audit_trail) == 3
        recorded_models = [r.model_name for r in tracker.audit_trail]
        assert recorded_models == models
        assert all(r.pricing_known for r in tracker.audit_trail)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_custom_pricing_table(self):
        custom = {
            "custom:model": ModelPricing(
                input_per_million_usd=1.00,
                output_per_million_usd=2.00,
            ),
        }
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget, pricing_table=custom)

        # Known in custom table
        record = tracker.record_usage(
            "custom:model", input_tokens=1_000_000, output_tokens=1_000_000
        )
        assert record.pricing_known is True
        assert record.cost_usd == pytest.approx(3.00)

    def test_custom_pricing_table_unknown_default_model(self):
        """When using a custom table, default models are unknown."""
        custom = {"custom:model": ModelPricing(1.00, 2.00)}
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget, pricing_table=custom)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            record = tracker.record_usage(
                "openai:gpt-4o", input_tokens=1000, output_tokens=500
            )

        assert record.pricing_known is False
        assert len(w) == 1

    def test_mixed_known_and_unknown_models(self):
        budget = BudgetConfig(soft_budget_usd=10.0)
        tracker = BudgetTracker(budget=budget, strict_unknown_model_cost=False)

        # Known model
        tracker.record_usage(
            "openai:gpt-4o-mini", input_tokens=1_000_000, output_tokens=0
        )
        known_cost = budget.spent_usd

        # Unknown model — priced at $0
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            tracker.record_usage(
                "unknown:model", input_tokens=1_000_000, output_tokens=1_000_000
            )

        # Budget only increased from the known model call
        assert budget.spent_usd == pytest.approx(known_cost)
        assert len(tracker.audit_trail) == 2
