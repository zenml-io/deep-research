import pytest

from deep_research.config import ModelPricing
from deep_research.flow.costing import (
    budget_from_agent_result,
    estimate_cost_usd,
    merge_usage,
)
from deep_research.models import IterationBudget


def test_estimate_cost_usd_uses_input_and_output_pricing() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)

    assert estimate_cost_usd(1000, 500, pricing) == 0.002


def test_estimate_cost_usd_keeps_full_precision() -> None:
    pricing = ModelPricing(
        input_per_million_usd=1.111111, output_per_million_usd=2.222222
    )

    assert estimate_cost_usd(1, 1, pricing) == pytest.approx(0.000003333333)


def test_merge_usage_adds_all_token_fields() -> None:
    combined = merge_usage(
        IterationBudget(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            estimated_cost_usd=0.1,
        ),
        IterationBudget(
            input_tokens=4,
            output_tokens=5,
            total_tokens=9,
            estimated_cost_usd=0.2,
        ),
    )

    assert combined.total_tokens == 12
    assert combined.estimated_cost_usd == pytest.approx(0.3)


def test_merge_usage_keeps_full_precision() -> None:
    combined = merge_usage(
        IterationBudget(
            input_tokens=1,
            output_tokens=0,
            total_tokens=1,
            estimated_cost_usd=0.000001111111,
        ),
        IterationBudget(
            input_tokens=0,
            output_tokens=1,
            total_tokens=1,
            estimated_cost_usd=0.000002222222,
        ),
    )

    assert combined.total_tokens == 2
    assert combined.estimated_cost_usd == pytest.approx(0.000003333333)


def test_budget_from_agent_result_supports_callable_usage() -> None:
    class FakeUsage:
        prompt_tokens = 1000
        completion_tokens = 500
        total_tokens = 1500

    class FakeResult:
        def usage(self) -> FakeUsage:
            return FakeUsage()

    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)

    budget = budget_from_agent_result(FakeResult(), pricing)

    assert budget.total_tokens == 1500
    assert budget.estimated_cost_usd == 0.002


def test_budget_from_agent_result_supports_usage_attributes() -> None:
    class FakeUsage:
        input_tokens = 750
        output_tokens = 250
        total_tokens = 1000

    class FakeResult:
        usage = FakeUsage()

    pricing = ModelPricing(
        input_per_million_usd=1.111111,
        output_per_million_usd=2.222222,
    )

    budget = budget_from_agent_result(FakeResult(), pricing)

    assert budget.input_tokens == 750
    assert budget.output_tokens == 250
    assert budget.total_tokens == 1000
    assert budget.estimated_cost_usd == pytest.approx(0.00138888875)
