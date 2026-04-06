from deep_research.config import ModelPricing
from deep_research.flow.costing import (
    budget_from_agent_result,
    estimate_cost_usd,
    merge_usage,
)
from deep_research.models import IterationBudget


def test_estimate_cost_usd_uses_input_and_output_pricing() -> None:
    pricing = ModelPricing(input_per_million_usd=1.0, output_per_million_usd=2.0)

    assert round(estimate_cost_usd(1000, 500, pricing), 6) == 0.002


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
    assert combined.estimated_cost_usd == 0.3


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
    assert round(budget.estimated_cost_usd, 6) == 0.002
