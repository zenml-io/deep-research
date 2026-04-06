from deep_research.config import ModelPricing
from deep_research.models import IterationBudget


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, pricing: ModelPricing
) -> float:
    input_cost = pricing.input_per_million_usd * input_tokens / 1_000_000
    output_cost = pricing.output_per_million_usd * output_tokens / 1_000_000
    return round(input_cost + output_cost, 6)


def budget_from_agent_result(result: object, pricing: ModelPricing) -> IterationBudget:
    usage_attr = getattr(result, "usage", None)
    usage = usage_attr() if callable(usage_attr) else usage_attr
    input_tokens = int(
        getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0
    )
    total_tokens = int(
        getattr(usage, "total_tokens", input_tokens + output_tokens)
        or (input_tokens + output_tokens)
    )
    return IterationBudget(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, pricing),
    )


def merge_usage(left: IterationBudget, right: IterationBudget) -> IterationBudget:
    return IterationBudget(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost_usd=round(left.estimated_cost_usd + right.estimated_cost_usd, 6),
    )


def is_budget_exhausted(spent_usd: float, limit_usd: float) -> bool:
    return spent_usd >= limit_usd
