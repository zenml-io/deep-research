from deep_research.config import ModelPricing
from deep_research.models import IterationBudget


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, pricing: ModelPricing
) -> float:
    """Calculate USD cost from token counts and per-million pricing."""
    input_cost = pricing.input_per_million_usd * input_tokens / 1_000_000
    output_cost = pricing.output_per_million_usd * output_tokens / 1_000_000
    return input_cost + output_cost


def budget_from_agent_result(result: object, pricing: ModelPricing) -> IterationBudget:
    """Extract token usage from an agent result and return a costed IterationBudget."""
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
    """Combine two iteration budgets by summing their token counts and costs."""
    return IterationBudget(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        estimated_cost_usd=left.estimated_cost_usd + right.estimated_cost_usd,
    )
