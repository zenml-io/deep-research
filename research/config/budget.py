"""Budget tracking for a research run.

BudgetConfig is intentionally NOT frozen — spent_usd must be mutable
during a run as costs accumulate.
"""

from pydantic import BaseModel


class BudgetConfig(BaseModel):
    """Tracks cost budget for a single research run.

    soft_budget_usd: advisory limit — triggers convergence checks.
    hard_budget_usd: absolute ceiling — run stops immediately.
    spent_usd: running total, mutated during execution.
    """

    soft_budget_usd: float = 0.10
    hard_budget_usd: float | None = None
    spent_usd: float = 0.0

    def is_exceeded(self) -> bool:
        """True when spent has reached or exceeded the soft budget."""
        return self.spent_usd >= self.soft_budget_usd

    def is_hard_exceeded(self) -> bool:
        """True when a hard budget is set and spent has reached or exceeded it."""
        if self.hard_budget_usd is None:
            return False
        return self.spent_usd >= self.hard_budget_usd
