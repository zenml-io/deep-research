from enum import Enum


class StopReason(str, Enum):
    CONVERGED = "converged"
    DIMINISHING_RETURNS = "diminishing_returns"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    MAX_ITERATIONS = "max_iterations"
    LOOP_STALL = "loop_stall"
    CANCELLED = "cancelled"


class Tier(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    CUSTOM = "custom"


class SourceKind(str, Enum):
    PAPER = "paper"
    DOCS = "docs"
    WEB = "web"
    DATASET = "dataset"
