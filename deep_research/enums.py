from enum import Enum


class StopReason(str, Enum):
    CONVERGED = "converged"
    DIMINISHING_RETURNS = "diminishing_returns"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_EXHAUSTED = "time_exhausted"
    ACTIVE_TIME_EXHAUSTED = "active_time_exhausted"
    MAX_ITERATIONS = "max_iterations"
    LOOP_STALL = "loop_stall"
    SUPERVISOR_COMPLETE = "supervisor_complete"
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
    REPOSITORY = "repository"
    BLOG = "blog"
    FORUM = "forum"
    BENCHMARK = "benchmark"


class SourceGroup(str, Enum):
    PAPERS = "papers"
    WEB = "web"
    NEWS = "news"
    REPOS = "repos"
    SOCIAL = "social"
    DOCS = "docs"
    BLOGS = "blogs"
    BENCHMARKS = "benchmarks"
    FORUMS = "forums"


class DeliverableMode(str, Enum):
    RESEARCH_PACKAGE = "research_package"
    FINAL_REPORT = "final_report"
    COMPARISON_MEMO = "comparison_memo"
    RECOMMENDATION_BRIEF = "recommendation_brief"
    ANSWER_ONLY = "answer_only"


class PlanningMode(str, Enum):
    BROAD_SCAN = "broad_scan"
    DECISION_SUPPORT = "decision_support"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    DEEP_DIVE = "deep_dive"
