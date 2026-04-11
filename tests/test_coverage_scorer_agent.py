"""Test coverage scorer agent produces valid CoverageScore output."""

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai.models.test import TestModel  # noqa: E402

from deep_research.agents.coverage_scorer import build_coverage_scorer_agent  # noqa: E402
from deep_research.models import CoverageScore  # noqa: E402


def test_coverage_scorer_returns_valid_score():
    build_coverage_scorer_agent.cache_clear()
    agent = build_coverage_scorer_agent.__wrapped__("test")
    with agent.override(model=TestModel()):
        result = agent.run_sync("Score coverage for test plan")
    assert isinstance(result.output, CoverageScore)
    assert 0.0 <= result.output.total <= 1.0
