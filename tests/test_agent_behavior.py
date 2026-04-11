"""Tests that exercise real PydanticAI agent behavior using TestModel.

These complement the factory tests in test_agent_factories.py, which verify
construction kwargs via monkeypatching. Here we build actual PydanticAI agents
and run them against TestModel to confirm structured output parsing, field
validation, and type correctness.

Requires ``pydantic-ai`` to be installed (skipped otherwise).
"""

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")
Agent = pydantic_ai.Agent

from pydantic_ai.models.test import TestModel  # noqa: E402

from deep_research.enums import Tier  # noqa: E402
from deep_research.models import (  # noqa: E402
    CoherenceResult,
    CritiqueResult,
    CritiqueDimensionScore,
    GroundingResult,
    GroundingVerdict,
    RequestClassification,
)
from deep_research.prompts.loader import load_prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Classifier agent
# ---------------------------------------------------------------------------


def _build_classifier_agent() -> Agent[None, RequestClassification]:
    """Build a classifier agent with the production output type and system prompt."""
    return Agent(
        "test",
        name="classifier",
        output_type=RequestClassification,
        instructions=load_prompt("classifier"),
    )


def test_classifier_returns_structured_output() -> None:
    """Default TestModel output parses into a valid RequestClassification."""
    agent = _build_classifier_agent()
    with agent.override(model=TestModel()):
        result = agent.run_sync("Research quantum computing advances")

    assert isinstance(result.output, RequestClassification)
    assert isinstance(result.output.recommended_tier, Tier)
    assert isinstance(result.output.needs_clarification, bool)


def test_classifier_with_custom_fields() -> None:
    """Custom output args are forwarded into the parsed model."""
    agent = _build_classifier_agent()
    custom_args = {
        "audience_mode": "expert",
        "freshness_mode": "recent",
        "recommended_tier": "deep",
        "needs_clarification": False,
        "clarification_question": None,
    }
    with agent.override(model=TestModel(custom_output_args=custom_args)):
        result = agent.run_sync("Explain dark matter research")

    assert result.output.audience_mode == "expert"
    assert result.output.freshness_mode == "recent"
    assert result.output.recommended_tier == Tier.DEEP
    assert result.output.needs_clarification is False
    assert result.output.clarification_question is None


def test_classifier_clarification_branch() -> None:
    """When needs_clarification is True, the question field must be present."""
    agent = _build_classifier_agent()
    custom_args = {
        "audience_mode": "general",
        "freshness_mode": "any",
        "recommended_tier": "quick",
        "needs_clarification": True,
        "clarification_question": "Could you narrow the topic?",
    }
    with agent.override(model=TestModel(custom_output_args=custom_args)):
        result = agent.run_sync("Research stuff")

    assert result.output.needs_clarification is True
    assert result.output.clarification_question == "Could you narrow the topic?"


# ---------------------------------------------------------------------------
# Reviewer agent
# ---------------------------------------------------------------------------


def _build_reviewer_agent() -> Agent[None, CritiqueResult]:
    """Build a reviewer agent with the production output type and system prompt."""
    return Agent(
        "test",
        name="reviewer",
        output_type=CritiqueResult,
        instructions=load_prompt("reviewer"),
    )


def test_reviewer_returns_structured_output() -> None:
    """Default TestModel output parses into a valid CritiqueResult."""
    agent = _build_reviewer_agent()
    with agent.override(model=TestModel()):
        result = agent.run_sync("Review this report about climate change")

    assert isinstance(result.output, CritiqueResult)
    assert isinstance(result.output.summary, str)
    assert isinstance(result.output.revision_recommended, bool)


def test_reviewer_with_dimensions() -> None:
    """CritiqueResult correctly parses nested CritiqueDimensionScore items."""
    agent = _build_reviewer_agent()
    custom_args = {
        "dimensions": [
            {"name": "accuracy", "score": 0.9, "rationale": "Well sourced"},
            {"name": "clarity", "score": 0.7, "rationale": "Some jargon"},
        ],
        "summary": "Good report with minor clarity issues",
        "revision_suggestions": ["Simplify technical terms"],
        "revision_recommended": True,
    }
    with agent.override(model=TestModel(custom_output_args=custom_args)):
        result = agent.run_sync("Review this report")

    output = result.output
    assert len(output.dimensions) == 2
    assert all(isinstance(d, CritiqueDimensionScore) for d in output.dimensions)
    assert output.dimensions[0].name == "accuracy"
    assert output.dimensions[0].score == pytest.approx(0.9)
    assert output.revision_recommended is True
    assert output.revision_suggestions == ["Simplify technical terms"]


# ---------------------------------------------------------------------------
# Grounding judge agent
# ---------------------------------------------------------------------------


def _build_grounding_judge_agent() -> Agent[None, GroundingResult]:
    """Build a grounding judge agent with the production output type and system prompt."""
    return Agent(
        "test",
        name="grounding_judge",
        output_type=GroundingResult,
        instructions=load_prompt("judge_grounding"),
    )


def test_grounding_judge_returns_structured_output() -> None:
    """Default TestModel output parses into a valid GroundingResult."""
    agent = _build_grounding_judge_agent()
    with agent.override(model=TestModel()):
        result = agent.run_sync("Judge grounding of this report")

    assert isinstance(result.output, GroundingResult)
    assert 0.0 <= result.output.score <= 1.0


def test_grounding_judge_with_verdicts() -> None:
    """GroundingResult correctly parses nested GroundingVerdict items."""
    agent = _build_grounding_judge_agent()
    custom_args = {
        "score": 0.85,
        "verdicts": [
            {
                "citation": "[1]",
                "candidate_key": "src-001",
                "supported": True,
                "rationale": "Claim matches source",
            },
            {
                "citation": "[2]",
                "candidate_key": "src-002",
                "supported": False,
                "rationale": "Source does not support this claim",
            },
        ],
    }
    with agent.override(model=TestModel(custom_output_args=custom_args)):
        result = agent.run_sync("Judge grounding")

    output = result.output
    assert output.score == pytest.approx(0.85)
    assert len(output.verdicts) == 2
    assert all(isinstance(v, GroundingVerdict) for v in output.verdicts)
    assert output.verdicts[0].supported is True
    assert output.verdicts[1].supported is False


# ---------------------------------------------------------------------------
# Coherence judge agent
# ---------------------------------------------------------------------------


def _build_coherence_judge_agent() -> Agent[None, CoherenceResult]:
    """Build a coherence judge agent with the production output type and system prompt."""
    return Agent(
        "test",
        name="coherence_judge",
        output_type=CoherenceResult,
        instructions=load_prompt("judge_coherence"),
    )


def test_coherence_judge_returns_structured_output() -> None:
    """Default TestModel output parses into a valid CoherenceResult."""
    agent = _build_coherence_judge_agent()
    with agent.override(model=TestModel()):
        result = agent.run_sync("Judge coherence of this report")

    assert isinstance(result.output, CoherenceResult)
    for field_name in ("relevance", "logical_flow", "completeness", "consistency"):
        value = getattr(result.output, field_name)
        assert 0.0 <= value <= 1.0, f"{field_name} out of range: {value}"


def test_coherence_judge_with_custom_scores() -> None:
    """Custom scores are correctly parsed and satisfy validators."""
    agent = _build_coherence_judge_agent()
    custom_args = {
        "relevance": 0.95,
        "logical_flow": 0.8,
        "completeness": 0.7,
        "consistency": 0.85,
        "summary": "Strong coherence with minor flow issues",
    }
    with agent.override(model=TestModel(custom_output_args=custom_args)):
        result = agent.run_sync("Judge coherence")

    output = result.output
    assert output.relevance == pytest.approx(0.95)
    assert output.logical_flow == pytest.approx(0.8)
    assert output.completeness == pytest.approx(0.7)
    assert output.consistency == pytest.approx(0.85)
    assert output.summary == "Strong coherence with minor flow issues"
