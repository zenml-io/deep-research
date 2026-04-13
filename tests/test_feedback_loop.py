"""Unit tests for the Wave 4.5 synthesis-to-unsupported-claims feedback loop."""

from __future__ import annotations

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.flow._pipeline import _wave_feedback
from deep_research.models import (
    CoverageScore,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
    ResearchPlan,
)


def _plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Evaluate harnesses for deep research agents",
        key_questions=["What trade-offs exist?"],
        subtopics=["orchestration"],
        queries=["harness eval"],
        sections=["Overview"],
        success_criteria=["decision brief"],
    )


def _coverage_no_gaps() -> CoverageScore:
    return CoverageScore(
        subtopic_coverage=1.0,
        plan_fidelity=1.0,
        source_diversity=1.0,
        evidence_density=1.0,
        total=1.0,
        uncovered_subtopics=[],
        unanswered_questions=[],
    )


def _ledger_with_top_snippets() -> EvidenceLedger:
    candidates = [
        EvidenceCandidate(
            key=f"candidate-{idx}",
            title=f"Harness option {idx}",
            url=f"https://example.com/harness-{idx}",
            provider="docs",
            source_kind="docs",
            matched_subtopics=["orchestration"],
            quality_score=0.8,
            relevance_score=0.75,
            authority_score=0.8,
            snippets=[
                EvidenceSnippet(
                    text=(
                        f"Harness {idx} supports durable execution with clear "
                        f"checkpoint boundaries"
                    )
                )
            ],
            selected=True,
            raw_metadata={"selection_score": 0.9 - idx * 0.01},
        )
        for idx in range(3)
    ]
    return EvidenceLedger(
        considered=candidates,
        selected=candidates,
        rejected=[],
        dedupe_log=[],
    )


def test_feedback_loop_caps_at_two_iterations() -> None:
    config = ResearchConfig.for_tier(Tier.DEEP)
    assert config.selection_policy.feedback_loop_max_iterations == 2

    ledger = _ledger_with_top_snippets()
    plan = _plan()
    coverage = _coverage_no_gaps()

    first = _wave_feedback(
        coverage, plan=plan, config=config, ledger=ledger, feedback_iteration_count=0
    )
    assert first.feedback_iteration_count == 1
    assert any("evidence" in action.query for action in first.carryover_actions)

    second = _wave_feedback(
        coverage,
        plan=plan,
        config=config,
        ledger=ledger,
        feedback_iteration_count=first.feedback_iteration_count,
    )
    assert second.feedback_iteration_count == 2

    # Third call must produce no synthesis claims — the cap is reached.
    third = _wave_feedback(
        coverage,
        plan=plan,
        config=config,
        ledger=ledger,
        feedback_iteration_count=second.feedback_iteration_count,
    )
    assert third.feedback_iteration_count == 2
    assert third.carryover_actions == []
    assert third.reason is None


def test_feedback_loop_is_disabled_on_standard_tier() -> None:
    config = ResearchConfig.for_tier(Tier.STANDARD)
    ledger = _ledger_with_top_snippets()
    result = _wave_feedback(
        _coverage_no_gaps(),
        plan=_plan(),
        config=config,
        ledger=ledger,
        feedback_iteration_count=0,
    )
    # Standard tier never consumes the synthesis budget, even with populated ledger.
    assert result.feedback_iteration_count == 0
    assert result.carryover_actions == []
