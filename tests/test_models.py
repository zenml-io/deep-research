from deep_research.enums import StopReason, Tier
from deep_research.models import (
    CoverageScore,
    EvidenceCandidate,
    InvestigationPackage,
    IterationBudget,
    RawToolResult,
    RequestClassification,
    ResearchPlan,
    RelevanceCheckpointResult,
    SupervisorCheckpointResult,
    SupervisorDecision,
    ToolCallRecord,
)


def test_research_plan_round_trip() -> None:
    plan = ResearchPlan(
        goal="Understand Kitaru",
        key_questions=["What is replay?"],
        subtopics=["replay"],
        queries=["kitaru replay checkpoints"],
        sections=["Overview"],
        success_criteria=["Explain replay anchors"],
    )

    restored = ResearchPlan.model_validate(plan.model_dump())

    assert restored == plan


def test_package_minimum_shape() -> None:
    package = InvestigationPackage(
        run_summary={
            "run_id": "run-1",
            "brief": "test",
            "tier": Tier.STANDARD,
            "stop_reason": StopReason.MAX_ITERATIONS,
            "status": "completed",
        },
        research_plan={
            "goal": "x",
            "key_questions": [],
            "subtopics": [],
            "queries": [],
            "sections": [],
            "success_criteria": [],
        },
        evidence_ledger={"entries": []},
        selection_graph={"items": []},
        iteration_trace={"iterations": []},
        renders=[],
    )

    assert package.run_summary.run_id == "run-1"


def test_additional_models_are_importable_and_validate() -> None:
    candidate = EvidenceCandidate(
        snippet={
            "title": "Replay",
            "url": "https://example.com/replay",
            "excerpt": "Replay resumes from a checkpoint.",
        },
        rationale="Relevant to the goal.",
    )

    decision = SupervisorDecision(
        rationale="Need more targeted search.",
        search_actions=["search replay anchors"],
    )

    tool_record = ToolCallRecord(tool_name="search", status="ok", provider="test")
    budget = IterationBudget(input_tokens=10, output_tokens=20, total_tokens=30)
    raw_result = RawToolResult(
        tool_name="search", provider="test", payload={"items": []}
    )
    coverage = CoverageScore(
        subtopic_coverage=0.8,
        source_diversity=0.7,
        evidence_density=0.9,
        total=0.8,
    )
    classification = RequestClassification(
        audience_mode="technical",
        freshness_mode="current",
        recommended_tier=Tier.DEEP,
    )
    supervisor_result = SupervisorCheckpointResult(
        raw_results=[raw_result], budget=budget
    )
    relevance_result = RelevanceCheckpointResult(candidates=[candidate], budget=budget)

    assert decision.search_actions == ["search replay anchors"]
    assert tool_record.provider == "test"
    assert raw_result.ok is True
    assert coverage.total == 0.8
    assert classification.recommended_tier is Tier.DEEP
    assert supervisor_result.raw_results[0] == raw_result
    assert relevance_result.candidates[0] == candidate
