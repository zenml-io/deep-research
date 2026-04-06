import pytest
from pydantic import ValidationError

from deep_research.enums import StopReason, Tier
from deep_research.models import (
    CoverageScore,
    EvidenceCandidate,
    EvidenceLedger,
    EvidenceSnippet,
    InvestigationPackage,
    IterationBudget,
    IterationRecord,
    IterationTrace,
    RawToolResult,
    RequestClassification,
    RenderPayload,
    ResearchPlan,
    RelevanceCheckpointResult,
    RunSummary,
    SelectionGraph,
    SelectionItem,
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
    snippet = EvidenceSnippet(
        text="Replay resumes from a checkpoint.",
        source_locator="#section-1",
    )
    candidate = EvidenceCandidate(
        key="candidate-1",
        title="Replay",
        url="https://example.com/replay",
        snippets=[snippet],
        provider="test",
        source_kind="web",
    )
    ledger = EvidenceLedger(entries=[candidate])
    selection_item = SelectionItem(
        candidate_key="candidate-1",
        rationale="Best match for replay.",
    )
    selection_graph = SelectionGraph(items=[selection_item])
    iteration = IterationRecord(iteration=1, new_candidate_count=1, coverage=0.5)
    trace = IterationTrace(iterations=[iteration])
    render = RenderPayload(
        name="final_report",
        content_markdown="# Report",
        citation_map={"[1]": "candidate-1"},
    )
    summary = RunSummary(
        run_id="run-1",
        brief="test",
        tier=Tier.STANDARD,
        stop_reason=StopReason.CONVERGED,
        status="completed",
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

    assert snippet.source_locator == "#section-1"
    assert ledger.entries == [candidate]
    assert selection_graph.items == [selection_item]
    assert trace.iterations == [iteration]
    assert render.citation_map == {"[1]": "candidate-1"}
    assert summary.stop_reason is StopReason.CONVERGED
    assert decision.search_actions == ["search replay anchors"]
    assert tool_record.provider == "test"
    assert raw_result.ok is True
    assert coverage.total == 0.8
    assert classification.recommended_tier is Tier.DEEP
    assert supervisor_result.raw_results[0] == raw_result
    assert relevance_result.candidates[0] == candidate


def test_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            goal="Understand Kitaru",
            key_questions=[],
            subtopics=[],
            queries=[],
            sections=[],
            success_criteria=[],
            extra_field="unexpected",
        )


def test_coverage_score_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        CoverageScore(
            subtopic_coverage=1.1,
            source_diversity=0.7,
            evidence_density=0.9,
            total=0.8,
        )


def test_iteration_budget_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        IterationBudget(input_tokens=-1)


def test_iteration_budget_rejects_inconsistent_total_tokens() -> None:
    with pytest.raises(ValidationError):
        IterationBudget(input_tokens=10, output_tokens=20, total_tokens=25)


def test_iteration_record_rejects_invalid_ranges() -> None:
    with pytest.raises(ValidationError):
        IterationRecord(iteration=-1)

    with pytest.raises(ValidationError):
        IterationRecord(iteration=1, new_candidate_count=-1)

    with pytest.raises(ValidationError):
        IterationRecord(iteration=1, coverage=1.1)


def test_evidence_candidate_rejects_invalid_url() -> None:
    with pytest.raises(ValidationError):
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="not-a-url",
            provider="test",
            source_kind="web",
        )


def test_evidence_candidate_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValidationError):
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="test",
            source_kind="web",
            quality_score=1.1,
        )

    with pytest.raises(ValidationError):
        EvidenceCandidate(
            key="candidate-1",
            title="Replay",
            url="https://example.com/replay",
            provider="test",
            source_kind="web",
            relevance_score=-0.1,
        )


def test_request_classification_rejects_inconsistent_clarification_state() -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            audience_mode="technical",
            freshness_mode="current",
            recommended_tier=Tier.DEEP,
            needs_clarification=True,
        )

    with pytest.raises(ValidationError):
        RequestClassification(
            audience_mode="technical",
            freshness_mode="current",
            recommended_tier=Tier.DEEP,
            needs_clarification=False,
            clarification_question="What is the deadline?",
        )


def test_request_classification_rejects_whitespace_only_question() -> None:
    with pytest.raises(ValidationError):
        RequestClassification(
            audience_mode="technical",
            freshness_mode="current",
            recommended_tier=Tier.DEEP,
            needs_clarification=True,
            clarification_question="   ",
        )
