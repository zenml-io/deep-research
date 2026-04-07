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


def test_research_plan_phase_one_defaults_preserve_existing_payloads() -> None:
    plan = ResearchPlan(
        goal="Understand Kitaru",
        key_questions=["What is replay?"],
        subtopics=["replay"],
        queries=["kitaru replay checkpoints"],
        sections=["Overview"],
        success_criteria=["Explain replay anchors"],
    )

    assert plan.query_groups == {}
    assert plan.allowed_source_groups == []
    assert plan.approval_status == "not_requested"


def test_run_summary_and_render_payload_accept_phase_one_metadata() -> None:
    summary = RunSummary(
        run_id="run-1",
        brief="test",
        tier=Tier.STANDARD,
        stop_reason=StopReason.CONVERGED,
        status="completed",
        estimated_cost_usd=1.25,
        elapsed_seconds=42,
        iteration_count=3,
        provider_usage_summary={"openai": 2, "perplexity": 1},
        council_enabled=True,
        council_size=3,
        council_models=["gpt-4.1", "sonnet"],
        started_at="2026-04-07T10:00:00Z",
        completed_at="2026-04-07T10:00:42Z",
    )
    render = RenderPayload(
        name="final_report",
        content_markdown="# Report",
        citation_map={"[1]": "candidate-1"},
        structured_content={"sections": ["Overview"], "word_count": 1200},
        generated_at="2026-04-07T10:00:42Z",
    )

    assert summary.estimated_cost_usd == 1.25
    assert summary.elapsed_seconds == 42
    assert summary.iteration_count == 3
    assert summary.provider_usage_summary == {"openai": 2, "perplexity": 1}
    assert summary.council_enabled is True
    assert summary.council_size == 3
    assert summary.council_models == ["gpt-4.1", "sonnet"]
    assert summary.started_at == "2026-04-07T10:00:00Z"
    assert isinstance(summary.started_at, str)
    assert summary.completed_at == "2026-04-07T10:00:42Z"
    assert isinstance(summary.completed_at, str)
    assert render.structured_content == {
        "sections": ["Overview"],
        "word_count": 1200,
    }
    assert render.generated_at == "2026-04-07T10:00:42Z"
    assert isinstance(render.generated_at, str)


def test_iteration_record_accepts_phase_one_cost_metadata() -> None:
    record = IterationRecord(
        iteration=1,
        new_candidate_count=2,
        coverage=0.5,
        estimated_cost_usd=0.75,
    )

    assert record.estimated_cost_usd == 0.75


def test_iteration_record_rejects_negative_phase_one_cost() -> None:
    with pytest.raises(ValidationError):
        IterationRecord(iteration=1, estimated_cost_usd=-1)


def test_run_summary_rejects_negative_phase_one_counters() -> None:
    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            estimated_cost_usd=-0.1,
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            elapsed_seconds=-1,
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            iteration_count=-1,
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            provider_usage_summary={"openai": -1},
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            council_size=0,
        )


def test_research_plan_rejects_invalid_phase_one_approval_status() -> None:
    with pytest.raises(ValidationError):
        ResearchPlan(
            goal="Understand Kitaru",
            key_questions=["What is replay?"],
            subtopics=["replay"],
            queries=["kitaru replay checkpoints"],
            sections=["Overview"],
            success_criteria=["Explain replay anchors"],
            approval_status="in_review",
        )


def test_phase_one_models_reject_non_finite_costs() -> None:
    with pytest.raises(ValidationError):
        IterationRecord(iteration=1, estimated_cost_usd=float("nan"))

    with pytest.raises(ValidationError):
        IterationRecord(iteration=1, estimated_cost_usd=float("inf"))

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            estimated_cost_usd=float("nan"),
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            estimated_cost_usd=float("inf"),
        )


def test_phase_one_models_reject_invalid_timestamps() -> None:
    with pytest.raises(ValidationError):
        RenderPayload(
            name="final_report",
            content_markdown="# Report",
            generated_at="not-a-timestamp",
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            started_at="yesterday",
        )

    with pytest.raises(ValidationError):
        RunSummary(
            run_id="run-1",
            brief="test",
            tier=Tier.STANDARD,
            stop_reason=StopReason.CONVERGED,
            status="completed",
            completed_at="not-a-date",
        )


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
