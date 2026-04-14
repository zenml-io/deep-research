"""Tests for V2 data contracts: brief, plan, and evidence."""

import pytest
from pydantic import ValidationError

from research.contracts import (
    EvidenceItem,
    EvidenceLedger,
    ResearchBrief,
    ResearchPlan,
    StrictBase,
    SubagentTask,
)


# ---------------------------------------------------------------------------
# StrictBase
# ---------------------------------------------------------------------------


class _Dummy(StrictBase):
    name: str


class TestStrictBase:
    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _Dummy(name="ok", surprise="boom")

    def test_accepts_valid_fields(self):
        obj = _Dummy(name="ok")
        assert obj.name == "ok"


# ---------------------------------------------------------------------------
# ResearchBrief
# ---------------------------------------------------------------------------


class TestResearchBrief:
    def test_requires_topic_and_raw_request(self):
        brief = ResearchBrief(topic="RLHF", raw_request="Tell me about RLHF")
        assert brief.topic == "RLHF"
        assert brief.raw_request == "Tell me about RLHF"

    def test_missing_topic_raises(self):
        with pytest.raises(ValidationError):
            ResearchBrief(raw_request="Tell me about RLHF")  # type: ignore[call-arg]

    def test_missing_raw_request_raises(self):
        with pytest.raises(ValidationError):
            ResearchBrief(topic="RLHF")  # type: ignore[call-arg]

    def test_optional_fields_default_to_none_or_empty(self):
        brief = ResearchBrief(topic="RLHF", raw_request="q")
        assert brief.audience is None
        assert brief.scope is None
        assert brief.freshness_constraint is None
        assert brief.source_preferences == []

    def test_optional_fields_accepted(self):
        brief = ResearchBrief(
            topic="RLHF",
            raw_request="q",
            audience="researchers",
            scope="2024",
            freshness_constraint="last 6 months",
            source_preferences=["arxiv", "peer-reviewed"],
        )
        assert brief.audience == "researchers"
        assert brief.scope == "2024"
        assert brief.freshness_constraint == "last 6 months"
        assert brief.source_preferences == ["arxiv", "peer-reviewed"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ResearchBrief(topic="X", raw_request="q", unknown_field="bad")


# ---------------------------------------------------------------------------
# ResearchPlan
# ---------------------------------------------------------------------------


class TestResearchPlan:
    def test_requires_goal_and_key_questions(self):
        plan = ResearchPlan(goal="Understand RLHF", key_questions=["What is RLHF?"])
        assert plan.goal == "Understand RLHF"
        assert plan.key_questions == ["What is RLHF?"]

    def test_missing_goal_raises(self):
        with pytest.raises(ValidationError):
            ResearchPlan(key_questions=["Q"])  # type: ignore[call-arg]

    def test_missing_key_questions_raises(self):
        with pytest.raises(ValidationError):
            ResearchPlan(goal="G")  # type: ignore[call-arg]

    def test_optional_fields_default_empty(self):
        plan = ResearchPlan(goal="G", key_questions=["Q"])
        assert plan.subtopics == []
        assert plan.query_strategies == []
        assert plan.sections == []
        assert plan.success_criteria == []

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ResearchPlan(goal="G", key_questions=["Q"], bogus="x")


# ---------------------------------------------------------------------------
# SubagentTask
# ---------------------------------------------------------------------------


class TestSubagentTask:
    def test_requires_task_description_and_target_subtopic(self):
        task = SubagentTask(
            task_description="Search for DPO papers",
            target_subtopic="Direct Preference Optimization",
        )
        assert task.task_description == "Search for DPO papers"
        assert task.target_subtopic == "Direct Preference Optimization"

    def test_missing_task_description_raises(self):
        with pytest.raises(ValidationError):
            SubagentTask(target_subtopic="X")  # type: ignore[call-arg]

    def test_missing_target_subtopic_raises(self):
        with pytest.raises(ValidationError):
            SubagentTask(task_description="D")  # type: ignore[call-arg]

    def test_search_strategy_hints_default_empty(self):
        task = SubagentTask(task_description="D", target_subtopic="S")
        assert task.search_strategy_hints == []

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            SubagentTask(task_description="D", target_subtopic="S", extra="bad")


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_requires_core_fields(self):
        item = EvidenceItem(
            evidence_id="ev-001",
            title="DPO Paper",
            synthesis="DPO removes the reward model from RLHF.",
            iteration_added=1,
        )
        assert item.evidence_id == "ev-001"
        assert item.title == "DPO Paper"
        assert item.synthesis == "DPO removes the reward model from RLHF."
        assert item.iteration_added == 1

    def test_missing_evidence_id_raises(self):
        with pytest.raises(ValidationError):
            EvidenceItem(title="T", synthesis="S", iteration_added=1)  # type: ignore[call-arg]

    def test_missing_title_raises(self):
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="e", synthesis="S", iteration_added=1)  # type: ignore[call-arg]

    def test_missing_synthesis_raises(self):
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="e", title="T", iteration_added=1)  # type: ignore[call-arg]

    def test_missing_iteration_added_raises(self):
        with pytest.raises(ValidationError):
            EvidenceItem(evidence_id="e", title="T", synthesis="S")  # type: ignore[call-arg]

    def test_optional_fields_default_to_none_or_empty(self):
        item = EvidenceItem(
            evidence_id="e", title="T", synthesis="S", iteration_added=0
        )
        assert item.url is None
        assert item.doi is None
        assert item.arxiv_id is None
        assert item.canonical_url is None
        assert item.source_type is None
        assert item.excerpts == []
        assert item.confidence_notes is None
        assert item.provider is None

    def test_all_optional_fields_accepted(self):
        item = EvidenceItem(
            evidence_id="e",
            title="T",
            synthesis="S",
            iteration_added=2,
            url="https://example.com",
            doi="10.1234/test",
            arxiv_id="2301.12345",
            canonical_url="https://example.com/canonical",
            source_type="preprint",
            excerpts=["quote 1", "quote 2"],
            confidence_notes="High confidence",
            provider="arxiv",
        )
        assert item.url == "https://example.com"
        assert item.doi == "10.1234/test"
        assert item.arxiv_id == "2301.12345"
        assert item.excerpts == ["quote 1", "quote 2"]
        assert item.provider == "arxiv"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EvidenceItem(
                evidence_id="e",
                title="T",
                synthesis="S",
                iteration_added=1,
                score=0.9,
            )


# ---------------------------------------------------------------------------
# EvidenceLedger
# ---------------------------------------------------------------------------


class TestEvidenceLedger:
    def test_defaults_to_empty_items_and_version(self):
        ledger = EvidenceLedger()
        assert ledger.items == []
        assert ledger.schema_version == "1.0"

    def test_accepts_items(self):
        item = EvidenceItem(
            evidence_id="e", title="T", synthesis="S", iteration_added=1
        )
        ledger = EvidenceLedger(items=[item])
        assert len(ledger.items) == 1
        assert ledger.items[0].evidence_id == "e"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            EvidenceLedger(items=[], extra_field="bad")

    def test_schema_version_overridable(self):
        ledger = EvidenceLedger(schema_version="2.0")
        assert ledger.schema_version == "2.0"


# ---------------------------------------------------------------------------
# Cross-cutting: all models reject extras
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls,valid_kwargs",
    [
        (ResearchBrief, {"topic": "T", "raw_request": "R"}),
        (ResearchPlan, {"goal": "G", "key_questions": ["Q"]}),
        (SubagentTask, {"task_description": "D", "target_subtopic": "S"}),
        (
            EvidenceItem,
            {
                "evidence_id": "e",
                "title": "T",
                "synthesis": "S",
                "iteration_added": 1,
            },
        ),
        (EvidenceLedger, {}),
    ],
)
def test_all_models_reject_extras(model_cls, valid_kwargs):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model_cls(**valid_kwargs, _sneaky="nope")
