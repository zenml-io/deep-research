"""Tests for V2 data contracts: brief, plan, evidence, decisions, reports, package."""

import pytest
from pydantic import ValidationError

from research.contracts import (
    CouncilComparison,
    CouncilPackage,
    CritiqueDimensionScore,
    CritiqueReport,
    DraftReport,
    EvidenceItem,
    EvidenceLedger,
    FinalReport,
    InvestigationPackage,
    IterationRecord,
    ProviderResolution,
    ResearchBrief,
    ResearchPlan,
    RunMetadata,
    StrictBase,
    ToolProviderManifest,
    ToolResolution,
    SubagentFindings,
    SubagentTask,
    SupervisorDecision,
    VerificationIssue,
    VerificationReport,
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
        assert brief.recency_days is None
        assert brief.source_preferences == []

    def test_optional_fields_accepted(self):
        brief = ResearchBrief(
            topic="RLHF",
            raw_request="q",
            audience="researchers",
            scope="2024",
            freshness_constraint="last 6 months",
            recency_days=180,
            source_preferences=["arxiv", "peer-reviewed"],
        )
        assert brief.audience == "researchers"
        assert brief.scope == "2024"
        assert brief.freshness_constraint == "last 6 months"
        assert brief.recency_days == 180
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
        assert task.recency_days is None

    def test_recency_days_accepted(self):
        task = SubagentTask(
            task_description="D",
            target_subtopic="S",
            recency_days=30,
        )
        assert task.recency_days == 30

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
# SubagentFindings
# ---------------------------------------------------------------------------


class TestSubagentFindings:
    def test_requires_findings(self):
        sf = SubagentFindings(findings=["DPO is effective"])
        assert sf.findings == ["DPO is effective"]

    def test_missing_findings_raises(self):
        with pytest.raises(ValidationError):
            SubagentFindings()  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        sf = SubagentFindings(findings=["f"])
        assert sf.source_references == []
        assert sf.excerpts == []
        assert sf.confidence_notes is None

    def test_all_fields_accepted(self):
        sf = SubagentFindings(
            findings=["f1", "f2"],
            source_references=["ref1"],
            excerpts=["excerpt1"],
            confidence_notes="High",
        )
        assert sf.findings == ["f1", "f2"]
        assert sf.source_references == ["ref1"]
        assert sf.excerpts == ["excerpt1"]
        assert sf.confidence_notes == "High"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            SubagentFindings(findings=["f"], bogus="x")


# ---------------------------------------------------------------------------
# SupervisorDecision
# ---------------------------------------------------------------------------


class TestSupervisorDecision:
    def test_requires_done_and_rationale(self):
        sd = SupervisorDecision(done=False, rationale="Need more data")
        assert sd.done is False
        assert sd.rationale == "Need more data"

    def test_missing_done_raises(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(rationale="R")  # type: ignore[call-arg]

    def test_missing_rationale_raises(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(done=True)  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        sd = SupervisorDecision(done=True, rationale="Complete")
        assert sd.gaps == []
        assert sd.subagent_tasks == []
        assert sd.pinned_evidence_ids == []

    def test_with_subagent_tasks(self):
        task = SubagentTask(task_description="Search DPO", target_subtopic="DPO")
        sd = SupervisorDecision(
            done=False,
            rationale="Need DPO coverage",
            gaps=["DPO alternatives"],
            subagent_tasks=[task],
            pinned_evidence_ids=["ev-001"],
        )
        assert len(sd.subagent_tasks) == 1
        assert sd.gaps == ["DPO alternatives"]
        assert sd.pinned_evidence_ids == ["ev-001"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            SupervisorDecision(done=True, rationale="R", extra="bad")


# ---------------------------------------------------------------------------
# IterationRecord
# ---------------------------------------------------------------------------


class TestIterationRecord:
    def test_requires_iteration_index_and_supervisor_decision(self):
        sd = SupervisorDecision(done=False, rationale="Continue")
        rec = IterationRecord(iteration_index=0, supervisor_decision=sd)
        assert rec.iteration_index == 0
        assert rec.supervisor_decision.done is False

    def test_missing_iteration_index_raises(self):
        sd = SupervisorDecision(done=False, rationale="R")
        with pytest.raises(ValidationError):
            IterationRecord(supervisor_decision=sd)  # type: ignore[call-arg]

    def test_missing_supervisor_decision_raises(self):
        with pytest.raises(ValidationError):
            IterationRecord(iteration_index=0)  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        sd = SupervisorDecision(done=True, rationale="Done")
        rec = IterationRecord(iteration_index=1, supervisor_decision=sd)
        assert rec.subagent_results == []
        assert rec.ledger_size == 0
        assert rec.supervisor_done_ignored is False
        assert rec.cost_usd == 0.0
        assert rec.duration_seconds == 0.0

    def test_with_subagent_results(self):
        sd = SupervisorDecision(done=False, rationale="Continue")
        findings = SubagentFindings(findings=["Found DPO paper"])
        rec = IterationRecord(
            iteration_index=2,
            supervisor_decision=sd,
            subagent_results=[findings],
            ledger_size=5,
            cost_usd=0.02,
            duration_seconds=12.5,
        )
        assert len(rec.subagent_results) == 1
        assert rec.ledger_size == 5
        assert rec.cost_usd == 0.02
        assert rec.duration_seconds == 12.5

    def test_rejects_extra_fields(self):
        sd = SupervisorDecision(done=True, rationale="R")
        with pytest.raises(ValidationError, match="extra_forbidden"):
            IterationRecord(iteration_index=0, supervisor_decision=sd, extra="bad")


# ---------------------------------------------------------------------------
# DraftReport
# ---------------------------------------------------------------------------


class TestDraftReport:
    def test_requires_content(self):
        dr = DraftReport(content="# Report\n\nFindings [ev-001].")
        assert dr.content == "# Report\n\nFindings [ev-001]."

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            DraftReport()  # type: ignore[call-arg]

    def test_sections_default_empty(self):
        dr = DraftReport(content="text")
        assert dr.sections == []

    def test_with_sections(self):
        dr = DraftReport(content="text", sections=["Introduction", "Results"])
        assert dr.sections == ["Introduction", "Results"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            DraftReport(content="text", extra="bad")

    def test_from_markdown_extracts_sections(self):
        md = "## Executive Summary\nOverview.\n\n### Findings\nDetails.\n\n## Limitations\nGaps."
        dr = DraftReport.from_markdown(md)
        assert dr.content == md
        assert dr.sections == ["Executive Summary", "Findings", "Limitations"]

    def test_from_markdown_empty_text(self):
        dr = DraftReport.from_markdown("")
        assert dr.content == ""
        assert dr.sections == []

    def test_from_markdown_no_headings(self):
        md = "Just a plain paragraph with no headings."
        dr = DraftReport.from_markdown(md)
        assert dr.content == md
        assert dr.sections == []


# ---------------------------------------------------------------------------
# CritiqueDimensionScore
# ---------------------------------------------------------------------------


class TestCritiqueDimensionScore:
    def test_requires_dimension_score_explanation(self):
        cds = CritiqueDimensionScore(
            dimension="source_reliability", score=0.85, explanation="Well sourced"
        )
        assert cds.dimension == "source_reliability"
        assert cds.score == 0.85
        assert cds.explanation == "Well sourced"

    def test_missing_dimension_raises(self):
        with pytest.raises(ValidationError):
            CritiqueDimensionScore(score=0.5, explanation="E")  # type: ignore[call-arg]

    def test_missing_score_raises(self):
        with pytest.raises(ValidationError):
            CritiqueDimensionScore(dimension="D", explanation="E")  # type: ignore[call-arg]

    def test_missing_explanation_raises(self):
        with pytest.raises(ValidationError):
            CritiqueDimensionScore(dimension="D", score=0.5)  # type: ignore[call-arg]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CritiqueDimensionScore(
                dimension="D", score=0.5, explanation="E", extra="bad"
            )


# ---------------------------------------------------------------------------
# CritiqueReport
# ---------------------------------------------------------------------------


class TestCritiqueReport:
    def test_requires_dimensions_and_require_more_research(self):
        dims = [
            CritiqueDimensionScore(
                dimension="source_reliability", score=0.9, explanation="Good"
            ),
            CritiqueDimensionScore(
                dimension="completeness", score=0.7, explanation="Missing areas"
            ),
            CritiqueDimensionScore(
                dimension="grounding", score=0.8, explanation="Well grounded"
            ),
        ]
        cr = CritiqueReport(dimensions=dims, require_more_research=True)
        assert len(cr.dimensions) == 3
        assert cr.require_more_research is True

    def test_missing_dimensions_raises(self):
        with pytest.raises(ValidationError):
            CritiqueReport(require_more_research=False)  # type: ignore[call-arg]

    def test_missing_require_more_research_raises(self):
        with pytest.raises(ValidationError):
            CritiqueReport(dimensions=[])  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        cr = CritiqueReport(dimensions=[], require_more_research=False)
        assert cr.issues == []
        assert cr.reviewer_provenance == []

    def test_with_optional_fields(self):
        cr = CritiqueReport(
            dimensions=[],
            require_more_research=False,
            issues=["Missing recent papers"],
            reviewer_provenance=["reviewer-1", "reviewer-2"],
        )
        assert cr.issues == ["Missing recent papers"]
        assert cr.reviewer_provenance == ["reviewer-1", "reviewer-2"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CritiqueReport(dimensions=[], require_more_research=False, extra="bad")


# ---------------------------------------------------------------------------
# FinalReport
# ---------------------------------------------------------------------------


class TestFinalReport:
    def test_requires_content(self):
        fr = FinalReport(content="# Final Report\n\n[ev-001] supports X.")
        assert fr.content == "# Final Report\n\n[ev-001] supports X."

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            FinalReport()  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        fr = FinalReport(content="text")
        assert fr.sections == []
        assert fr.stop_reason is None

    def test_with_optional_fields(self):
        fr = FinalReport(
            content="text",
            sections=["Summary", "Conclusions"],
            stop_reason="converged",
        )
        assert fr.sections == ["Summary", "Conclusions"]
        assert fr.stop_reason == "converged"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            FinalReport(content="text", extra="bad")

    def test_from_markdown_extracts_sections(self):
        md = "## Summary\nContent.\n\n### Details\nMore.\n\n## Conclusions\nEnd."
        fr = FinalReport.from_markdown(md, stop_reason="converged")
        assert fr.content == md
        assert fr.sections == ["Summary", "Details", "Conclusions"]
        assert fr.stop_reason == "converged"

    def test_from_markdown_no_stop_reason(self):
        md = "## Report\nBody."
        fr = FinalReport.from_markdown(md)
        assert fr.content == md
        assert fr.sections == ["Report"]
        assert fr.stop_reason is None

    def test_from_markdown_empty_text(self):
        fr = FinalReport.from_markdown("", stop_reason="budget_exhausted")
        assert fr.content == ""
        assert fr.sections == []
        assert fr.stop_reason == "budget_exhausted"


# ---------------------------------------------------------------------------
# VerificationIssue
# ---------------------------------------------------------------------------


class TestVerificationIssue:
    def test_requires_claim_excerpt(self):
        issue = VerificationIssue(claim_excerpt="Claim text")
        assert issue.claim_excerpt == "Claim text"

    def test_missing_claim_excerpt_raises(self):
        with pytest.raises(ValidationError):
            VerificationIssue()  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        issue = VerificationIssue(claim_excerpt="Claim text")
        assert issue.evidence_ids == []
        assert issue.status == "unsupported"
        assert issue.reason is None
        assert issue.suggested_fix is None

    def test_with_optional_fields(self):
        issue = VerificationIssue(
            claim_excerpt="Claim text",
            evidence_ids=["ev-001"],
            status="partial",
            reason="Only narrower support exists",
            suggested_fix="Narrow the wording",
        )
        assert issue.evidence_ids == ["ev-001"]
        assert issue.status == "partial"
        assert issue.reason == "Only narrower support exists"
        assert issue.suggested_fix == "Narrow the wording"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            VerificationIssue(claim_excerpt="Claim text", extra="bad")


# ---------------------------------------------------------------------------
# VerificationReport
# ---------------------------------------------------------------------------


class TestVerificationReport:
    def test_optional_fields_default(self):
        report = VerificationReport()
        assert report.issues == []
        assert report.verified_claim_count == 0
        assert report.unsupported_claim_count == 0
        assert report.needs_revision is False

    def test_with_issues(self):
        report = VerificationReport(
            issues=[
                VerificationIssue(
                    claim_excerpt="Claim text",
                    evidence_ids=["ev-001"],
                    status="contradicted",
                    reason="Evidence says the opposite",
                )
            ],
            verified_claim_count=5,
            unsupported_claim_count=1,
            needs_revision=True,
        )
        assert len(report.issues) == 1
        assert report.issues[0].status == "contradicted"
        assert report.verified_claim_count == 5
        assert report.unsupported_claim_count == 1
        assert report.needs_revision is True

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            VerificationReport(extra="bad")


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_requires_run_id_tier_started_at(self):
        rm = RunMetadata(
            run_id="run-001", tier="standard", started_at="2024-01-01T00:00:00Z"
        )
        assert rm.run_id == "run-001"
        assert rm.tier == "standard"
        assert rm.started_at == "2024-01-01T00:00:00Z"

    def test_missing_run_id_raises(self):
        with pytest.raises(ValidationError):
            RunMetadata(tier="standard", started_at="2024-01-01T00:00:00Z")  # type: ignore[call-arg]

    def test_missing_tier_raises(self):
        with pytest.raises(ValidationError):
            RunMetadata(run_id="r", started_at="2024-01-01T00:00:00Z")  # type: ignore[call-arg]

    def test_missing_started_at_raises(self):
        with pytest.raises(ValidationError):
            RunMetadata(run_id="r", tier="standard")  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        rm = RunMetadata(run_id="r", tier="quick", started_at="2024-01-01T00:00:00Z")
        assert rm.completed_at is None
        assert rm.total_cost_usd == 0.0
        assert rm.total_iterations == 0
        assert rm.stop_reason is None

    def test_with_optional_fields(self):
        rm = RunMetadata(
            run_id="r",
            tier="deep",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T01:00:00Z",
            total_cost_usd=1.50,
            total_iterations=5,
            stop_reason="converged",
        )
        assert rm.completed_at == "2024-01-01T01:00:00Z"
        assert rm.total_cost_usd == 1.50
        assert rm.total_iterations == 5
        assert rm.stop_reason == "converged"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            RunMetadata(
                run_id="r",
                tier="standard",
                started_at="2024-01-01T00:00:00Z",
                extra="bad",
            )


# ---------------------------------------------------------------------------
# InvestigationPackage
# ---------------------------------------------------------------------------


def _make_metadata():
    return RunMetadata(
        run_id="run-001", tier="standard", started_at="2024-01-01T00:00:00Z"
    )


def _make_brief():
    return ResearchBrief(topic="RLHF", raw_request="Tell me about RLHF")


def _make_plan():
    return ResearchPlan(goal="Understand RLHF", key_questions=["What is RLHF?"])


def _make_ledger():
    return EvidenceLedger()


class TestInvestigationPackage:
    def test_requires_metadata_brief_plan_ledger(self):
        pkg = InvestigationPackage(
            metadata=_make_metadata(),
            brief=_make_brief(),
            plan=_make_plan(),
            ledger=_make_ledger(),
        )
        assert pkg.metadata.run_id == "run-001"
        assert pkg.brief.topic == "RLHF"
        assert pkg.plan.goal == "Understand RLHF"
        assert pkg.ledger.items == []

    def test_defaults_schema_version_to_1_0(self):
        pkg = InvestigationPackage(
            metadata=_make_metadata(),
            brief=_make_brief(),
            plan=_make_plan(),
            ledger=_make_ledger(),
        )
        assert pkg.schema_version == "1.0"

    def test_missing_metadata_raises(self):
        with pytest.raises(ValidationError):
            InvestigationPackage(  # type: ignore[call-arg]
                brief=_make_brief(),
                plan=_make_plan(),
                ledger=_make_ledger(),
            )

    def test_missing_brief_raises(self):
        with pytest.raises(ValidationError):
            InvestigationPackage(  # type: ignore[call-arg]
                metadata=_make_metadata(),
                plan=_make_plan(),
                ledger=_make_ledger(),
            )

    def test_missing_plan_raises(self):
        with pytest.raises(ValidationError):
            InvestigationPackage(  # type: ignore[call-arg]
                metadata=_make_metadata(),
                brief=_make_brief(),
                ledger=_make_ledger(),
            )

    def test_missing_ledger_raises(self):
        with pytest.raises(ValidationError):
            InvestigationPackage(  # type: ignore[call-arg]
                metadata=_make_metadata(),
                brief=_make_brief(),
                plan=_make_plan(),
            )

    def test_optional_fields_default(self):
        pkg = InvestigationPackage(
            metadata=_make_metadata(),
            brief=_make_brief(),
            plan=_make_plan(),
            ledger=_make_ledger(),
        )
        assert pkg.iterations == []
        assert pkg.draft is None
        assert pkg.critique is None
        assert pkg.final_report is None
        assert pkg.verification is None
        assert pkg.revised_plan is None
        assert pkg.prompt_hashes == {}
        assert pkg.tool_provider_manifest.configured_providers == []
        assert pkg.metadata.export_path is None

    def test_with_all_optional_fields(self):
        sd = SupervisorDecision(done=True, rationale="Complete")
        iteration = IterationRecord(iteration_index=0, supervisor_decision=sd)
        draft = DraftReport(content="draft text")
        dims = [
            CritiqueDimensionScore(dimension="grounding", score=0.9, explanation="Good")
        ]
        critique = CritiqueReport(dimensions=dims, require_more_research=False)
        final = FinalReport(content="final text", stop_reason="converged")
        verification = VerificationReport(
            issues=[
                VerificationIssue(
                    claim_excerpt="final text",
                    evidence_ids=["ev-001"],
                    status="partial",
                )
            ],
            verified_claim_count=3,
            unsupported_claim_count=0,
            needs_revision=False,
        )

        pkg = InvestigationPackage(
            metadata=_make_metadata(),
            brief=_make_brief(),
            plan=_make_plan(),
            revised_plan=ResearchPlan(
                goal="Revised RLHF plan",
                key_questions=["What changed after critique?"],
            ),
            ledger=_make_ledger(),
            iterations=[iteration],
            draft=draft,
            critique=critique,
            final_report=final,
            verification=verification,
            prompt_hashes={"supervisor": "abc123"},
            tool_provider_manifest=ToolProviderManifest(
                configured_providers=["arxiv"],
                provider_resolutions=[ProviderResolution(provider="arxiv", instantiated=True, available=True)],
                tool_resolutions=[ToolResolution(tool="search", enabled=True)],
            ),
        )
        assert len(pkg.iterations) == 1
        assert pkg.draft.content == "draft text"
        assert pkg.critique.require_more_research is False
        assert pkg.final_report.stop_reason == "converged"
        assert pkg.verification is verification
        assert pkg.revised_plan is not None
        assert pkg.prompt_hashes == {"supervisor": "abc123"}
        assert pkg.tool_provider_manifest.configured_providers == ["arxiv"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            InvestigationPackage(
                metadata=_make_metadata(),
                brief=_make_brief(),
                plan=_make_plan(),
                ledger=_make_ledger(),
                extra="bad",
            )


# ---------------------------------------------------------------------------
# CouncilComparison
# ---------------------------------------------------------------------------


class TestCouncilComparison:
    def test_requires_comparison(self):
        cc = CouncilComparison(comparison="Generator A was better")
        assert cc.comparison == "Generator A was better"

    def test_missing_comparison_raises(self):
        with pytest.raises(ValidationError):
            CouncilComparison()  # type: ignore[call-arg]

    def test_optional_fields_default(self):
        cc = CouncilComparison(comparison="text")
        assert cc.generator_scores == {}
        assert cc.recommended_generator is None

    def test_with_optional_fields(self):
        cc = CouncilComparison(
            comparison="A > B",
            generator_scores={"gemini": 0.9, "claude": 0.85},
            recommended_generator="gemini",
        )
        assert cc.generator_scores == {"gemini": 0.9, "claude": 0.85}
        assert cc.recommended_generator == "gemini"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CouncilComparison(comparison="text", extra="bad")


# ---------------------------------------------------------------------------
# CouncilPackage
# ---------------------------------------------------------------------------


class TestCouncilPackage:
    def test_defaults_schema_version_to_1_0(self):
        cp = CouncilPackage()
        assert cp.schema_version == "1.0"

    def test_optional_fields_default(self):
        cp = CouncilPackage()
        assert cp.canonical_generator is None
        assert cp.council_provider_compromise is False
        assert cp.comparison is None
        assert cp.packages == {}

    def test_with_packages(self):
        pkg = InvestigationPackage(
            metadata=_make_metadata(),
            brief=_make_brief(),
            plan=_make_plan(),
            ledger=_make_ledger(),
        )
        cp = CouncilPackage(
            canonical_generator="gemini",
            packages={"gemini": pkg},
            comparison=CouncilComparison(comparison="Only one generator"),
        )
        assert cp.canonical_generator == "gemini"
        assert "gemini" in cp.packages
        assert cp.comparison.comparison == "Only one generator"

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            CouncilPackage(extra="bad")


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
        (SubagentFindings, {"findings": ["f"]}),
        (SupervisorDecision, {"done": True, "rationale": "R"}),
        (
            IterationRecord,
            {
                "iteration_index": 0,
                "supervisor_decision": SupervisorDecision(done=True, rationale="R"),
            },
        ),
        (DraftReport, {"content": "C"}),
        (CritiqueDimensionScore, {"dimension": "D", "score": 0.5, "explanation": "E"}),
        (CritiqueReport, {"dimensions": [], "require_more_research": False}),
        (FinalReport, {"content": "C"}),
        (
            RunMetadata,
            {"run_id": "r", "tier": "standard", "started_at": "2024-01-01T00:00:00Z"},
        ),
        (ProviderResolution, {"provider": "arxiv"}),
        (ToolResolution, {"tool": "search"}),
        (ToolProviderManifest, {}),
        (
            InvestigationPackage,
            {
                "metadata": RunMetadata(
                    run_id="r", tier="standard", started_at="2024-01-01T00:00:00Z"
                ),
                "brief": ResearchBrief(topic="T", raw_request="R"),
                "plan": ResearchPlan(goal="G", key_questions=["Q"]),
                "ledger": EvidenceLedger(),
            },
        ),
        (CouncilComparison, {"comparison": "C"}),
        (CouncilPackage, {}),
    ],
)
def test_all_models_reject_extras(model_cls, valid_kwargs):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model_cls(**valid_kwargs, _sneaky="nope")
