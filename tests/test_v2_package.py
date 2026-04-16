"""Tests for research.package — assembly helpers and export IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.contracts.brief import ResearchBrief
from research.contracts.decisions import SubagentFindings, SupervisorDecision
from research.contracts.evidence import EvidenceItem, EvidenceLedger
from research.contracts.iteration import IterationRecord
from research.contracts.package import (
    InvestigationPackage,
    ProviderResolution,
    RunMetadata,
    ToolProviderManifest,
    ToolResolution,
)
from research.contracts.plan import ResearchPlan
from research.contracts.reports import DraftReport, FinalReport
from research.contracts.reports import VerificationIssue, VerificationReport
from research.package.assembly import compute_evidence_stats, compute_run_summary
from research.package.export import read_package, sanitize_path_component, write_package


# ---------------------------------------------------------------------------
# Fixtures (inline, self-contained)
# ---------------------------------------------------------------------------


def _make_evidence_item(
    *,
    evidence_id: str = "ev-001",
    title: str = "Test Paper",
    url: str | None = "https://example.com/paper",
    doi: str | None = None,
    arxiv_id: str | None = None,
    provider: str | None = "arxiv",
    iteration_added: int = 0,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        title=title,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        synthesis="A synthesized finding.",
        iteration_added=iteration_added,
        provider=provider,
    )


def _make_iteration_record(*, index: int = 0) -> IterationRecord:
    return IterationRecord(
        iteration_index=index,
        supervisor_decision=SupervisorDecision(
            done=False,
            rationale="Continuing research.",
        ),
        subagent_results=[
            SubagentFindings(findings=["finding-1"]),
        ],
        ledger_size=1,
        cost_usd=0.002,
        duration_seconds=1.5,
    )


def _make_package(
    *,
    run_id: str = "run-abc-123",
    with_draft: bool = False,
    with_final: bool = False,
    evidence_items: list[EvidenceItem] | None = None,
    iterations: list[IterationRecord] | None = None,
) -> InvestigationPackage:
    draft = DraftReport(content="# Draft\nSome draft text.") if with_draft else None
    final = (
        FinalReport(content="# Final Report\nPolished content.") if with_final else None
    )
    return InvestigationPackage(
        metadata=RunMetadata(
            run_id=run_id,
            tier="standard",
            started_at="2025-01-01T00:00:00Z",
            completed_at="2025-01-01T00:05:00Z",
            total_cost_usd=0.042,
            total_iterations=2,
            stop_reason="converged",
            export_path=f"artifacts/{run_id}",
        ),
        brief=ResearchBrief(
            topic="What are RLHF alternatives?",
            raw_request="What are RLHF alternatives?",
        ),
        plan=ResearchPlan(
            goal="Survey RLHF alternatives",
            key_questions=["What methods exist beyond RLHF?"],
            subtopics=["DPO", "RLAIF"],
        ),
        ledger=EvidenceLedger(items=evidence_items or []),
        iterations=iterations or [],
        draft=draft,
        final_report=final,
        tool_provider_manifest=ToolProviderManifest(
            configured_providers=["arxiv"],
            instantiated_providers=["arxiv"],
            active_providers=["arxiv"],
            available_tools=["search", "fetch"],
            provider_resolutions=[ProviderResolution(provider="arxiv", instantiated=True, available=True)],
            tool_resolutions=[
                ToolResolution(tool="search", enabled=True),
                ToolResolution(tool="fetch", enabled=True),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# export.write_package — directory structure
# ---------------------------------------------------------------------------


def test_write_package_creates_directory_structure(tmp_path: Path) -> None:
    pkg = _make_package(
        evidence_items=[_make_evidence_item()],
        iterations=[_make_iteration_record(index=0)],
        with_draft=True,
    )

    run_dir = write_package(pkg, tmp_path)

    assert run_dir == tmp_path / "run-abc-123"
    assert (run_dir / "package.json").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "evidence" / "ledger.json").is_file()
    assert (run_dir / "iterations" / "000.json").is_file()


# ---------------------------------------------------------------------------
# export.write_package + read_package — round trip
# ---------------------------------------------------------------------------


def test_write_package_round_trip(tmp_path: Path) -> None:
    pkg = _make_package(
        evidence_items=[_make_evidence_item()],
        iterations=[_make_iteration_record(index=0)],
        with_final=True,
        with_draft=True,
    )

    run_dir = write_package(pkg, tmp_path)
    restored = read_package(run_dir)

    assert restored == pkg
    assert restored.metadata.run_id == "run-abc-123"
    assert restored.final_report is not None
    assert restored.final_report.content == "# Final Report\nPolished content."
    assert restored.metadata.export_path == "artifacts/run-abc-123"
    assert restored.tool_provider_manifest.available_tools == ["search", "fetch"]


def test_write_package_round_trip_with_verification(tmp_path: Path) -> None:
    pkg = _make_package(with_final=True, with_draft=True)
    pkg = pkg.model_copy(
        update={
            "verification": VerificationReport(
                issues=[
                    VerificationIssue(
                        claim_excerpt="Claim",
                        evidence_ids=["ev-001"],
                        status="partial",
                        reason="Only narrower support exists",
                    )
                ],
                verified_claim_count=4,
                unsupported_claim_count=0,
                needs_revision=False,
            )
        }
    )

    run_dir = write_package(pkg, tmp_path)
    restored = read_package(run_dir)

    assert restored.verification == pkg.verification


# ---------------------------------------------------------------------------
# export.write_package — report.md content
# ---------------------------------------------------------------------------


def test_write_package_with_final_report(tmp_path: Path) -> None:
    pkg = _make_package(with_final=True, with_draft=True)
    run_dir = write_package(pkg, tmp_path)

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report_md == "# Final Report\nPolished content."


def test_write_package_with_draft_only(tmp_path: Path) -> None:
    pkg = _make_package(with_draft=True, with_final=False)
    run_dir = write_package(pkg, tmp_path)

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert report_md == "# Draft\nSome draft text."


def test_write_package_no_report(tmp_path: Path) -> None:
    pkg = _make_package(with_draft=False, with_final=False)
    run_dir = write_package(pkg, tmp_path)

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "No report" in report_md


# ---------------------------------------------------------------------------
# export.write_package — per-iteration JSON
# ---------------------------------------------------------------------------


def test_write_package_iterations(tmp_path: Path) -> None:
    iters = [_make_iteration_record(index=i) for i in range(3)]
    pkg = _make_package(iterations=iters)
    run_dir = write_package(pkg, tmp_path)

    for i in range(3):
        iter_path = run_dir / "iterations" / f"{i:03d}.json"
        assert iter_path.is_file(), f"Missing {iter_path}"
        data = json.loads(iter_path.read_text(encoding="utf-8"))
        assert data["iteration_index"] == i


# ---------------------------------------------------------------------------
# export.read_package — error path
# ---------------------------------------------------------------------------


def test_read_package_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No package.json found"):
        read_package(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# export.sanitize_path_component
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "run-abc-123",
        "simple",
        "with-dashes",
        "with_underscores",
        "MixedCase99",
    ],
)
def test_sanitize_path_component_valid(value: str) -> None:
    assert sanitize_path_component(value, field_name="test") == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "foo/bar",
        "foo\\bar",
        " leading",
        "trailing ",
        " both ",
    ],
)
def test_sanitize_path_component_rejects_traversal(value: str) -> None:
    with pytest.raises(ValueError):
        sanitize_path_component(value, field_name="test")


# ---------------------------------------------------------------------------
# assembly.compute_run_summary
# ---------------------------------------------------------------------------


def test_compute_run_summary() -> None:
    pkg = _make_package(
        evidence_items=[_make_evidence_item()],
        with_final=True,
    )
    summary = compute_run_summary(pkg)

    assert "run-abc-123" in summary
    assert "standard" in summary
    assert "What are RLHF alternatives?" in summary
    assert "Iterations: 2" in summary
    assert "$0.0420" in summary
    assert "converged" in summary
    assert "Evidence items: 1" in summary
    assert "final report available" in summary


def test_compute_run_summary_draft_only() -> None:
    pkg = _make_package(with_draft=True)
    summary = compute_run_summary(pkg)
    assert "draft only" in summary


def test_compute_run_summary_no_report() -> None:
    pkg = _make_package()
    summary = compute_run_summary(pkg)
    assert "Report: none" in summary


# ---------------------------------------------------------------------------
# assembly.compute_evidence_stats
# ---------------------------------------------------------------------------


def test_compute_evidence_stats() -> None:
    items = [
        _make_evidence_item(
            evidence_id="ev-1",
            url="https://arxiv.org/abs/2301.1",
            doi="10.1234/test",
            arxiv_id="2301.1",
            provider="arxiv",
            iteration_added=0,
        ),
        _make_evidence_item(
            evidence_id="ev-2",
            url="https://example.com/page",
            provider="brave",
            iteration_added=1,
        ),
        _make_evidence_item(
            evidence_id="ev-3",
            url=None,
            provider=None,
            iteration_added=1,
        ),
    ]
    pkg = _make_package(evidence_items=items)
    stats = compute_evidence_stats(pkg)

    assert stats["total_items"] == 3
    assert "arxiv.org" in stats["unique_domains"]
    assert "example.com" in stats["unique_domains"]
    assert len(stats["unique_domains"]) == 2
    assert set(stats["providers"]) == {"arxiv", "brave"}
    assert stats["items_with_doi"] == 1
    assert stats["items_with_arxiv_id"] == 1
    assert stats["items_with_url"] == 2
    assert stats["iterations_represented"] == [0, 1]


def test_compute_evidence_stats_empty_ledger() -> None:
    pkg = _make_package()
    stats = compute_evidence_stats(pkg)

    assert stats["total_items"] == 0
    assert stats["unique_domains"] == []
    assert stats["providers"] == []
    assert stats["items_with_doi"] == 0
    assert stats["items_with_arxiv_id"] == 0
    assert stats["items_with_url"] == 0
    assert stats["iterations_represented"] == []
