import sys
from inspect import signature
from pathlib import Path
from types import ModuleType

import pytest

from deep_research.models import (
    EvidenceLedger,
    EvidenceCandidate,
    InvestigationPackage,
    IterationRecord,
    IterationTrace,
    RenderPayload,
    ResearchPlan,
    RunSummary,
    SelectionItem,
    SelectionGraph,
)
from deep_research.package.assembly import assemble_package
from deep_research.package.io import (
    read_package,
    write_full_report,
    write_markdown,
    write_package,
)


def make_package(
    *, run_id: str = "run-1", render_names: list[str] | None = None
) -> InvestigationPackage:
    names = render_names or ["reading_path"]

    return InvestigationPackage(
        run_summary={
            "run_id": run_id,
            "brief": "brief",
            "tier": "standard",
            "stop_reason": "max_iterations",
            "status": "completed",
            "estimated_cost_usd": 4.25,
            "elapsed_seconds": 93,
            "iteration_count": 1,
            "provider_usage_summary": {"openai:gpt-5": 2},
            "council_enabled": True,
            "council_size": 2,
            "council_models": ["openai:gpt-5", "anthropic:claude-sonnet-4.5"],
            "started_at": "2026-04-01T10:00:00Z",
            "completed_at": "2026-04-01T10:01:33Z",
        },
        research_plan={
            "goal": "goal",
            "key_questions": ["What changed?"],
            "subtopics": ["overview"],
            "queries": ["phase one package surface"],
            "sections": ["Summary"],
            "success_criteria": ["Round-trip package data"],
            "query_groups": {"primary": ["phase one package surface"]},
            "allowed_source_groups": ["web"],
            "approval_status": "approved",
        },
        evidence_ledger={
            "entries": [
                {
                    "key": "source-1",
                    "title": "Source 1",
                    "url": "https://example.com/source-1",
                    "snippets": [
                        {
                            "text": "A supporting fact.",
                            "source_locator": "p.1",
                        }
                    ],
                    "provider": "example",
                    "source_kind": "web",
                    "quality_score": 0.8,
                    "relevance_score": 0.9,
                    "selected": True,
                }
            ]
        },
        selection_graph={
            "items": [
                {
                    "candidate_key": "source-1",
                    "rationale": "Covers the primary question.",
                }
            ]
        },
        iteration_trace={
            "iterations": [
                {
                    "iteration": 0,
                    "new_candidate_count": 1,
                    "coverage": 0.5,
                    "estimated_cost_usd": 1.75,
                }
            ]
        },
        renders=[
            {
                "name": name,
                "content_markdown": f"# {name}",
                "citation_map": {},
                "structured_content": {"slug": name},
                "generated_at": "2026-04-01T10:01:33Z",
            }
            for name in names
        ],
    )


def test_write_markdown_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"

    write_markdown("# Title", path)

    assert path.read_text() == "# Title"


def test_write_package_materializes_phase_one_views(tmp_path: Path) -> None:
    sample_package = make_package()

    run_dir = write_package(sample_package, tmp_path)

    assert (run_dir / "renders" / "reading_path.md").read_text(encoding="utf-8") == (
        "# reading_path"
    )
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "plan.json").read_text(encoding="utf-8") == (
        sample_package.research_plan.model_dump_json(indent=2)
    )
    assert (run_dir / "plan.md").read_text(encoding="utf-8") == (
        "# Research Plan\n\n"
        "Goal: goal\n\n"
        "Approval status: approved\n\n"
        "## Key Questions\n"
        "- What changed?\n\n"
        "## Query Groups\n"
        "- primary\n"
        "  - phase one package surface\n"
    )
    assert (run_dir / "evidence" / "ledger.json").read_text(encoding="utf-8") == (
        sample_package.evidence_ledger.model_dump_json(indent=2)
    )
    assert (run_dir / "evidence" / "ledger.md").read_text(encoding="utf-8") == (
        "# Evidence Ledger\n\n"
        "Entries: 1\n\n"
        "## source-1\n"
        "- Title: Source 1\n"
        "- URL: https://example.com/source-1\n"
        "- Selected: yes\n"
    )
    assert (run_dir / "iterations" / "000.json").read_text(encoding="utf-8") == (
        sample_package.iteration_trace.iterations[0].model_dump_json(indent=2)
    )


def test_write_full_report_materializes_lazy_render_from_package_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_package = make_package(render_names=[])
    run_dir = write_package(sample_package, tmp_path)
    renderer_module = ModuleType("deep_research.renderers.full_report")

    assert signature(write_full_report).return_annotation is RenderPayload

    def render_full_report(package: InvestigationPackage) -> RenderPayload:
        assert package == sample_package
        return RenderPayload(
            name="full_report",
            content_markdown="# Full Report\n\nFrom package state.",
            citation_map={"source-1": "https://example.com/source-1"},
            structured_content={"sections": ["summary"]},
            generated_at="2026-04-01T10:02:00Z",
        )

    renderer_module.render_full_report = render_full_report
    monkeypatch.setitem(
        sys.modules,
        "deep_research.renderers.full_report",
        renderer_module,
    )

    render = write_full_report(sample_package, run_dir)

    assert render.name == "full_report"
    assert (run_dir / "renders" / "full_report.md").read_text(encoding="utf-8") == (
        "# Full Report\n\nFrom package state."
    )


def test_write_full_report_rejects_unexpected_lazy_render_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample_package = make_package(render_names=[])
    run_dir = write_package(sample_package, tmp_path)
    renderer_module = ModuleType("deep_research.renderers.full_report")

    def render_full_report(package: InvestigationPackage) -> RenderPayload:
        assert package == sample_package
        return RenderPayload(
            name="backing_report",
            content_markdown="# Wrong Report",
            citation_map={},
        )

    renderer_module.render_full_report = render_full_report
    monkeypatch.setitem(
        sys.modules,
        "deep_research.renderers.full_report",
        renderer_module,
    )

    with pytest.raises(ValueError, match="Expected render.name to be 'full_report'"):
        write_full_report(sample_package, run_dir)


def test_write_and_read_package_round_trip(tmp_path: Path) -> None:
    sample_package = make_package()

    run_dir = write_package(sample_package, tmp_path)
    restored = read_package(run_dir)

    assert restored == sample_package


@pytest.mark.parametrize("run_id", ["", ".", "..", "run/1", "run\\1"])
def test_write_package_rejects_invalid_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe path component"):
        write_package(make_package(run_id=run_id), tmp_path)


@pytest.mark.parametrize(
    "render_name", ["", ".", "..", "notes/part-1", "notes\\part-1"]
)
def test_write_package_rejects_invalid_render_name(
    tmp_path: Path, render_name: str
) -> None:
    with pytest.raises(ValueError, match="Unsafe path component"):
        write_package(make_package(render_names=[render_name]), tmp_path)


def test_write_package_rejects_duplicate_render_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate render output filename"):
        write_package(
            make_package(render_names=["reading_path", "reading_path"]), tmp_path
        )


def test_write_package_rejects_case_insensitive_duplicate_render_names(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Duplicate render output filename"):
        write_package(
            make_package(render_names=["Reading_Path", "reading_path"]), tmp_path
        )


def test_write_package_rejects_reserved_full_report_render_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate render output filename"):
        write_package(
            make_package(render_names=["reading_path", "full_report"]), tmp_path
        )

    with pytest.raises(ValueError, match="Duplicate render output filename"):
        write_package(
            make_package(render_names=["reading_path", "Full_Report"]), tmp_path
        )


def test_write_package_rewrites_stale_render_and_iteration_files(
    tmp_path: Path,
) -> None:
    initial_package = make_package(render_names=["reading_path", "backing_report"])
    rewritten_package = make_package(render_names=["reading_path"])
    rewritten_package.iteration_trace = IterationTrace(iterations=[])

    run_dir = write_package(initial_package, tmp_path)

    assert (run_dir / "renders" / "backing_report.md").exists()
    assert (run_dir / "iterations" / "000.json").exists()

    write_package(rewritten_package, tmp_path)

    assert (run_dir / "renders" / "reading_path.md").exists()
    assert not (run_dir / "renders" / "backing_report.md").exists()
    assert not (run_dir / "iterations" / "000.json").exists()


def test_assemble_package_builds_investigation_package() -> None:
    package = assemble_package(
        run_summary=RunSummary(
            run_id="run-1",
            brief="brief",
            tier="standard",
            stop_reason="max_iterations",
            status="completed",
            estimated_cost_usd=4.25,
            elapsed_seconds=93,
            iteration_count=1,
            provider_usage_summary={"openai:gpt-5": 2},
            council_enabled=True,
            council_size=2,
            council_models=["openai:gpt-5", "anthropic:claude-sonnet-4.5"],
            started_at="2026-04-01T10:00:00Z",
            completed_at="2026-04-01T10:01:33Z",
        ),
        research_plan=ResearchPlan(
            goal="goal",
            key_questions=["What changed?"],
            subtopics=["overview"],
            queries=["phase one package surface"],
            sections=["Summary"],
            success_criteria=["Round-trip package data"],
            query_groups={"primary": ["phase one package surface"]},
            allowed_source_groups=["web"],
            approval_status="approved",
        ),
        evidence_ledger=EvidenceLedger(
            entries=[
                EvidenceCandidate(
                    key="source-1",
                    title="Source 1",
                    url="https://example.com/source-1",
                    provider="example",
                    source_kind="web",
                    quality_score=0.8,
                    relevance_score=0.9,
                    selected=True,
                )
            ]
        ),
        selection_graph=SelectionGraph(
            items=[
                SelectionItem(
                    candidate_key="source-1",
                    rationale="Covers the primary question.",
                )
            ]
        ),
        iteration_trace=IterationTrace(
            iterations=[
                IterationRecord(
                    iteration=0,
                    new_candidate_count=1,
                    coverage=0.5,
                    estimated_cost_usd=1.75,
                )
            ]
        ),
        renders=[
            RenderPayload(
                name="reading_path",
                content_markdown="# Reading Path",
                citation_map={},
                structured_content={"slug": "reading_path"},
                generated_at="2026-04-01T10:01:33Z",
            )
        ],
    )

    assert isinstance(package, InvestigationPackage)
    assert package.run_summary.run_id == "run-1"
    assert package.renders[0].name == "reading_path"
