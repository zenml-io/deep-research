from pathlib import Path

import evals.peval as peval
from evals.loader import load_dataset
from evals.runner import run_selected_suites
from evals.settings import EvalSettings


DATASET_ROOT = Path("evals/datasets")


def test_eval_datasets_load() -> None:
    brief_cases = load_dataset(DATASET_ROOT / "brief_to_plan.json")
    supervisor_cases = load_dataset(DATASET_ROOT / "supervisor_trace_and_safety.json")
    render_cases = load_dataset(DATASET_ROOT / "render_quality.json")

    assert len(brief_cases) >= 2
    assert len(supervisor_cases) >= 2
    assert len(render_cases) >= 2


def test_eval_runner_all_baseline_suites_smoke() -> None:
    summary = run_selected_suites(
        selected_suites=[
            "brief_to_plan",
            "supervisor_trace_and_safety",
            "render_quality",
        ],
        dataset_root=DATASET_ROOT,
    )

    assert summary["status"] == "ok"
    assert summary["total_suites"] == 3
    assert summary["failed_cases"] == 0
    assert summary["skipped_suites"] == 0
    assert summary["use_llm_judge"] is False
    assert summary["judge_model"] is None

    suite_names = {suite["suite"] for suite in summary["suites"]}
    assert suite_names == {"brief_to_plan", "supervisor_trace_and_safety", "render_quality"}

    for suite in summary["suites"]:
        assert suite["status"] == "ok"
        assert suite["total_cases"] >= 1
        assert "engine" in suite
        assert "judge" not in suite


def test_eval_runner_sets_judge_mode_without_optional_dependency() -> None:
    summary = run_selected_suites(
        selected_suites=["brief_to_plan"],
        dataset_root=DATASET_ROOT,
        settings=EvalSettings(use_llm_judge=True),
    )

    assert summary["use_llm_judge"] is True
    assert summary["judge_model"] == "openai:gpt-4o-mini"
    judge = summary["suites"][0]["judge"]
    assert judge["judge_mode"] == "skipped_unavailable"
    assert judge["llm_judge_available"] is False
    assert "uv sync --extra evals" in judge["guidance"]


def test_pydantic_evals_optional_fallback(monkeypatch) -> None:
    monkeypatch.setattr(peval, "Case", None)
    monkeypatch.setattr(peval, "Dataset", None)
    monkeypatch.setattr(peval, "EqualsExpected", None)

    fallback = peval.run_bool_assertion_dataset(
        suite_name="fallback_check",
        cases=[{"id": "case_1"}],
        assertion_fn=lambda _case: True,
    )

    assert fallback["engine"] == "native"
    assert fallback["pydantic_evals_available"] is False
    assert "uv sync --extra evals" in fallback["guidance"]


def test_llm_judge_optional_fallback(monkeypatch) -> None:
    monkeypatch.setattr(peval, "Case", None)
    monkeypatch.setattr(peval, "Dataset", None)
    monkeypatch.setattr(peval, "EqualsExpected", None)
    monkeypatch.setattr(peval, "LLMJudge", None)
    monkeypatch.setattr(peval, "ModelSettings", None)

    fallback = peval.run_llm_judge_dataset(
        suite_name="judge_check",
        cases=[{"id": "case_1", "constraints": {}}],
        task_fn=lambda case: case,
        rubric="Output is acceptable",
        include_input=True,
        include_expected_output=False,
        judge_model="openai:gpt-4o-mini",
        include_reason=True,
        max_concurrency=1,
    )

    assert fallback["judge_mode"] == "skipped_unavailable"
    assert fallback["llm_judge_available"] is False
    assert "uv sync --extra evals" in fallback["guidance"]
