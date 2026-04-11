"""Optional pydantic-evals integration helpers for eval suites."""

from __future__ import annotations

from typing import Any, Callable

try:
    from pydantic_ai import ModelSettings
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EqualsExpected, LLMJudge
except Exception:  # pragma: no cover - optional dependency
    Case = None
    Dataset = None
    EqualsExpected = None
    LLMJudge = None
    ModelSettings = None


def pydantic_evals_available() -> bool:
    return Case is not None and Dataset is not None and EqualsExpected is not None


def llm_judge_available() -> bool:
    return pydantic_evals_available() and LLMJudge is not None and ModelSettings is not None


def run_bool_assertion_dataset(
    suite_name: str,
    cases: list[dict],
    assertion_fn: Callable[[dict], bool],
) -> dict:
    """Run a boolean assertion dataset via pydantic-evals when available.

    Falls back to native execution metadata when the optional dependency is absent.
    """
    if not pydantic_evals_available():
        return {
            "engine": "native",
            "pydantic_evals_available": False,
            "guidance": "Install eval extras with: uv sync --extra evals",
        }

    pe_cases = [
        Case(
            name=str(case["id"]),
            inputs=case,
            expected_output=True,
        )
        for case in cases
    ]
    report = Dataset(name=suite_name, cases=pe_cases, evaluators=[EqualsExpected()]).evaluate_sync(
        assertion_fn,
        progress=False,
    )

    failures: list[str] = []
    for report_case in report.cases:
        assertion = report_case.assertions.get("EqualsExpected")
        if assertion is None or not bool(assertion.value):
            failures.append(report_case.name)

    return {
        "engine": "pydantic-evals",
        "pydantic_evals_available": True,
        "pydantic_evals_failures": failures,
    }


def run_llm_judge_dataset(
    *,
    suite_name: str,
    cases: list[dict],
    task_fn: Callable[[dict], Any],
    rubric: str,
    include_input: bool,
    include_expected_output: bool,
    judge_model: str,
    include_reason: bool,
    max_concurrency: int | None,
    experiment_metadata: dict[str, Any] | None = None,
) -> dict:
    """Run an opt-in LLMJudge dataset and normalize report details for the harness."""
    if not llm_judge_available():
        return {
            "judge_mode": "skipped_unavailable",
            "llm_judge_available": False,
            "guidance": "Install eval extras with: uv sync --extra evals",
        }

    pe_cases = [
        Case(
            name=str(case["id"]),
            inputs=case,
            expected_output=case.get("expected_output"),
            metadata={
                "constraints": case.get("constraints", {}),
                "forbidden_behaviors": case.get("forbidden_behaviors", []),
            },
        )
        for case in cases
    ]
    judge = LLMJudge(
        rubric=rubric,
        model=judge_model,
        include_input=include_input,
        include_expected_output=include_expected_output,
        model_settings=ModelSettings(temperature=0.0),
        score={"evaluation_name": "judge_score", "include_reason": include_reason},
        assertion={"evaluation_name": "judge_pass", "include_reason": include_reason},
    )
    report = Dataset(name=suite_name, cases=pe_cases, evaluators=[judge]).evaluate_sync(
        task_fn,
        max_concurrency=max_concurrency,
        progress=False,
        task_name=f"{suite_name}_judge",
        metadata=experiment_metadata,
    )

    case_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for report_case in report.cases:
        assertions = getattr(report_case, "assertions", {}) or {}
        scores = getattr(report_case, "scores", {}) or {}

        assertion = assertions.get("judge_pass")
        score = scores.get("judge_score")
        judge_pass = None if assertion is None else bool(assertion.value)
        judge_reason = None
        if score is not None and getattr(score, "reason", None):
            judge_reason = score.reason
        elif assertion is not None and getattr(assertion, "reason", None):
            judge_reason = assertion.reason

        score_value = None if score is None else score.value
        case_results.append(
            {
                "id": report_case.name,
                "judge_pass": judge_pass,
                "judge_score": score_value,
                "judge_reason": judge_reason,
            }
        )
        if judge_pass is False:
            message = f"{report_case.name}: judge rubric failed"
            if judge_reason:
                message += f" ({judge_reason})"
            failures.append(message)

    return {
        "judge_mode": "enabled",
        "llm_judge_available": True,
        "judge_model": judge_model,
        "judge_failures": failures,
        "judge_results": case_results,
    }
