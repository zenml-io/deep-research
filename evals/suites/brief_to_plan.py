"""Brief-to-plan baseline suite with concrete rule checks."""

from __future__ import annotations

from evals.peval import run_bool_assertion_dataset, run_llm_judge_dataset
from evals.settings import EvalSettings


SUITE_NAME = "brief_to_plan"
JUDGE_RUBRIC = (
    "The proposed plan framing is appropriate for the research brief. "
    "It should capture the right planning mode, cover the important user intent, avoid redundant "
    "angles, and stay balanced for comparisons or recommendation briefs."
)


def _infer_planning_mode(brief: str) -> str:
    text = brief.lower()
    if "compare" in text or " vs " in text or "versus" in text:
        return "comparison"
    if "timeline" in text or "history" in text:
        return "timeline"
    if "recommend" in text or "decision" in text:
        return "decision_support"
    return "broad_scan"


def _judge_output(case: dict) -> dict:
    brief = case["input"].get("brief", "")
    constraints = case.get("constraints", {})
    return {
        "brief": brief,
        "inferred_planning_mode": _infer_planning_mode(brief) if isinstance(brief, str) else "",
        "comparison_targets": constraints.get("comparison_targets", []),
        "must_include_terms": constraints.get("must_include_terms", []),
        "deliverable_mode": constraints.get("deliverable_mode"),
    }


def _evaluate_case(case: dict) -> dict:
    case_id = case["id"]
    brief = case["input"].get("brief", "")
    constraints = case["constraints"]

    checks: list[dict] = []

    checks.append(
        {
            "name": "brief_non_empty",
            "passed": isinstance(brief, str) and bool(brief.strip()),
            "detail": "input.brief must be a non-empty string",
        }
    )

    expected_mode = constraints.get("planning_mode")
    inferred_mode = _infer_planning_mode(brief) if isinstance(brief, str) else ""
    checks.append(
        {
            "name": "planning_mode_alignment",
            "passed": bool(expected_mode) and inferred_mode == expected_mode,
            "detail": f"expected planning_mode={expected_mode}, inferred={inferred_mode}",
        }
    )

    comparison_targets = constraints.get("comparison_targets", [])
    if expected_mode == "comparison":
        checks.append(
            {
                "name": "comparison_targets_present",
                "passed": isinstance(comparison_targets, list) and len(comparison_targets) >= 2,
                "detail": "comparison mode requires at least 2 comparison_targets",
            }
        )

    required_terms = constraints.get("must_include_terms", [])
    if required_terms:
        lower_brief = brief.lower() if isinstance(brief, str) else ""
        missing_terms = [t for t in required_terms if str(t).lower() not in lower_brief]
        checks.append(
            {
                "name": "required_terms_present",
                "passed": not missing_terms,
                "detail": "missing terms: " + ", ".join(missing_terms) if missing_terms else "all present",
            }
        )

    passed = all(check["passed"] for check in checks)
    return {
        "id": case_id,
        "passed": passed,
        "checks": checks,
    }


def run(cases: list[dict], settings: EvalSettings | None = None) -> dict:
    settings = settings or EvalSettings()

    case_results = [_evaluate_case(case) for case in cases]
    failures = [
        f"{result['id']}: "
        + "; ".join(check["detail"] for check in result["checks"] if not check["passed"])
        for result in case_results
        if not result["passed"]
    ]

    peval = run_bool_assertion_dataset(
        suite_name=SUITE_NAME,
        cases=cases,
        assertion_fn=lambda case: _evaluate_case(case)["passed"],
    )

    result = {
        "suite": SUITE_NAME,
        "total_cases": len(cases),
        "failed_cases": len(failures),
        "failures": failures,
        "case_results": case_results,
        **peval,
    }

    if settings.use_llm_judge:
        result["judge"] = run_llm_judge_dataset(
            suite_name=SUITE_NAME,
            cases=cases,
            task_fn=_judge_output,
            rubric=JUDGE_RUBRIC,
            include_input=True,
            include_expected_output=False,
            judge_model=settings.judge_model,
            include_reason=settings.judge_include_reason,
            max_concurrency=settings.judge_max_concurrency,
            experiment_metadata={"suite": SUITE_NAME},
        )

    return result
