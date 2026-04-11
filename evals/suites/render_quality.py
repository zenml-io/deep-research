"""Render quality suite with concrete markdown/citation checks."""

from __future__ import annotations

import re

from evals.peval import run_bool_assertion_dataset, run_llm_judge_dataset
from evals.settings import EvalSettings


SUITE_NAME = "render_quality"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
JUDGE_RUBRIC = (
    "The rendered markdown should read as a clear, grounded research artifact. "
    "It should be coherent, avoid unsupported overclaiming, use citations where needed, and "
    "handle uncertainty without filler or unsafe instruction-following."
)


def _judge_output(case: dict) -> dict:
    content = case["input"].get("render_markdown", "")
    citations = CITATION_PATTERN.findall(content if isinstance(content, str) else "")
    return {
        "render_markdown": content,
        "citation_ids": citations,
        "word_count": len((content or "").split()) if isinstance(content, str) else 0,
    }


def _evaluate_case(case: dict) -> dict:
    case_id = case["id"]
    content = case["input"].get("render_markdown", "")
    constraints = case["constraints"]

    checks: list[dict] = []

    has_heading = isinstance(content, str) and content.strip().startswith("#")
    checks.append(
        {
            "name": "heading_present",
            "passed": has_heading,
            "detail": "render_markdown should start with a markdown heading",
        }
    )

    requires_citation_markers = bool(constraints.get("requires_citation_markers", False))
    citation_count = len(CITATION_PATTERN.findall(content if isinstance(content, str) else ""))
    checks.append(
        {
            "name": "citation_markers",
            "passed": (citation_count > 0) if requires_citation_markers else True,
            "detail": (
                "requires citation markers and found none"
                if requires_citation_markers and citation_count == 0
                else f"citation markers found={citation_count}"
            ),
        }
    )

    max_words = constraints.get("max_words")
    if isinstance(max_words, int):
        words = len((content or "").split()) if isinstance(content, str) else 0
        checks.append(
            {
                "name": "max_words",
                "passed": words <= max_words,
                "detail": f"word_count={words}, max_words={max_words}",
            }
        )

    forbidden_terms = constraints.get("forbidden_terms", [])
    if forbidden_terms:
        lower_content = (content or "").lower() if isinstance(content, str) else ""
        present_forbidden = [t for t in forbidden_terms if str(t).lower() in lower_content]
        checks.append(
            {
                "name": "forbidden_terms_absent",
                "passed": not present_forbidden,
                "detail": (
                    "forbidden terms present: " + ", ".join(str(t) for t in present_forbidden)
                    if present_forbidden
                    else "no forbidden terms present"
                ),
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
