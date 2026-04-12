"""Trajectory-quality suite driven by committed package-artifact fixtures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.enums import SourceKind, StopReason
from deep_research.models import InvestigationPackage

from evals.peval import run_bool_assertion_dataset, run_llm_judge_dataset
from evals.settings import EvalSettings


SUITE_NAME = "trajectory_quality"
JUDGE_RUBRIC = (
    "The package artifact should describe a coherent end-to-end research trajectory. "
    "It should satisfy the declared stop reason, evidence-count threshold, coverage "
    "threshold, plan-fidelity threshold, and source-diversity requirements. The final "
    "render should read as a grounded research artifact rather than an unsupported summary."
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "and",
    "are",
    "behave",
    "does",
    "doesnt",
    "final",
    "first",
    "from",
    "how",
    "into",
    "is",
    "it",
    "its",
    "loop",
    "more",
    "of",
    "plan",
    "question",
    "questions",
    "render",
    "replay",
    "source",
    "sources",
    "the",
    "this",
    "what",
    "which",
    "why",
    "when",
    "with",
}


def _case_id(case: dict) -> str:
    return str(case.get("id", "unknown"))


def _case_input(case: dict) -> dict:
    inputs = case.get("input")
    if not isinstance(inputs, dict):
        raise ValueError(f"{_case_id(case)}: input must be an object")
    return inputs


def _case_constraints(case: dict) -> dict:
    constraints = case.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError(f"{_case_id(case)}: constraints must be an object")
    return constraints


def _resolve_artifact_path(artifact_value: str) -> Path:
    path = Path(artifact_value)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


@lru_cache(maxsize=32)
def _load_package_artifact(artifact_value: str) -> InvestigationPackage:
    path = _resolve_artifact_path(artifact_value)
    if not path.exists():
        raise ValueError(f"Trajectory package artifact not found: {path}")
    try:
        return InvestigationPackage.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid trajectory package artifact {path}: {exc}") from exc


def _selected_entries(package: InvestigationPackage):
    return resolve_selected_entries(package.evidence_ledger)


def _corpus_text(package: InvestigationPackage) -> str:
    parts: list[str] = []
    parts.extend(render.content_markdown for render in package.renders)
    for candidate in _selected_entries(package):
        parts.append(candidate.title)
        parts.extend(snippet.text for snippet in candidate.snippets)
        parts.extend(candidate.matched_subtopics)
    for item in package.selection_graph.items:
        parts.append(item.rationale)
        if item.bridge_note:
            parts.append(item.bridge_note)
        if item.ordering_rationale:
            parts.append(item.ordering_rationale)
    return " ".join(part.lower() for part in parts if part)


def _question_tokens(question: str) -> list[str]:
    return [
        token
        for token in _WORD_RE.findall(question.lower())
        if len(token) >= 4 and token not in _STOPWORDS
    ]


def _plan_fidelity(package: InvestigationPackage) -> tuple[float, list[str]]:
    questions = [question for question in package.research_plan.key_questions if question.strip()]
    if not questions:
        return 0.0, []

    corpus = _corpus_text(package)
    unanswered: list[str] = []
    answered = 0
    for question in questions:
        tokens = _question_tokens(question)
        if tokens and any(token in corpus for token in tokens):
            answered += 1
            continue
        unanswered.append(question)

    return round(answered / len(questions), 4), unanswered


def _normalize_stop_reason(value: str | StopReason) -> str:
    return value.value if isinstance(value, StopReason) else str(value)


def _evaluate_case(case: dict) -> dict:
    case_id = _case_id(case)
    inputs = _case_input(case)
    constraints = _case_constraints(case)

    artifact_value = inputs.get("package_artifact")
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        raise ValueError(f"{case_id}: input.package_artifact must be a non-empty string")

    package = _load_package_artifact(artifact_value)
    selected_entries = list(_selected_entries(package))
    final_iteration = package.iteration_trace.iterations[-1] if package.iteration_trace.iterations else None
    coverage_total = final_iteration.coverage if final_iteration is not None else 0.0
    plan_fidelity, unanswered_questions = _plan_fidelity(package)
    selected_source_kinds = sorted({candidate.source_kind.value for candidate in selected_entries})
    selected_providers = sorted({candidate.provider for candidate in selected_entries})
    render_names = [render.name for render in package.renders]

    checks: list[dict] = []

    expected_stop_reason = constraints.get("expected_stop_reason")
    if expected_stop_reason is not None:
        checks.append(
            {
                "name": "stop_reason",
                "passed": _normalize_stop_reason(package.run_summary.stop_reason)
                == _normalize_stop_reason(expected_stop_reason),
                "detail": (
                    f"expected stop_reason={expected_stop_reason}, "
                    f"actual={package.run_summary.stop_reason.value}"
                ),
            }
        )

    min_selected = constraints.get("min_selected_evidence_count")
    if isinstance(min_selected, int):
        checks.append(
            {
                "name": "selected_evidence_count",
                "passed": len(selected_entries) >= min_selected,
                "detail": f"selected_count={len(selected_entries)}, min_selected={min_selected}",
            }
        )

    min_coverage = constraints.get("min_coverage_total")
    if isinstance(min_coverage, (int, float)):
        checks.append(
            {
                "name": "coverage_total",
                "passed": coverage_total >= float(min_coverage),
                "detail": f"coverage_total={coverage_total:.4f}, min_coverage_total={float(min_coverage):.4f}",
            }
        )

    min_plan_fidelity = constraints.get("min_plan_fidelity")
    if isinstance(min_plan_fidelity, (int, float)):
        checks.append(
            {
                "name": "plan_fidelity",
                "passed": plan_fidelity >= float(min_plan_fidelity),
                "detail": f"plan_fidelity={plan_fidelity:.4f}, min_plan_fidelity={float(min_plan_fidelity):.4f}",
            }
        )

    required_source_kinds = constraints.get("required_source_kinds", [])
    if required_source_kinds:
        normalized_required = []
        for kind in required_source_kinds:
            try:
                normalized_required.append(SourceKind(kind).value)
            except Exception as exc:
                raise ValueError(
                    f"{case_id}: invalid required_source_kinds entry {kind!r}"
                ) from exc
        missing_source_kinds = [
            kind for kind in normalized_required if kind not in selected_source_kinds
        ]
        checks.append(
            {
                "name": "required_source_kinds",
                "passed": not missing_source_kinds,
                "detail": (
                    "missing source kinds: " + ", ".join(missing_source_kinds)
                    if missing_source_kinds
                    else f"selected_source_kinds={selected_source_kinds}"
                ),
            }
        )

    min_unique_providers = constraints.get("min_unique_providers")
    if isinstance(min_unique_providers, int):
        checks.append(
            {
                "name": "provider_diversity",
                "passed": len(selected_providers) >= min_unique_providers,
                "detail": (
                    f"unique_providers={len(selected_providers)}, "
                    f"min_unique_providers={min_unique_providers}"
                ),
            }
        )

    allow_unanswered_questions = bool(constraints.get("allow_unanswered_questions", True))
    if not allow_unanswered_questions:
        checks.append(
            {
                "name": "unanswered_questions",
                "passed": not unanswered_questions,
                "detail": (
                    "unanswered questions: " + ", ".join(unanswered_questions)
                    if unanswered_questions
                    else "all key questions were answered"
                ),
            }
        )

    checks.append(
        {
            "name": "render_present",
            "passed": bool(package.renders),
            "detail": f"render_names={render_names}",
        }
    )

    passed = all(check["passed"] for check in checks)
    return {
        "id": case_id,
        "passed": passed,
        "checks": checks,
        "package_artifact": str(_resolve_artifact_path(artifact_value)),
        "selected_evidence_count": len(selected_entries),
        "coverage_total": coverage_total,
        "plan_fidelity": plan_fidelity,
        "unanswered_questions": unanswered_questions,
        "selected_source_kinds": selected_source_kinds,
        "selected_providers": selected_providers,
        "render_names": render_names,
    }


def _judge_output(case: dict) -> dict:
    result = _evaluate_case(case)
    return {
        "package_artifact": result["package_artifact"],
        "selected_evidence_count": result["selected_evidence_count"],
        "coverage_total": result["coverage_total"],
        "plan_fidelity": result["plan_fidelity"],
        "unanswered_questions": result["unanswered_questions"],
        "selected_source_kinds": result["selected_source_kinds"],
        "selected_providers": result["selected_providers"],
        "render_names": result["render_names"],
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
