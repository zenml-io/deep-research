"""Web-first regression suite for grounded comparison-style research artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from deep_research.evidence.resolution import resolve_selected_entries
from deep_research.enums import SourceGroup, SourceKind
from deep_research.models import InvestigationPackage

from evals.peval import run_bool_assertion_dataset, run_llm_judge_dataset
from evals.settings import EvalSettings


SUITE_NAME = "web_regression"
JUDGE_RUBRIC = (
    "The artifact should look like a practical web-first research result for an "
    "engineering question. It should surface concrete named systems, rely on a "
    "diverse web/docs/repos evidence mix, avoid generic academic filler, and keep "
    "unsupported claims low when claim inventory is available."
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
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


@lru_cache(maxsize=16)
def _load_package_artifact(artifact_value: str) -> InvestigationPackage:
    path = _resolve_artifact_path(artifact_value)
    if not path.exists():
        raise ValueError(f"Web regression package artifact not found: {path}")
    try:
        return InvestigationPackage.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid web regression package artifact {path}: {exc}") from exc


def _selected_entries(package: InvestigationPackage):
    return resolve_selected_entries(package.evidence_ledger)


def _render_markdown(package: InvestigationPackage) -> str:
    return "\n\n".join(render.content_markdown for render in package.renders if render.content_markdown)


def _corpus_text(package: InvestigationPackage) -> str:
    parts = [_render_markdown(package), package.run_summary.brief, package.research_plan.goal]
    for candidate in _selected_entries(package):
        parts.append(candidate.title)
        parts.extend(snippet.text for snippet in candidate.snippets)
        parts.extend(candidate.matched_subtopics)
    for item in package.selection_graph.items:
        parts.append(item.rationale)
        if item.bridge_note:
            parts.append(item.bridge_note)
    return " ".join(part for part in parts if part).lower()


def _selected_source_groups(selected_entries: list) -> list[str]:
    groups: set[str] = set()
    for candidate in selected_entries:
        source_group = getattr(candidate, "source_group", None)
        if source_group is not None:
            groups.add(source_group.value if hasattr(source_group, "value") else str(source_group))
            continue
        if candidate.source_kind == SourceKind.PAPER:
            groups.add(SourceGroup.PAPERS.value)
        elif candidate.source_kind == SourceKind.DOCS:
            groups.add(SourceGroup.DOCS.value)
        elif candidate.source_kind == SourceKind.REPOSITORY:
            groups.add(SourceGroup.REPOS.value)
        elif candidate.source_kind == SourceKind.BLOG:
            groups.add(SourceGroup.BLOGS.value)
        elif candidate.source_kind == SourceKind.FORUM:
            groups.add(SourceGroup.FORUMS.value)
        elif candidate.source_kind == SourceKind.BENCHMARK:
            groups.add(SourceGroup.BENCHMARKS.value)
        else:
            groups.add(SourceGroup.WEB.value)
    return sorted(groups)


def _paper_ratio(selected_entries: list) -> float:
    if not selected_entries:
        return 1.0
    paper_count = 0
    for candidate in selected_entries:
        source_group = getattr(candidate, "source_group", None)
        if source_group == SourceGroup.PAPERS or candidate.source_kind == SourceKind.PAPER:
            paper_count += 1
    return round(paper_count / len(selected_entries), 4)


def _claim_inventory_stats(package: InvestigationPackage) -> dict[str, object]:
    claim_inventory = getattr(package, "claim_inventory", None)
    if claim_inventory is None:
        return {"present": False, "unsupported_ratio": None, "total_claims": 0}
    claims = getattr(claim_inventory, "claims", [])
    unsupported_ratio = getattr(claim_inventory, "unsupported_ratio", None)
    if unsupported_ratio is None and claims:
        unsupported_count = sum(
            1
            for claim in claims
            if getattr(claim, "support_status", None) in {"unsupported", "weak"}
        )
        unsupported_ratio = round(unsupported_count / len(claims), 4)
    return {
        "present": True,
        "unsupported_ratio": unsupported_ratio,
        "total_claims": getattr(claim_inventory, "total_claims", len(claims)),
    }


def _section_headings(package: InvestigationPackage) -> list[str]:
    headings: list[str] = []
    for render in package.renders:
        for line in render.content_markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                headings.append(stripped.lstrip("#").strip().lower())
    return headings


def _evaluate_case(case: dict) -> dict:
    case_id = _case_id(case)
    inputs = _case_input(case)
    constraints = _case_constraints(case)

    artifact_value = inputs.get("package_artifact")
    if not isinstance(artifact_value, str) or not artifact_value.strip():
        raise ValueError(f"{case_id}: input.package_artifact must be a non-empty string")

    package = _load_package_artifact(artifact_value)
    selected_entries = list(_selected_entries(package))
    corpus = _corpus_text(package)
    source_groups = _selected_source_groups(selected_entries)
    selected_providers = sorted({candidate.provider for candidate in selected_entries})
    render_names = [render.name for render in package.renders]
    headings = _section_headings(package)
    claim_stats = _claim_inventory_stats(package)
    checks: list[dict] = []

    expected_entities = constraints.get("expected_entities", [])
    if expected_entities:
        missing_entities = [
            entity for entity in expected_entities if str(entity).lower() not in corpus
        ]
        checks.append(
            {
                "name": "expected_entities",
                "passed": not missing_entities,
                "detail": (
                    "missing entities: " + ", ".join(str(entity) for entity in missing_entities)
                    if missing_entities
                    else "all expected entities present"
                ),
            }
        )

    required_source_groups = constraints.get("required_source_groups", [])
    if required_source_groups:
        normalized_required = []
        for group in required_source_groups:
            try:
                normalized_required.append(SourceGroup(group).value)
            except Exception as exc:
                raise ValueError(
                    f"{case_id}: invalid required_source_groups entry {group!r}"
                ) from exc
        missing_groups = [group for group in normalized_required if group not in source_groups]
        checks.append(
            {
                "name": "required_source_groups",
                "passed": not missing_groups,
                "detail": (
                    "missing source groups: " + ", ".join(missing_groups)
                    if missing_groups
                    else f"selected_source_groups={source_groups}"
                ),
            }
        )

    forbidden_drift_terms = constraints.get("forbidden_drift_terms", [])
    if forbidden_drift_terms:
        present_forbidden = [
            term for term in forbidden_drift_terms if str(term).lower() in corpus
        ]
        checks.append(
            {
                "name": "forbidden_drift_terms",
                "passed": not present_forbidden,
                "detail": (
                    "forbidden drift terms present: "
                    + ", ".join(str(term) for term in present_forbidden)
                    if present_forbidden
                    else "no forbidden drift terms present"
                ),
            }
        )

    min_provider_diversity = constraints.get("min_provider_diversity")
    if isinstance(min_provider_diversity, int):
        checks.append(
            {
                "name": "provider_diversity",
                "passed": len(selected_providers) >= min_provider_diversity,
                "detail": (
                    f"unique_providers={len(selected_providers)}, "
                    f"min_provider_diversity={min_provider_diversity}"
                ),
            }
        )

    max_paper_ratio = constraints.get("max_paper_ratio")
    if isinstance(max_paper_ratio, (int, float)):
        paper_ratio = _paper_ratio(selected_entries)
        checks.append(
            {
                "name": "paper_ratio",
                "passed": paper_ratio <= float(max_paper_ratio),
                "detail": f"paper_ratio={paper_ratio:.4f}, max_paper_ratio={float(max_paper_ratio):.4f}",
            }
        )

    required_output_sections = constraints.get("required_output_sections", [])
    if required_output_sections:
        missing_sections = [
            section
            for section in required_output_sections
            if str(section).lower() not in headings
        ]
        checks.append(
            {
                "name": "required_output_sections",
                "passed": not missing_sections,
                "detail": (
                    "missing sections: " + ", ".join(str(section) for section in missing_sections)
                    if missing_sections
                    else f"headings={headings}"
                ),
            }
        )

    max_unsupported_claim_ratio = constraints.get("max_unsupported_claim_ratio")
    if isinstance(max_unsupported_claim_ratio, (int, float)):
        checks.append(
            {
                "name": "unsupported_claim_ratio",
                "passed": claim_stats["present"]
                and claim_stats["unsupported_ratio"] is not None
                and float(claim_stats["unsupported_ratio"])
                <= float(max_unsupported_claim_ratio),
                "detail": (
                    "claim inventory missing"
                    if not claim_stats["present"]
                    else "unsupported_ratio="
                    f"{claim_stats['unsupported_ratio']}, "
                    f"max_unsupported_claim_ratio={float(max_unsupported_claim_ratio):.4f}"
                ),
            }
        )

    checks.append(
        {
            "name": "render_present",
            "passed": bool(render_names),
            "detail": f"render_names={render_names}",
        }
    )

    passed = all(check["passed"] for check in checks)
    return {
        "id": case_id,
        "passed": passed,
        "checks": checks,
        "package_artifact": str(_resolve_artifact_path(artifact_value)),
        "selected_source_groups": source_groups,
        "selected_providers": selected_providers,
        "claim_stats": claim_stats,
        "render_names": render_names,
    }


def _judge_output(case: dict) -> dict:
    result = _evaluate_case(case)
    return {
        "package_artifact": result["package_artifact"],
        "selected_source_groups": result["selected_source_groups"],
        "selected_providers": result["selected_providers"],
        "claim_stats": result["claim_stats"],
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
