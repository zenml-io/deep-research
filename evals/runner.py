"""CLI runner for offline eval harness baseline suites."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from evals.loader import load_dataset
from evals.logfire import bootstrap_logfire
from evals.settings import EvalSettings
from evals.suites import SUITE_REGISTRY

try:
    from evals.suites import web_regression as _web_regression
    _WEB_REGRESSION_AVAILABLE = True
except ImportError:
    _web_regression = None  # type: ignore[assignment]
    _WEB_REGRESSION_AVAILABLE = False


if _WEB_REGRESSION_AVAILABLE and _web_regression is not None:
    SUITE_REGISTRY = {
        **SUITE_REGISTRY,
        _web_regression.SUITE_NAME: _web_regression.run,
    }
else:
    SUITE_REGISTRY = {**SUITE_REGISTRY}


def _write_baseline(summary: dict, output_dir: Path, baseline_name: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = baseline_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    baseline_path = output_dir / f"baseline-{stamp}.json"
    baseline_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_path = output_dir / "baseline-latest.json"
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return baseline_path


def run_selected_suites(
    selected_suites: list[str],
    dataset_root: Path,
    max_cases: int | None = None,
    settings: EvalSettings | None = None,
) -> dict:
    settings = settings or EvalSettings()
    suites_out: list[dict] = []

    for suite_name in selected_suites:
        run_suite = SUITE_REGISTRY.get(suite_name)
        if run_suite is None:
            raise ValueError(f"Unknown suite '{suite_name}'. Available: {sorted(SUITE_REGISTRY)}")

        dataset_path = dataset_root / f"{suite_name}.json"
        if not dataset_path.exists():
            suites_out.append(
                {
                    "suite": suite_name,
                    "status": "skipped_missing_dataset",
                    "dataset": str(dataset_path),
                    "total_cases": 0,
                    "failed_cases": 0,
                    "failures": [],
                }
            )
            continue

        cases = load_dataset(dataset_path)
        if max_cases is not None:
            cases = cases[:max_cases]

        result = run_suite(cases, settings=settings)
        result["status"] = "failed" if result.get("failed_cases", 0) else "ok"
        result["dataset"] = str(dataset_path)
        suites_out.append(result)

    total_cases = sum(item.get("total_cases", 0) for item in suites_out)
    failed_cases = sum(item.get("failed_cases", 0) for item in suites_out)
    skipped_suites = sum(1 for item in suites_out if item.get("status") == "skipped_missing_dataset")

    return {
        "suites": suites_out,
        "total_suites": len(suites_out),
        "total_cases": total_cases,
        "failed_cases": failed_cases,
        "skipped_suites": skipped_suites,
        "status": "failed" if failed_cases else "ok",
        "use_llm_judge": settings.use_llm_judge,
        "judge_model": settings.judge_model if settings.use_llm_judge else None,
    }


def main() -> int:
    env_settings = EvalSettings.from_env()

    parser = argparse.ArgumentParser(description="Run offline eval harness baseline suites.")
    parser.add_argument("--suite", action="append", help="Suite name (repeatable)")
    parser.add_argument("--dataset-root", type=Path, help="Path to dataset directory")
    parser.add_argument("--max-cases", type=int, help="Optional max cases per suite")
    parser.add_argument("--list-suites", action="store_true", help="List available suites and exit")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write suite summary JSON baseline artifact",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        help="Optional baseline suffix name (otherwise UTC timestamp)",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable opt-in LLMJudge scoring for supported suites",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Override the LLMJudge model (default: openai:gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-max-concurrency",
        type=int,
        help="Max concurrency for LLMJudge runs",
    )
    parser.add_argument(
        "--disable-judge-reasons",
        action="store_true",
        help="Skip judge reason strings in summary output",
    )
    parser.add_argument(
        "--enable-logfire",
        action="store_true",
        help="Enable Logfire export for eval runs when configured",
    )
    parser.add_argument(
        "--logfire-environment",
        type=str,
        help="Override Logfire environment name for eval runs",
    )
    args = parser.parse_args()

    if args.list_suites:
        print("\n".join(sorted(SUITE_REGISTRY)))
        return 0

    settings = EvalSettings(
        suites=env_settings.suites,
        dataset_root=env_settings.dataset_root,
        output_dir=env_settings.output_dir,
        max_cases=env_settings.max_cases,
        use_llm_judge=args.use_llm_judge or env_settings.use_llm_judge,
        judge_model=args.judge_model or env_settings.judge_model,
        judge_include_reason=(
            False if args.disable_judge_reasons else env_settings.judge_include_reason
        ),
        judge_max_concurrency=(
            args.judge_max_concurrency
            if args.judge_max_concurrency is not None
            else env_settings.judge_max_concurrency
        ),
        enable_logfire=args.enable_logfire or env_settings.enable_logfire,
        logfire_service_name=env_settings.logfire_service_name,
        logfire_environment=args.logfire_environment or env_settings.logfire_environment,
    )

    if settings.enable_logfire or os.getenv("LOGFIRE_TOKEN"):
        bootstrap_logfire(settings)

    selected_suites = args.suite or list(settings.suites)
    dataset_root = args.dataset_root or settings.dataset_root
    max_cases = args.max_cases if args.max_cases is not None else settings.max_cases

    summary = run_selected_suites(
        selected_suites=selected_suites,
        dataset_root=dataset_root,
        max_cases=max_cases,
        settings=settings,
    )

    if args.write_baseline:
        baseline_path = _write_baseline(
            summary=summary,
            output_dir=settings.output_dir,
            baseline_name=args.baseline_name,
        )
        summary["baseline_file"] = str(baseline_path)
        summary["baseline_latest"] = str(settings.output_dir / "baseline-latest.json")

    print(json.dumps(summary, indent=2))
    return 1 if summary["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
