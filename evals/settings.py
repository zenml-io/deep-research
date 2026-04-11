"""Eval harness settings (dev-only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


DEFAULT_SUITES: tuple[str, ...] = (
    "brief_to_plan",
    "supervisor_trace_and_safety",
    "render_quality",
)


@dataclass(frozen=True)
class EvalSettings:
    """Settings for offline eval harness execution."""

    suites: tuple[str, ...] = DEFAULT_SUITES
    dataset_root: Path = Path(__file__).resolve().parent / "datasets"
    output_dir: Path = Path("artifacts/evals")
    max_cases: int | None = None
    use_llm_judge: bool = False
    judge_model: str = "openai:gpt-4o-mini"
    judge_include_reason: bool = True
    judge_max_concurrency: int | None = 3
    enable_logfire: bool = False
    logfire_service_name: str = "deep-research-evals"
    logfire_environment: str = "development"

    @classmethod
    def from_env(cls) -> "EvalSettings":
        suites_raw = os.getenv("EVAL_SUITES", "").strip()
        suites = tuple(s.strip() for s in suites_raw.split(",") if s.strip()) or DEFAULT_SUITES

        dataset_root = Path(
            os.getenv("EVAL_DATASET_ROOT", str(Path(__file__).resolve().parent / "datasets"))
        )
        output_dir = Path(os.getenv("EVAL_OUTPUT_DIR", "artifacts/evals"))

        max_cases_raw = os.getenv("EVAL_MAX_CASES", "").strip()
        max_cases = int(max_cases_raw) if max_cases_raw else None

        use_llm_judge = os.getenv("EVAL_USE_LLM_JUDGE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        judge_model = os.getenv("EVAL_JUDGE_MODEL", "openai:gpt-4o-mini").strip() or "openai:gpt-4o-mini"
        judge_include_reason = os.getenv("EVAL_JUDGE_INCLUDE_REASON", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

        judge_max_concurrency_raw = os.getenv("EVAL_JUDGE_MAX_CONCURRENCY", "").strip()
        judge_max_concurrency = (
            int(judge_max_concurrency_raw) if judge_max_concurrency_raw else 3
        )

        enable_logfire = os.getenv("EVAL_ENABLE_LOGFIRE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        logfire_service_name = (
            os.getenv("EVAL_LOGFIRE_SERVICE_NAME", "deep-research-evals").strip()
            or "deep-research-evals"
        )
        logfire_environment = (
            os.getenv("EVAL_LOGFIRE_ENVIRONMENT", "development").strip() or "development"
        )

        return cls(
            suites=suites,
            dataset_root=dataset_root,
            output_dir=output_dir,
            max_cases=max_cases,
            use_llm_judge=use_llm_judge,
            judge_model=judge_model,
            judge_include_reason=judge_include_reason,
            judge_max_concurrency=judge_max_concurrency,
            enable_logfire=enable_logfire,
            logfire_service_name=logfire_service_name,
            logfire_environment=logfire_environment,
        )
