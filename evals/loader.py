"""Dataset loading and minimal schema validation for offline evals."""

from __future__ import annotations

from pathlib import Path
import json


REQUIRED_CASE_KEYS = ("id", "input", "constraints")


def _validate_case_shape(case: object, dataset_path: Path, index: int) -> dict:
    if not isinstance(case, dict):
        raise ValueError(
            f"Invalid case at {dataset_path}:{index}. Expected object, got {type(case).__name__}."
        )

    for key in REQUIRED_CASE_KEYS:
        if key not in case:
            raise ValueError(
                f"Invalid case at {dataset_path}:{index}. Missing required key '{key}'."
            )

    if not isinstance(case["id"], str) or not case["id"].strip():
        raise ValueError(f"Invalid case at {dataset_path}:{index}. 'id' must be a non-empty string.")

    if not isinstance(case["input"], dict):
        raise ValueError(f"Invalid case at {dataset_path}:{index}. 'input' must be an object.")

    if not isinstance(case["constraints"], dict):
        raise ValueError(f"Invalid case at {dataset_path}:{index}. 'constraints' must be an object.")

    return case


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load a JSON dataset file and validate a minimal common case schema."""
    with dataset_path.open("r", encoding="utf-8") as f:
        parsed = json.load(f)

    if not isinstance(parsed, list):
        raise ValueError(f"Invalid dataset {dataset_path}. Expected a top-level array.")

    return [_validate_case_shape(case, dataset_path=dataset_path, index=i) for i, case in enumerate(parsed)]
