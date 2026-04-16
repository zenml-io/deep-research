"""Thin real-runtime durability coverage using the actual local Kitaru runtime."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
PROBE = REPO_ROOT / "tests" / "_runtime_kitaru_probe.py"


def _runtime_env(tmp_path: Path) -> dict[str, str]:
    """Build an isolated HOME/XDG/PATH env for a real local Kitaru run."""
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["XDG_DATA_HOME"] = str(xdg)
    env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    return env


def _run_probe(tmp_path: Path, scenario: str) -> tuple[dict[str, object], str]:
    """Run a runtime probe in a subprocess and return its JSON payload + stdout."""
    result = subprocess.run(
        [str(PYTHON), str(PROBE), scenario],
        cwd=tmp_path,
        env=_runtime_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe failed for {scenario}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    payload_line = next(
        line for line in reversed(result.stdout.splitlines()) if line.startswith("JSON_RESULT=")
    )
    return json.loads(payload_line.removeprefix("JSON_RESULT=")), result.stdout


class TestRealRuntimeDurability:
    """Small high-fidelity checks against actual Kitaru runtime behavior."""

    def test_replay_from_completed_checkpoint_boundary(self, tmp_path):
        payload, stdout = _run_probe(tmp_path, "replay")

        assert payload["scenario"] == "replay"
        assert payload["initial_result"] == 7
        assert payload["replay_result"] == 103
        assert payload["calls"] == {"first": 1, "second": 1, "third": 2}
        assert "Skipping checkpoint `replay_first`." in stdout
        assert "Checkpoint `replay_second` cached." in stdout

    def test_wait_input_resume_with_real_runtime(self, tmp_path):
        payload, stdout = _run_probe(tmp_path, "wait")

        assert payload["scenario"] == "wait"
        assert payload["wait_names"] == ["approve_draft"]
        assert payload["resume_status"] == "completed"
        assert payload["result"] == "draft:True"
        assert payload["calls"] == {"produce": 1, "finalize": 1}
        assert "Waiting on `approve_draft`" in stdout
        assert "Pausing execution" in stdout
        assert "Resuming execution" in stdout
