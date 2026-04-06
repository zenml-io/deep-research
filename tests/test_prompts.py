import os
import subprocess
import sys
from pathlib import Path

from deep_research.prompts.loader import load_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_prompt_returns_markdown_contents() -> None:
    prompt = load_prompt("planner")
    assert "research" in prompt.lower()


def test_load_prompt_rejects_unknown_name() -> None:
    try:
        load_prompt("missing")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_load_prompt_from_installed_wheel(tmp_path) -> None:
    dist_dir = tmp_path / "dist"
    install_dir = tmp_path / "installed"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(dist_dir),
            str(REPO_ROOT),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    wheels = sorted(dist_dir.glob("*.whl"))
    assert wheels, "expected wheel to be built"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_dir),
            str(wheels[0]),
        ],
        check=True,
    )

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(install_dir)

    repo_root = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from deep_research.prompts.loader import load_prompt; "
                'prompt = load_prompt("planner"); '
                'assert "research" in prompt.lower()'
            ),
        ],
        check=False,
        cwd=repo_root,
        env=child_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
