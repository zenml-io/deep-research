import os
import subprocess
import sys
from pathlib import Path
import zipfile

from deep_research.prompts.loader import load_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_prompt_returns_markdown_contents() -> None:
    prompt = load_prompt("planner")
    assert "research" in prompt.lower()


def test_load_writer_prompts_returns_expected_contents() -> None:
    writer_prompt = load_prompt("writer")
    reading_path_prompt = load_prompt("writer_reading_path")
    backing_report_prompt = load_prompt("writer_backing_report")
    full_report_prompt = load_prompt("writer_full_report")

    assert "citation" in writer_prompt.lower()
    assert "ordered reading guide" in reading_path_prompt.lower()
    assert "analytical backing report" in backing_report_prompt.lower()
    assert "sectioned final report" in full_report_prompt.lower()


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
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(dist_dir),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    wheels = sorted(dist_dir.glob("*.whl"))
    assert wheels, "expected wheel to be built"

    install_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheels[0]) as wheel_zip:
        wheel_zip.extractall(install_dir)

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
                'assert "research" in prompt.lower(); '
                'assert "ordered reading guide" in load_prompt("writer_reading_path").lower(); '
                'assert "analytical backing report" in load_prompt("writer_backing_report").lower(); '
                'assert "sectioned final report" in load_prompt("writer_full_report").lower()'
            ),
        ],
        check=False,
        cwd=repo_root,
        env=child_env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
