from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys
import tempfile


MODULES = [
    "deep_research",
    "deep_research.flow",
    "deep_research.checkpoints",
    "deep_research.agents",
    "deep_research.providers",
    "deep_research.evidence",
    "deep_research.renderers",
    "deep_research.critique",
    "deep_research.package",
    "deep_research.prompts",
    "deep_research.tools",
]


def test_package_modules_import() -> None:
    for module in MODULES:
        import_module(module)


def test_package_builds_wheel() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wheel_dir = Path(tmpdir) / "wheel"
        install_dir = Path(tmpdir) / "site"
        run_dir = Path(tmpdir) / "run"

        wheel_dir.mkdir()
        install_dir.mkdir()
        run_dir.mkdir()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--wheel-dir",
                str(wheel_dir),
            ],
            check=True,
        )

        wheels = list(wheel_dir.glob("*.whl"))

        assert wheels

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

        subprocess.run(
            [
                sys.executable,
                "-c",
                "from importlib import import_module; "
                f"modules = {MODULES!r}; "
                "[import_module(module) for module in modules]",
            ],
            check=True,
            cwd=run_dir,
            env={**os.environ, "PYTHONPATH": str(install_dir)},
        )
