from importlib import import_module
from pathlib import Path
import subprocess
import sys
import tempfile


def test_package_modules_import() -> None:
    modules = [
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
    for module in modules:
        import_module(module)


def test_package_builds_wheel() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--wheel-dir",
                tmpdir,
            ],
            check=True,
        )

        assert list(Path(tmpdir).glob("*.whl"))
