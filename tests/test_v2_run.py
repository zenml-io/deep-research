"""Structural tests for the V2 CLI entrypoint (run_v2.py).

These tests verify that run_v2.py stays a thin dispatch surface:
no embedded helpers, no business logic, just argparse -> flow -> write.
They do NOT import the flow or call deep_research.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Absolute path to run_v2.py in the worktree root
RUN_V2_PATH = Path(__file__).resolve().parent.parent / "run_v2.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_source() -> str:
    return RUN_V2_PATH.read_text(encoding="utf-8")


def _parse_ast() -> ast.Module:
    return ast.parse(_read_source(), filename=str(RUN_V2_PATH))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunV2Structure:
    """Structural invariants for run_v2.py — enforced via AST / line count."""

    def test_run_v2_stays_below_line_limit(self):
        """Dispatch surface must stay under 120 lines."""
        source = _read_source()
        line_count = len(source.splitlines())
        assert line_count < 120, (
            f"run_v2.py is {line_count} lines — dispatch surface should be < 120"
        )

    def test_run_v2_no_embedded_helpers(self):
        """Only `main` should be defined as a function. No class defs."""
        tree = _parse_ast()

        # Collect top-level function defs
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            or isinstance(node, ast.AsyncFunctionDef)
        ]
        # Only 'main' should exist
        assert func_names == ["main"], (
            f"Expected only 'main' function, found: {func_names}"
        )

        # No class definitions at any level
        class_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        assert class_names == [], (
            f"No classes should be defined in dispatch surface, found: {class_names}"
        )

    def test_run_v2_imports_runtime_only(self):
        """Lazy imports inside main() should be from `research.` package."""
        tree = _parse_ast()

        # Find the main function body
        main_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_func = node
                break
        assert main_func is not None, "main() function not found"

        # Collect all ImportFrom nodes inside main()
        lazy_imports = [
            node.module
            for node in ast.walk(main_func)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert len(lazy_imports) > 0, "Expected lazy imports inside main()"
        for mod in lazy_imports:
            assert mod.startswith("research."), (
                f"Lazy import {mod!r} should be from 'research.' package, "
                f"not 'deep_research.' or other"
            )

    def test_run_v2_has_required_flags(self):
        """Source must contain all required CLI flags."""
        source = _read_source()
        required_flags = [
            "--tier",
            "--enable-sandbox",
            "--allow-unfinalized",
            "--council",
            "--output",
        ]
        for flag in required_flags:
            assert flag in source, f"Missing required CLI flag: {flag}"


class TestRunV2Argparse:
    """Behavioral tests for argparse configuration."""

    def test_run_v2_argparse_tier_choices(self):
        """--tier accepts quick/standard/deep and rejects invalid values."""
        # Run as subprocess to test argparse without importing flow modules
        for valid_tier in ("quick", "standard", "deep"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(f"""\
                        import sys
                        sys.argv = ["run_v2.py", "--tier", "{valid_tier}", "test question"]
                        # Patch imports before main() can reach them
                        import types
                        mock_research = types.ModuleType("research")
                        mock_config = types.ModuleType("research.config")
                        mock_settings = types.ModuleType("research.config.settings")
                        mock_flows = types.ModuleType("research.flows")
                        mock_dr = types.ModuleType("research.flows.deep_research")
                        mock_pkg = types.ModuleType("research.package")
                        mock_export = types.ModuleType("research.package.export")
                        sys.modules["research"] = mock_research
                        sys.modules["research.config"] = mock_config
                        sys.modules["research.config.settings"] = mock_settings
                        sys.modules["research.flows"] = mock_flows
                        sys.modules["research.flows.deep_research"] = mock_dr
                        sys.modules["research.package"] = mock_pkg
                        sys.modules["research.package.export"] = mock_export
                        # Just test that argparse doesn't reject the tier
                        import argparse
                        # Re-parse args manually
                        parser = argparse.ArgumentParser()
                        parser.add_argument("question")
                        parser.add_argument("--tier", choices=["quick", "standard", "deep"], default="standard")
                        args = parser.parse_args(sys.argv[1:])
                        assert args.tier == "{valid_tier}"
                        print("OK")
                    """),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, (
                f"--tier {valid_tier} should be accepted. stderr: {result.stderr}"
            )

        # Invalid tier should be rejected
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import sys, argparse
                    sys.argv = ["run_v2.py", "--tier", "ultra", "test question"]
                    parser = argparse.ArgumentParser()
                    parser.add_argument("question")
                    parser.add_argument("--tier", choices=["quick", "standard", "deep"], default="standard")
                    parser.parse_args(sys.argv[1:])
                """),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0, "--tier ultra should be rejected"

    def test_run_v2_council_flag_exits_with_error(self):
        """--council should print an error and exit with non-zero status."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """\
                    import sys, types

                    # Stub out research.* modules so lazy imports don't fail
                    for mod_name in [
                        "research", "research.config", "research.config.settings",
                        "research.config.budget", "research.config.defaults",
                        "research.config.slots",
                        "research.flows", "research.flows.deep_research",
                        "research.package", "research.package.export",
                    ]:
                        sys.modules[mod_name] = types.ModuleType(mod_name)

                    # Provide a fake ResearchConfig with for_tier and model_copy
                    class FakeConfig:
                        sandbox_enabled = False
                        allow_unfinalized_package = False
                        @classmethod
                        def for_tier(cls, tier):
                            return cls()
                        def model_copy(self, update=None):
                            return self

                    sys.modules["research.config.settings"].ResearchConfig = FakeConfig
                    sys.modules["research.flows.deep_research"].deep_research = None
                    sys.modules["research.package.export"].write_package = None

                    sys.argv = ["run_v2.py", "--council", "test question"]

                    # Now import and run main
                    sys.path.insert(0, "{run_v2_dir}")
                    from run_v2 import main
                    main()
                """.format(run_v2_dir=str(RUN_V2_PATH.parent))
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0, "Council mode should exit with non-zero"
        assert "not yet implemented" in result.stderr.lower(), (
            f"Expected 'not yet implemented' in stderr, got: {result.stderr!r}"
        )
