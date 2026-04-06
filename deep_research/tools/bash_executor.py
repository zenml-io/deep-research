import shlex
import subprocess
import tempfile
from pathlib import Path

from deep_research.models import RawToolResult


ALLOWED_COMMANDS = {"echo", "ls", "pwd", "true", "false"}


def run_bash(command: str, timeout_sec: int = 20) -> RawToolResult:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return RawToolResult(
            tool_name="run_bash",
            provider="bash",
            payload={},
            ok=False,
            error=str(exc),
        )

    if not argv or not _is_allowed_command(argv[0]):
        return RawToolResult(
            tool_name="run_bash",
            provider="bash",
            payload={},
            ok=False,
            error="command not allowed",
        )

    if any(_is_disallowed_path_argument(arg) for arg in argv[1:]):
        return RawToolResult(
            tool_name="run_bash",
            provider="bash",
            payload={},
            ok=False,
            error="command not allowed",
        )

    with tempfile.TemporaryDirectory(prefix="deep-research-") as temp_dir:
        try:
            completed = subprocess.run(
                argv,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env={},
            )
        except FileNotFoundError as exc:
            return RawToolResult(
                tool_name="run_bash",
                provider="bash",
                payload={},
                ok=False,
                error=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            return RawToolResult(
                tool_name="run_bash",
                provider="bash",
                payload={"stdout": exc.stdout or "", "stderr": exc.stderr or ""},
                ok=False,
                error=f"command timed out after {timeout_sec} seconds",
            )

    return RawToolResult(
        tool_name="run_bash",
        provider="bash",
        payload={
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        },
        ok=completed.returncode == 0,
        error=None if completed.returncode == 0 else completed.stderr,
    )


def _is_allowed_command(command_name: str) -> bool:
    if Path(command_name).is_absolute():
        return False

    return command_name in ALLOWED_COMMANDS


def _is_disallowed_path_argument(argument: str) -> bool:
    if not argument or argument.startswith("-"):
        return False

    path = Path(argument)
    if path.is_absolute():
        return True

    return ".." in path.parts
