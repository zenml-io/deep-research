import shlex
import subprocess
import tempfile
from pathlib import Path

from deep_research.models import RawToolResult


ALLOWED_COMMANDS = {"echo", "ls", "pwd", "true", "false"}


def error_result(error: str, payload: dict | None = None) -> RawToolResult:
    """Construct a failed bash-tool result with a normalized payload and error message.

    Centralizing this shape keeps all early-return validation and timeout failures aligned
    with the `RawToolResult` contract used by the rest of the research pipeline.
    """
    return RawToolResult(
        tool_name="run_bash",
        provider="bash",
        payload=payload or {},
        ok=False,
        error=error,
    )


def run_bash(command: str, timeout_sec: int = 20) -> RawToolResult:
    """Execute an allow-listed bash command in a temporary directory."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return error_result(str(exc))

    if not argv or not is_allowed_command(argv[0]):
        return error_result("command not allowed")

    if any(is_disallowed_path_argument(arg) for arg in argv[1:]):
        return error_result("command not allowed")

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
            return error_result(str(exc))
        except subprocess.TimeoutExpired as exc:
            return error_result(
                f"command timed out after {timeout_sec} seconds",
                payload={"stdout": exc.stdout or "", "stderr": exc.stderr or ""},
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


def is_allowed_command(command_name: str) -> bool:
    """Return whether a command name is allowed by the executor allow-list.

    Absolute paths are always rejected so callers cannot bypass the curated command set
    by pointing directly at binaries outside the approved allow-listed names.
    """
    if Path(command_name).is_absolute():
        return False

    return command_name in ALLOWED_COMMANDS


def is_disallowed_path_argument(argument: str) -> bool:
    """Reject path-like arguments that escape the temporary execution directory.

    Option flags are ignored, but absolute paths and parent-directory traversal segments
    are blocked so allow-listed commands cannot be repurposed to read arbitrary files.
    """
    if not argument or argument.startswith("-"):
        return False

    path = Path(argument)
    if path.is_absolute():
        return True

    return ".." in path.parts
