import shlex
import subprocess
import tempfile

from deep_research.models import RawToolResult


DENYLIST = {"rm", "mv", "sudo", "chmod", "chown", "dd"}


def run_bash(command: str, timeout_sec: int = 20) -> RawToolResult:
    argv = shlex.split(command)
    if not argv or argv[0] in DENYLIST:
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
