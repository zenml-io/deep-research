"""Sandbox code-execution interface (stub).

Provides a ``CodeExecResult`` dataclass and a ``SandboxExecutor`` that
delegates to a pluggable backend.  The actual backend is **not** implemented
yet — this module defines the interface only.

Design constraints:
- Gated by ``config.sandbox_enabled``.
- When disabled, ``code_exec`` is omitted from the tool surface (not emulated).
- Sandboxes are ephemeral and time-bounded.
- Network access is off by default.
- Data is passed in explicitly (no ambient filesystem access).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Defaults ------------------------------------------------------------------
DEFAULT_EXEC_TIMEOUT_SEC = 30
DEFAULT_MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class CodeExecResult:
    """Result of a sandboxed code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if the execution completed with exit code 0."""
        return self.exit_code == 0 and not self.timed_out


class SandboxNotAvailableError(RuntimeError):
    """Raised when sandbox execution is requested but not configured."""


class SandboxExecutor:
    """Interface for sandboxed code execution.

    Parameters
    ----------
    backend:
        Name of the sandbox backend (e.g. ``"docker"``, ``"e2b"``).
        Currently a placeholder — no backends are implemented yet.
    timeout_sec:
        Maximum wall-clock seconds for a single execution.
    max_output_chars:
        Truncation limit for stdout/stderr.
    allow_network:
        Whether the sandbox has network access. Off by default.
    """

    def __init__(
        self,
        backend: str,
        *,
        timeout_sec: int = DEFAULT_EXEC_TIMEOUT_SEC,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        allow_network: bool = False,
    ) -> None:
        self.backend = backend
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.allow_network = allow_network

    async def execute(
        self,
        code: str,
        *,
        language: str = "python",
        input_data: dict[str, str] | None = None,
    ) -> CodeExecResult:
        """Execute *code* in the sandbox and return the result.

        Parameters
        ----------
        code:
            Source code to execute.
        language:
            Programming language (default ``"python"``).
        input_data:
            Mapping of filename → content to make available in the sandbox.
            Data is passed in explicitly rather than via ambient filesystem.

        Raises
        ------
        SandboxNotAvailableError
            If the configured backend is not implemented.
        """
        # Stub: no backends are implemented yet.
        raise SandboxNotAvailableError(
            f"Sandbox backend {self.backend!r} is not implemented. "
            "This is a stub interface — plug in a real backend "
            "(e.g. Docker, E2B) to enable code execution."
        )
