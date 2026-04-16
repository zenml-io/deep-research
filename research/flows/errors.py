"""Flow-level exception types for the deep research engine.

Centralised here so both ``deep_research`` and ``council`` flows can import
without creating a cross-module dependency between the two flow modules.
"""

from __future__ import annotations

from research.contracts.evidence import EvidenceLedger


class FlowTimeoutError(Exception):
    """Raised when a flow-level wait() exceeds config.wait_timeout_seconds."""


class PlanApprovalRejectedError(Exception):
    """Raised when the operator rejects the generated research plan."""


class SupervisorError(Exception):
    """Raised when the supervisor fails twice consecutively.

    Carries the last valid ledger state so operators can inspect progress
    and replay from the last good checkpoint.
    """

    def __init__(self, message: str, ledger: EvidenceLedger | None = None):
        super().__init__(message)
        self.ledger = ledger


class FinalizerError(Exception):
    """Raised when the finalizer fails and allow_unfinalized_package is False.

    Draft and critique are preserved in the flow's checkpoint history for
    replay after the underlying issue is fixed.
    """


class CouncilConfigError(Exception):
    """Raised when council configuration is invalid."""


class CouncilSelectionError(Exception):
    """Raised when the operator selects an invalid council winner."""
