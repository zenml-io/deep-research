"""Checkpoints that encapsulate timing and run identity non-determinism.

Keeping uuid4 and datetime.now calls behind checkpoint boundaries means
the flow body is pure with respect to the checkpoint cache: on replay, the
Kitaru runtime returns the original stamped values rather than freshly minted
ones.
"""

from datetime import datetime, timezone
from uuid import uuid4

from kitaru import checkpoint
from pydantic import BaseModel, Field


class RunStamp(BaseModel):
    """Stable run identity minted once per execution."""

    run_id: str = Field(..., description="Unique run identifier (run-<uuid4>).")
    started_at: str = Field(..., description="ISO-8601 UTC start timestamp.")


class WallClockSnapshot(BaseModel):
    """Timing observation relative to a run start stamp."""

    now_iso: str = Field(..., description="ISO-8601 UTC timestamp of the snapshot.")
    elapsed_seconds: int = Field(..., ge=0, description="Seconds since started_at.")


class RunFinalization(BaseModel):
    """Completion metadata captured at the end of a run."""

    completed_at: str = Field(
        ..., description="ISO-8601 UTC timestamp when the run finished."
    )
    elapsed_seconds: int = Field(
        ..., ge=0, description="Seconds between started_at and completed_at."
    )


def _utc_now_iso() -> str:
    """Return current UTC time as Z-suffixed ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _elapsed_since(started_at: str) -> tuple[str, int]:
    """Return (now_iso, elapsed_seconds) since started_at."""
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")
    elapsed = max(0, int((now - started).total_seconds()))
    return now_iso, elapsed


@checkpoint
def stamp_run_metadata() -> RunStamp:
    """Mint a stable run id and start timestamp.

    On replay, returns the original cached values.
    """
    return RunStamp(
        run_id=f"run-{uuid4()}",
        started_at=_utc_now_iso(),
    )


@checkpoint
def snapshot_wall_clock(started_at: str) -> WallClockSnapshot:
    """Capture elapsed seconds relative to started_at.

    On replay, returns the cached snapshot.
    """
    now_iso, elapsed = _elapsed_since(started_at)
    return WallClockSnapshot(now_iso=now_iso, elapsed_seconds=elapsed)


@checkpoint
def finalize_run_metadata(started_at: str) -> RunFinalization:
    """Stamp run completion time.

    On replay, returns the original completion timestamp.
    """
    completed_at, elapsed = _elapsed_since(started_at)
    return RunFinalization(completed_at=completed_at, elapsed_seconds=elapsed)


@checkpoint(type="tool_call")
def record_iteration_spend(iteration_index: int, cost_usd: float) -> float:
    """Stamp per-iteration spend so the flow's budget view is replay-stable.

    ``BudgetTracker`` mutates ``BudgetConfig.spent_usd`` during live runs,
    but cached LLM checkpoints don't re-invoke the tracker on replay —
    spent_usd would reset to 0 and the flow's convergence decisions would
    diverge. Submitting this checkpoint with a stable per-iteration ``id``
    caches the original cost so replay sees the same cumulative spend.
    """
    return cost_usd
