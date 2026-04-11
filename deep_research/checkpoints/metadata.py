"""Checkpoints that encapsulate wall-clock and run identity non-determinism.

Keeping ``uuid4`` and ``datetime.now`` calls behind checkpoint boundaries means
the flow body is pure with respect to the checkpoint cache: on replay, the
Kitaru runtime returns the original stamped values rather than freshly minted
ones, which is exactly what we want for deterministic re-execution.
"""

from datetime import datetime, timezone
from uuid import uuid4

from kitaru import checkpoint
from pydantic import BaseModel, Field

from deep_research.models import RunMetadataStamp


def _utc_now_iso() -> str:
    """Return the current UTC time as a Z-suffixed ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class WallClockSnapshot(BaseModel):
    """Immutable wall-clock observation taken relative to a run start stamp.

    ``elapsed_seconds`` is an integer to match the ``RunSummary.elapsed_seconds``
    contract the downstream assembler expects.
    """

    now_iso: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of the moment the snapshot was taken.",
    )
    elapsed_seconds: int = Field(
        ...,
        ge=0,
        description="Whole seconds elapsed since the run's started_at stamp.",
    )


@checkpoint
def stamp_run_metadata() -> RunMetadataStamp:
    """Checkpoint: mint a stable run id and start timestamp for the flow.

    The checkpoint takes no flow-state arguments so Kitaru caches it once per
    run; replaying the flow returns the same ``run_id`` and ``started_at``.
    """
    return RunMetadataStamp(
        run_id=f"run-{uuid4()}",
        started_at=_utc_now_iso(),
    )


@checkpoint
def snapshot_wall_clock(started_at: str) -> WallClockSnapshot:
    """Checkpoint: capture elapsed seconds and an ISO ``now`` relative to ``started_at``.

    Replay returns the cached snapshot, so convergence decisions remain stable
    across re-runs even though wall-clock time would otherwise advance.
    """
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    elapsed = max(0, int((now - started).total_seconds()))
    return WallClockSnapshot(
        now_iso=now.isoformat().replace("+00:00", "Z"),
        elapsed_seconds=elapsed,
    )


class RunFinalization(BaseModel):
    """Completion metadata captured at the end of a run."""

    completed_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp when the run finished assembling.",
    )
    elapsed_seconds: int = Field(
        ...,
        ge=0,
        description="Whole seconds between started_at and completed_at.",
    )


@checkpoint
def finalize_run_metadata(started_at: str) -> RunFinalization:
    """Checkpoint: stamp the run completion time and total elapsed seconds.

    Mirrors ``stamp_run_metadata`` at the tail of the flow so that on replay we
    observe the original completion stamp rather than a fresh wall-clock read.
    """
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    elapsed = max(0, int((now - started).total_seconds()))
    return RunFinalization(
        completed_at=now.isoformat().replace("+00:00", "Z"),
        elapsed_seconds=elapsed,
    )
