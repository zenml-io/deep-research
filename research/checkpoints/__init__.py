"""Kitaru checkpoint functions — durable replay boundaries.

Uses lazy imports via ``__getattr__`` to avoid pulling in kitaru at
package-import time, matching the pattern in ``research.agents``.
"""

__all__ = [
    "CitationResolutionError",
    "GroundingError",
    "RunFinalization",
    "RunStamp",
    "WallClockSnapshot",
    "assemble_package",
    "finalize_run_metadata",
    "record_iteration_spend",
    "resolve_tool_surface",
    "snapshot_wall_clock",
    "stamp_run_metadata",
    "run_critique",
    "run_draft",
    "run_finalize",
    "run_verify",
    "run_plan",
    "run_plan_revision",
    "run_scope",
    "run_subagent",
    "run_supervisor",
    "run_judge",
]

_METADATA_NAMES = {
    "RunFinalization",
    "RunStamp",
    "WallClockSnapshot",
    "finalize_run_metadata",
    "record_iteration_spend",
    "snapshot_wall_clock",
    "stamp_run_metadata",
}

_SCOPE_NAMES = {"run_scope"}
_PLAN_NAMES = {"run_plan"}
_REPLAN_NAMES = {"run_plan_revision"}
_SUPERVISOR_NAMES = {"run_supervisor"}
_SUBAGENT_NAMES = {"run_subagent"}
_DRAFT_NAMES = {"run_draft"}
_CRITIQUE_NAMES = {"run_critique"}
_FINALIZE_NAMES = {"run_finalize"}
_VERIFY_NAMES = {"run_verify"}
_JUDGE_NAMES = {"run_judge"}
_ASSEMBLE_NAMES = {"assemble_package", "GroundingError", "CitationResolutionError"}
_TOOL_SURFACE_NAMES = {"resolve_tool_surface"}


def __getattr__(name: str):
    """Lazy import to avoid pulling in kitaru at package-import time.

    This lets ``import research.checkpoints`` succeed even when kitaru is
    not installed (e.g. in the lightweight test environment on Python 3.14).
    The actual import happens on first attribute access.
    """
    if name in _METADATA_NAMES:
        from research.checkpoints import metadata

        return getattr(metadata, name)
    if name in _SCOPE_NAMES:
        from research.checkpoints import scope

        return getattr(scope, name)
    if name in _PLAN_NAMES:
        from research.checkpoints import plan

        return getattr(plan, name)
    if name in _REPLAN_NAMES:
        from research.checkpoints import replan

        return getattr(replan, name)
    if name in _SUPERVISOR_NAMES:
        from research.checkpoints import supervisor

        return getattr(supervisor, name)
    if name in _SUBAGENT_NAMES:
        from research.checkpoints import subagent

        return getattr(subagent, name)
    if name in _DRAFT_NAMES:
        from research.checkpoints import draft

        return getattr(draft, name)
    if name in _CRITIQUE_NAMES:
        from research.checkpoints import critique

        return getattr(critique, name)
    if name in _FINALIZE_NAMES:
        from research.checkpoints import finalize

        return getattr(finalize, name)
    if name in _VERIFY_NAMES:
        from research.checkpoints import verify

        return getattr(verify, name)
    if name in _JUDGE_NAMES:
        from research.checkpoints import judge

        return getattr(judge, name)
    if name in _ASSEMBLE_NAMES:
        from research.checkpoints import assemble

        return getattr(assemble, name)
    if name in _TOOL_SURFACE_NAMES:
        from research.checkpoints import tool_surface

        return getattr(tool_surface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
