"""Offline eval suite scaffolding."""

from . import brief_to_plan
from . import render_quality
from . import supervisor_trace_and_safety

try:
    from . import trajectory_quality as _trajectory_quality
    _TRAJECTORY_AVAILABLE = True
except ImportError:
    _trajectory_quality = None  # type: ignore[assignment]
    _TRAJECTORY_AVAILABLE = False


SUITE_REGISTRY = {
    brief_to_plan.SUITE_NAME: brief_to_plan.run,
    supervisor_trace_and_safety.SUITE_NAME: supervisor_trace_and_safety.run,
    render_quality.SUITE_NAME: render_quality.run,
}

if _TRAJECTORY_AVAILABLE and _trajectory_quality is not None:
    SUITE_REGISTRY[_trajectory_quality.SUITE_NAME] = _trajectory_quality.run
