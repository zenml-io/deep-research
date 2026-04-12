"""Offline eval suite scaffolding."""

from . import brief_to_plan
from . import render_quality
from . import trajectory_quality
from . import supervisor_trace_and_safety


SUITE_REGISTRY = {
    brief_to_plan.SUITE_NAME: brief_to_plan.run,
    supervisor_trace_and_safety.SUITE_NAME: supervisor_trace_and_safety.run,
    render_quality.SUITE_NAME: render_quality.run,
    trajectory_quality.SUITE_NAME: trajectory_quality.run,
}
