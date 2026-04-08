from kitaru import checkpoint

from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    RenderPayload,
    ResearchPlan,
    SelectionGraph,
)
from deep_research.renderers.backing_report import (
    render_backing_report as _render_backing_report,
)
from deep_research.renderers.full_report import (
    render_full_report as _render_full_report,
)
from deep_research.renderers.reading_path import (
    render_reading_path as _render_reading_path,
)


@checkpoint(type="llm_call")
def render_reading_path(selection: SelectionGraph) -> RenderPayload:
    """Checkpoint: materialize the reading-path render from the current selection graph."""
    return _render_reading_path(selection)


@checkpoint(type="llm_call")
def render_backing_report(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
) -> RenderPayload:
    """Checkpoint: materialize the backing report render from the plan and selected evidence."""
    return _render_backing_report(selection, ledger, plan)


@checkpoint(type="llm_call")
def render_full_report(package: InvestigationPackage) -> RenderPayload:
    """Checkpoint: materialize the canonical full-report render from package state."""
    return _render_full_report(package)
