"""Synthesize prose from deterministic render scaffolds inside rendering checkpoints."""

from kitaru import checkpoint

from deep_research.config import ModelPricing, ResearchConfig
from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    IterationTrace,
    RenderCheckpointResult,
    RenderPayload,
    ResearchPlan,
    ResearchPreferences,
    SelectionGraph,
    StopReason,
)
from deep_research.renderers.materialization import materialize_render_payload
from deep_research.renderers import backing_report, full_report, reading_path


@checkpoint(type="llm_call")
def write_reading_path(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> RenderCheckpointResult:
    """Checkpoint: synthesize a reading path from the deterministic scaffold."""
    scaffold = reading_path.render_reading_path(selection, ledger, plan)
    return materialize_render_payload(
        scaffold,
        writer_model=config.writer_model,
        prompt_name="writer_reading_path",
        pricing=ModelPricing.model_validate(config.writer_pricing),
        preferences=preferences,
        max_context_chars=config.writer_context_budget_chars,
        snippet_budget_chars=config.context_snippet_budget_chars,
    )


@checkpoint(type="llm_call")
def write_backing_report(
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    iteration_trace: IterationTrace,
    provider_usage_summary: dict[str, int],
    stop_reason: StopReason,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> RenderCheckpointResult:
    """Checkpoint: synthesize a backing report from the deterministic scaffold."""
    scaffold = backing_report.render_backing_report(
        selection,
        ledger,
        plan,
        iteration_trace,
        provider_usage_summary,
        stop_reason,
    )
    return materialize_render_payload(
        scaffold,
        writer_model=config.writer_model,
        prompt_name="writer_backing_report",
        pricing=ModelPricing.model_validate(config.writer_pricing),
        preferences=preferences,
        max_context_chars=config.writer_context_budget_chars,
        snippet_budget_chars=config.context_snippet_budget_chars,
    )


@checkpoint(type="llm_call")
def write_full_report(
    package: InvestigationPackage,
    config: ResearchConfig,
    preferences: ResearchPreferences | None = None,
) -> RenderCheckpointResult:
    """Checkpoint: synthesize the canonical full report from package state."""
    return materialize_render_payload(
        full_report.render_full_report(package),
        writer_model=config.writer_model,
        prompt_name="writer_full_report",
        pricing=ModelPricing.model_validate(config.writer_pricing),
        preferences=preferences,
        max_context_chars=config.writer_context_budget_chars,
        snippet_budget_chars=config.context_snippet_budget_chars,
    )
