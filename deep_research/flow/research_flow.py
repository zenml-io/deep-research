"""Top-level research flow.

This module is intentionally thin: it imports every checkpoint the pipeline
uses (so tests can monkeypatch them on the module), wires them into the
``research_flow`` kitaru flow, and delegates the heavy lifting to private
helpers in ``deep_research.flow._pipeline``. Non-determinism (``uuid4`` /
``datetime.now``) is confined to ``deep_research.checkpoints.metadata`` so
replay returns the original stamped values.
"""

from kitaru import flow, log, wait

from deep_research.checkpoints.assemble import assemble_package
from deep_research.checkpoints.extract_claims import extract_claims
from deep_research.checkpoints.classify import classify_request
from deep_research.checkpoints.coherence import verify_coherence
from deep_research.checkpoints.council import (
    aggregate_council_results,
    run_council_generator,
)
from deep_research.checkpoints.evaluate import score_coverage
from deep_research.checkpoints.fetch import enrich_candidates
from deep_research.checkpoints.grounding import verify_grounding
from deep_research.checkpoints.merge import update_ledger
from deep_research.checkpoints.metadata import (
    finalize_run_metadata,
    snapshot_wall_clock,
    stamp_run_metadata,
)
from deep_research.checkpoints.normalize import extract_candidates
from deep_research.checkpoints.plan import build_plan
from deep_research.checkpoints.replan import evaluate_replan
from deep_research.checkpoints.relevance import score_relevance
from deep_research.checkpoints.rendering import (
    write_backing_report,
    write_full_report,
    write_reading_path,
)
from deep_research.checkpoints.review import critique_reports
from deep_research.checkpoints.revise import apply_revisions
from deep_research.checkpoints.search import execute_searches
from deep_research.checkpoints.select import rank_evidence
from deep_research.checkpoints.supervisor import run_supervisor
from deep_research.config import ResearchConfig
from deep_research.enums import StopReason, Tier
from deep_research.flow._types import (
    APPROVE_PLAN_WAIT_NAME,
    CLARIFY_BRIEF_WAIT_NAME,
    build_tool_call_records,
    merge_provider_counts,
)
from deep_research.flow.assembly import assemble_final_package
from deep_research.flow.classify import build_plan_with_grounding, resolve_config_and_classify
from deep_research.flow.claims import run_claim_extraction_if_enabled
from deep_research.flow.convergence import check_convergence
from deep_research.flow.critique import run_critique_if_enabled, run_judges_if_enabled
from deep_research.flow.loop import run_iteration_loop
from deep_research.flow.rendering import render_deliverable
from deep_research.models import (
    EvidenceLedger,
    InvestigationPackage,
    RawToolResult,
    ToolCallRecord,
)
from deep_research.observability import bootstrap_logfire, span


CLASSIFY_CHECKPOINT_NAME = "classify_request"
PLAN_CHECKPOINT_NAME = "build_plan"
SUPERVISOR_CHECKPOINT_NAME = "run_supervisor"
COUNCIL_GENERATOR_CHECKPOINT_NAME = "run_council_generator"


# Re-exported for backwards compatibility: a few tests import these helpers
# from this module directly. Keep them here so the public surface of the flow
# module stays stable.
_merge_provider_counts = merge_provider_counts
_build_tool_call_records = build_tool_call_records


@flow
def research_flow(
    brief: str,
    tier: str = "auto",
    config: ResearchConfig | None = None,
) -> InvestigationPackage:
    """Orchestrate the full research pipeline from brief to investigation package."""
    bootstrap_logfire()
    with span("research_flow", brief_length=len(brief), tier=tier):
        stamp = stamp_run_metadata.submit().load()
        run_state = resolve_config_and_classify(brief, tier, config)
        plan = build_plan_with_grounding(run_state)
        iteration_output = run_iteration_loop(plan, run_state, stamp)
        selection = rank_evidence.submit(
            iteration_output.ledger, plan, run_state.config
        ).load()
        renders, render_cost, degradations = render_deliverable(
            selection, iteration_output, plan, run_state, stamp
        )
        spent_usd = iteration_output.spent_usd + render_cost
        critique_bundle = run_critique_if_enabled(
            renders, plan, selection, iteration_output.ledger, run_state.config
        )
        spent_usd = spent_usd + critique_bundle.spent_usd
        judge_bundle = run_judges_if_enabled(
            critique_bundle.renders, plan, iteration_output.ledger, run_state.config
        )
        spent_usd = spent_usd + judge_bundle.spent_usd
        claim_inventory, claim_cost = run_claim_extraction_if_enabled(
            critique_bundle.renders, iteration_output.ledger, plan, run_state.config
        )
        spent_usd = spent_usd + claim_cost
        finalization = finalize_run_metadata.submit(
            stamp.started_at,
        ).load()
        return assemble_final_package(
            stamp=stamp,
            finalization=finalization,
            run_state=run_state,
            plan=plan,
            iteration_output=iteration_output,
            selection=selection,
            renders=critique_bundle.renders,
            spent_usd=spent_usd,
            critique_result=critique_bundle.critique_result,
            grounding_result=judge_bundle.grounding_result,
            coherence_result=judge_bundle.coherence_result,
            degradations=degradations,
            claim_inventory=claim_inventory,
        )
