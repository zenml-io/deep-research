"""Private helpers for ``research_flow``.

These functions are split out of ``research_flow.py`` so that module can stay a
short, readable orchestration script. They are intentionally private (``_``
prefix at module level) because their signatures are tailored to the flow and
are not a public API surface.

Checkpoint lookups go through ``deep_research.flow.research_flow`` so existing
tests that monkeypatch checkpoints on that module continue to work unchanged.
The lazy ``_flow`` accessor avoids a circular import at module load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from types import ModuleType
from typing import NamedTuple

from pydantic import BaseModel, Field

from deep_research.config import ResearchConfig
from deep_research.enums import DeliverableMode, StopReason, Tier
from deep_research.flow.convergence import ConvergenceSignal, detect_source_diversity_warning
from deep_research.models import (
    CoherenceResult,
    CoverageScore,
    CritiqueResult,
    EvidenceLedger,
    GroundingResult,
    InvestigationPackage,
    IterationRecord,
    IterationTrace,
    RawToolResult,
    RenderPayload,
    RenderSettingsSnapshot,
    RequestClassification,
    ResearchPlan,
    ResearchPreferences,
    RunSummary,
    SearchAction,
    SearchExecutionResult,
    SupervisorDecision,
    SelectionGraph,
    SupervisorCheckpointResult,
    ToolCallRecord,
)
from deep_research.observability import metric, span


APPROVE_PLAN_WAIT_NAME = "approve_plan"
CLARIFY_BRIEF_WAIT_NAME = "clarify_brief"

# Deliverable modes the writer fully supports; other modes still render but
# surface a degradation note in the final package.
_FULLY_SUPPORTED_MODES: frozenset[DeliverableMode] = frozenset(
    {DeliverableMode.RESEARCH_PACKAGE, DeliverableMode.FINAL_REPORT}
)


def _flow() -> ModuleType:
    """Return the live ``research_flow`` module for attribute lookup.

    Using an attribute lookup (rather than a direct import) means pytest
    monkeypatches applied to ``deep_research.flow.research_flow.<name>`` are
    honoured by these helpers without re-importing anything.
    """
    from deep_research.flow import research_flow

    return research_flow


@dataclass(frozen=True)
class RunState:
    """Resolved inputs the iteration loop + downstream helpers consume."""

    brief: str
    config: ResearchConfig
    classification: RequestClassification
    preferences: ResearchPreferences
    clarify_options: "ClarifyOptions | None" = None
    seeded_entities: "SeededEntities | None" = None


class IterationLoopOutput(NamedTuple):
    ledger: EvidenceLedger
    iteration_history: tuple[IterationRecord, ...]
    provider_usage_summary: dict[str, int]
    spent_usd: float
    stop_reason: StopReason
    council_models: list[str]
    active_elapsed_seconds: int
    wall_elapsed_seconds: int
    total_tokens: int


class CritiqueBundle(NamedTuple):
    renders: list[RenderPayload]
    critique_result: CritiqueResult | None
    spent_usd: float


class JudgeBundle(NamedTuple):
    grounding_result: GroundingResult | None
    coherence_result: CoherenceResult | None
    spent_usd: float


class ClarifyOptions(BaseModel):
    clarified_brief: str | None = Field(default=None)
    scope_adjustment: str | None = Field(default=None)
    source_preference: str | None = Field(default=None)
    depth_preference: str | None = Field(default=None)
    comparison_targets: list[str] = Field(default_factory=list)
    deliverable_mode: str | None = Field(default=None)


class PlanApproval(BaseModel):
    approved: bool = True
    notes: str | None = None
    scope_adjustment: str | None = None
    source_preference: str | None = None
    deliverable_mode: str | None = None


class SeededEntities(BaseModel):
    projects: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)

    def flattened(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for group in (
            self.projects,
            self.benchmarks,
            self.products,
            self.companies,
            self.key_terms,
        ):
            for value in group:
                cleaned = value.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    ordered.append(cleaned)
        return ordered


class WaveSearchResult(NamedTuple):
    decision: SupervisorDecision
    raw_results: list[RawToolResult]
    supervisor_cost: float
    search_cost: float
    warnings: list[str]
    tool_calls: list[ToolCallRecord]
    step_latencies_ms: dict[str, int]
    total_tokens: int


class WaveTriageResult(NamedTuple):
    candidates: list
    relevance_cost: float
    step_latencies_ms: dict[str, int]
    total_tokens: int


class WaveEnrichResult(NamedTuple):
    ledger: EvidenceLedger
    step_latencies_ms: dict[str, int]


class WaveScoreResult(NamedTuple):
    coverage: CoverageScore
    step_latencies_ms: dict[str, int]


class WaveFeedbackResult(NamedTuple):
    carryover_actions: list[SearchAction]
    reason: str | None
    # Wave 4.5: synthesis-to-unsupported-claims feedback. Count of synthesis
    # feedback iterations consumed so far (deep tier, capped by policy).
    feedback_iteration_count: int = 0


def merge_provider_counts(
    existing: dict[str, int],
    raw_results: list[RawToolResult],
) -> dict[str, int]:
    """Return a new dict with provider counts updated from ``raw_results``."""
    updated = dict(existing)
    for raw_result in raw_results:
        updated[raw_result.provider] = updated.get(raw_result.provider, 0) + 1
    return updated


def build_tool_call_records(
    raw_results: list[RawToolResult],
) -> list[ToolCallRecord]:
    """Build immutable tool-call records from raw results."""
    records: list[ToolCallRecord] = []
    for raw_result in raw_results:
        provider_label = f" via {raw_result.provider}" if raw_result.provider else ""
        if raw_result.ok:
            summary = f"{raw_result.tool_name}{provider_label} succeeded"
            status = "ok"
        else:
            detail = f": {raw_result.error}" if raw_result.error else ""
            summary = f"{raw_result.tool_name}{provider_label} failed{detail}"
            status = "error"
        records.append(
            ToolCallRecord(
                tool_name=raw_result.tool_name,
                status=status,
                provider=raw_result.provider,
                summary=summary,
            )
        )
    return records


def resolve_config_for_tier(
    user_config: ResearchConfig | None,
    resolved_tier: Tier,
) -> ResearchConfig:
    """Return a config for ``resolved_tier``, layering user overrides on top.

    Uses ``exclude_unset=True`` so Pydantic defaults on the user's config do
    NOT clobber the tier-specific defaults. Always returns a fresh frozen
    ``ResearchConfig``.
    """
    tier_config = ResearchConfig.for_tier(resolved_tier)
    if user_config is None:
        return tier_config
    if user_config.tier == resolved_tier:
        return user_config
    overrides = user_config.model_dump(exclude_unset=True)
    overrides["tier"] = resolved_tier
    return tier_config.model_copy(update=overrides)


def _classify_once(
    brief: str,
    tier: str,
    user_config: ResearchConfig | None,
) -> tuple[RequestClassification, ResearchConfig]:
    """Run classification once and fold the result into a resolved config."""
    flow = _flow()
    classification = flow.classify_request.submit(brief, user_config).load()
    resolved_tier = classification.recommended_tier if tier == "auto" else Tier(tier)
    return classification, resolve_config_for_tier(user_config, resolved_tier)


def _config_value(config: ResearchConfig, nested_attr: str, fallback_attr: str, default):
    nested = getattr(config, nested_attr, None)
    if nested is not None:
        if isinstance(default, dict):
            return {
                key: getattr(nested, key, nested_default)
                for key, nested_default in default.items()
            }
        return getattr(nested, fallback_attr, default)
    return getattr(config, fallback_attr, default)


def _apply_clarify_options(
    brief: str,
    classification: RequestClassification,
    options: ClarifyOptions | None,
) -> tuple[str, RequestClassification]:
    if options is None:
        return brief, classification
    updated_brief = options.clarified_brief.strip() if options.clarified_brief else brief
    preferences = classification.preferences
    updates: dict[str, object] = {}
    if options.comparison_targets:
        updates["comparison_targets"] = options.comparison_targets
    if options.source_preference:
        updates["cost_bias"] = options.source_preference
    if options.deliverable_mode:
        try:
            updates["deliverable_mode"] = DeliverableMode(options.deliverable_mode)
        except ValueError:
            pass
    if updates:
        preferences = preferences.model_copy(update=updates)
        classification = classification.model_copy(update={"preferences": preferences})
    if options.scope_adjustment:
        updated_brief = f"{updated_brief}\n\nScope guidance: {options.scope_adjustment}"
    if options.depth_preference:
        updated_brief = f"{updated_brief}\n\nDepth preference: {options.depth_preference}"
    return updated_brief, classification


def _seed_query_templates(brief: str, classification: RequestClassification) -> list[str]:
    focus = classification.preferences.comparison_targets[:3]
    seeds: list[str] = [brief]
    seeds.append(f"{brief} GitHub README architecture benchmark")
    seeds.append(f"{brief} official docs comparison")
    if focus:
        joined = " ".join(focus)
        seeds.append(f"{joined} deep research benchmark comparison")
        seeds.append(f"{joined} GitHub README")
    deduped: list[str] = []
    seen: set[str] = set()
    for seed in seeds:
        cleaned = seed.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped[:5]


def seed_entities_for_brief(
    brief: str,
    classification: RequestClassification,
    config: ResearchConfig,
) -> SeededEntities:
    """Run a lightweight pre-plan search and harvest candidate entity names."""
    flow = _flow()
    actions = [
        SearchAction(
            query=query,
            rationale="Seed concrete entities before planning",
            max_results=min(5, config.max_results_per_query),
        )
        for query in _seed_query_templates(brief, classification)
    ]
    if not actions:
        return SeededEntities()
    try:
        result = flow.execute_searches.submit(
            SupervisorDecision(rationale="Seed entities", search_actions=actions),
            config,
            preferences=classification.preferences,
        ).load()
        candidates = flow.extract_candidates.submit(result.raw_results).load()
    except Exception:
        return SeededEntities()
    entities = SeededEntities()
    for candidate in candidates:
        title = getattr(candidate, "title", "").strip()
        provider = getattr(candidate, "provider", "").strip()
        if title:
            if "bench" in title.lower():
                entities.benchmarks.append(title)
            elif provider in {"github", "exa", "brave"}:
                entities.projects.append(title)
            else:
                entities.key_terms.append(title)
        raw_url = str(getattr(candidate, "url", ""))
        if "github.com" in raw_url and title:
            entities.projects.append(title)
        if "docs." in raw_url and title:
            entities.products.append(title)
    return SeededEntities(
        projects=entities.projects[:5],
        benchmarks=entities.benchmarks[:5],
        products=entities.products[:5],
        companies=entities.companies[:5],
        key_terms=entities.key_terms[:8],
    )


def resolve_config_and_classify(
    brief: str,
    tier: str,
    user_config: ResearchConfig | None,
) -> RunState:
    """Return a ``RunState``, handling clarification inline when needed."""
    classification, resolved_config = _classify_once(brief, tier, user_config)
    current_brief = brief
    clarify_options: ClarifyOptions | None = None
    if classification.needs_clarification and classification.clarification_question:
        clarify_options = _flow().wait(
            name=CLARIFY_BRIEF_WAIT_NAME,
            schema=ClarifyOptions,
            question=classification.clarification_question,
        )
        if isinstance(clarify_options, str):
            clarify_options = ClarifyOptions(clarified_brief=clarify_options)
        current_brief, classification = _apply_clarify_options(
            current_brief, classification, clarify_options
        )
        classification, resolved_config = _classify_once(current_brief, tier, user_config)
    if resolved_config.tier == Tier.DEEP:
        seeded_entities = seed_entities_for_brief(
            current_brief, classification, resolved_config
        )
    else:
        seeded_entities = SeededEntities()
    return RunState(
        brief=current_brief,
        config=resolved_config,
        classification=classification,
        preferences=classification.preferences,
        clarify_options=clarify_options,
        seeded_entities=seeded_entities,
    )


def await_plan_approval_if_required(
    plan: ResearchPlan,
    config: ResearchConfig,
) -> ResearchPlan:
    """Block on a human approval wait when the tier requires one."""
    if not config.require_plan_approval:
        return plan
    approval = _flow().wait(
        name=APPROVE_PLAN_WAIT_NAME,
        schema=PlanApproval,
        question=f"Approve plan for: {plan.goal}?",
    )
    if isinstance(approval, bool):
        approval = PlanApproval(approved=approval)
    if approval is None:
        approval = PlanApproval(approved=True)
    if approval.approved is False:
        if "approval_status" in getattr(type(plan), "model_fields", {}):
            plan = plan.model_copy(update={"approval_status": "rejected"})
        raise ValueError("plan not approved")
    if approval.notes:
        success_criteria = [*plan.success_criteria, f"Approval note: {approval.notes}"]
        plan = plan.model_copy(update={"success_criteria": success_criteria})
    return plan


def build_plan_with_grounding(run_state: RunState) -> ResearchPlan:
    """Build a plan after entity seeding and return the approved plan."""
    flow = _flow()
    brief = run_state.brief
    seeded = run_state.seeded_entities.flattened() if run_state.seeded_entities else []
    if seeded:
        anchors = ", ".join(seeded[:8])
        brief = f"{brief}\n\nNamed entities to anchor on:\n- {anchors}"
    plan = flow.build_plan.submit(
        brief, run_state.classification, run_state.config.tier
    ).load()
    if seeded:
        if "queries" in getattr(type(plan), "model_fields", {}):
            queries = list(plan.queries)
            for entity in seeded[:5]:
                queries.extend(
                    [
                        f"{entity} GitHub README",
                        f"{entity} official docs",
                    ]
                )
            deduped_queries = list(dict.fromkeys(queries))
            plan = plan.model_copy(update={"queries": deduped_queries})
    return await_plan_approval_if_required(plan, run_state.config)


def _run_supervisor_turn(
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    run_state: RunState,
    uncovered_subtopics: list[str] | None,
    unanswered_questions: list[str] | None,
    council_models: list[str],
) -> SupervisorCheckpointResult:
    """Dispatch to the council or single-supervisor branch.

    Uses stable per-iteration checkpoint ids so operators get a replay anchor
    on each turn.
    """
    flow = _flow()
    config = run_state.config

    def _normalize(result: object) -> SupervisorCheckpointResult:
        if isinstance(result, SupervisorCheckpointResult):
            return result
        budget_obj = getattr(result, "budget", None)
        if budget_obj is None:
            budget = getattr(
                SupervisorCheckpointResult.model_fields["budget"],
                "default_factory",
                lambda: None,
            )()
        else:
            budget = budget_obj
            if not hasattr(budget_obj, "model_dump"):
                budget = {
                    "input_tokens": getattr(budget_obj, "input_tokens", 0),
                    "output_tokens": getattr(budget_obj, "output_tokens", 0),
                    "total_tokens": getattr(
                        budget_obj,
                        "total_tokens",
                        getattr(budget_obj, "input_tokens", 0)
                        + getattr(budget_obj, "output_tokens", 0),
                    ),
                    "estimated_cost_usd": getattr(
                        budget_obj, "estimated_cost_usd", 0.0
                    ),
                }
        raw_results = []
        for raw_result in getattr(result, "raw_results", []) or []:
            raw_results.append(
                RawToolResult(
                    tool_name=getattr(raw_result, "tool_name", "tool"),
                    provider=getattr(raw_result, "provider", "unknown"),
                    payload=getattr(raw_result, "payload", {}),
                    ok=getattr(raw_result, "ok", True),
                    error=getattr(raw_result, "error", None),
                )
            )
        return SupervisorCheckpointResult(
            decision=getattr(
                result,
                "decision",
                getattr(
                    result,
                    "output",
                    None,
                )
                or getattr(
                    SupervisorCheckpointResult.model_fields["decision"],
                    "default_factory",
                    lambda: None,
                )(),
            ),
            raw_results=raw_results,
            budget=budget,
            warnings=list(getattr(result, "warnings", []) or []),
        )

    if config.council_mode:
        futures = [
            flow.run_council_generator.submit(
                plan,
                ledger,
                iteration,
                model_name,
                config,
                uncovered_subtopics,
                unanswered_questions=unanswered_questions,
                brief=run_state.brief,
                preferences=run_state.preferences,
                id=f"council_{iteration}_{index}",
            )
            for index, model_name in enumerate(council_models)
        ]
        return _normalize(
            flow.aggregate_council_results.submit(
                [f.load() for f in futures],
            ).load()
        )
    return _normalize(
        flow.run_supervisor.submit(
            plan,
            ledger,
            iteration,
            config,
            uncovered_subtopics,
            unanswered_questions=unanswered_questions,
            brief=run_state.brief,
            preferences=run_state.preferences,
            id=f"supervisor_{iteration}",
        ).load()
    )


def _continue_reason(
    should_stop: bool,
    uncovered_subtopics: list[str],
    unanswered_questions: list[str],
    coverage_delta: float,
    convergence_continue_reason: str | None = None,
) -> str | None:
    """Explain why the loop will keep running (or ``None`` when stopping)."""
    if should_stop:
        return None
    if convergence_continue_reason:
        return convergence_continue_reason
    if unanswered_questions:
        return "remaining unanswered questions: " + ", ".join(unanswered_questions)
    if uncovered_subtopics:
        return "remaining uncovered subtopics: " + ", ".join(uncovered_subtopics)
    return f"coverage improved by {coverage_delta:.2f}; continue exploring"


def _wave_search(
    *,
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    iteration: int,
    run_state: RunState,
    uncovered_subtopics: list[str] | None,
    unanswered_questions: list[str] | None,
    council_models: list[str],
    carryover_actions: list[SearchAction],
) -> WaveSearchResult:
    flow = _flow()
    supervisor_started = perf_counter()
    supervisor_result = _run_supervisor_turn(
        plan,
        ledger,
        iteration,
        run_state,
        uncovered_subtopics,
        unanswered_questions,
        council_models,
    )
    supervisor_latency_ms = int((perf_counter() - supervisor_started) * 1000)
    decision = supervisor_result.decision
    decision_search_actions = list(decision.search_actions)
    search_actions = list(decision_search_actions)
    if carryover_actions:
        search_actions = [*carryover_actions, *search_actions]
        decision = decision.model_copy(update={"search_actions": search_actions})
    if search_actions:
        search_started = perf_counter()
        search_result = flow.execute_searches.submit(
            SupervisorDecision(
                rationale=decision.rationale,
                search_actions=search_actions,
                status=decision.status,
            ),
            run_state.config,
            preferences=run_state.preferences,
        ).load()
        search_latency_ms = int((perf_counter() - search_started) * 1000)
    else:
        search_result = SearchExecutionResult()
        search_latency_ms = 0
    combined_raw_results = [*supervisor_result.raw_results, *search_result.raw_results]
    return WaveSearchResult(
        decision=decision,
        raw_results=combined_raw_results,
        supervisor_cost=supervisor_result.budget.estimated_cost_usd,
        search_cost=search_result.budget.estimated_cost_usd,
        warnings=[*supervisor_result.warnings, *getattr(search_result, "warnings", [])],
        tool_calls=build_tool_call_records(
            [*supervisor_result.raw_results, *search_result.raw_results]
            if decision_search_actions
            else supervisor_result.raw_results
        ),
        step_latencies_ms={
            "supervisor": supervisor_latency_ms,
            "search": search_latency_ms,
        },
        total_tokens=(
            getattr(supervisor_result.budget, "total_tokens", 0)
            + getattr(search_result.budget, "total_tokens", 0)
        ),
    )


def _wave_triage(
    raw_results: list[RawToolResult],
    plan: ResearchPlan,
    config: ResearchConfig,
    iteration: int,
) -> WaveTriageResult:
    flow = _flow()
    candidates = flow.extract_candidates.submit(raw_results).load()
    relevance_started = perf_counter()
    relevance_result = flow.score_relevance.submit(candidates, plan, config).load()
    relevance_latency_ms = int((perf_counter() - relevance_started) * 1000)
    scored_candidates = [
        candidate
        if getattr(candidate, "iteration_added", None) is not None
        or not hasattr(candidate, "model_copy")
        else candidate.model_copy(update={"iteration_added": iteration})
        for candidate in relevance_result.candidates
    ]
    return WaveTriageResult(
        candidates=scored_candidates,
        relevance_cost=relevance_result.budget.estimated_cost_usd,
        step_latencies_ms={"relevance": relevance_latency_ms},
        total_tokens=getattr(relevance_result.budget, "total_tokens", 0),
    )


def _wave_enrich(
    candidates: list,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> WaveEnrichResult:
    flow = _flow()
    merge_started = perf_counter()
    updated_ledger = flow.update_ledger.submit(
        candidates,
        ledger,
        config=config,
    ).load()
    merge_latency_ms = int((perf_counter() - merge_started) * 1000)
    enrich_started = perf_counter()
    updated_ledger = flow.enrich_candidates.submit(updated_ledger, config).load()
    enrich_latency_ms = int((perf_counter() - enrich_started) * 1000)
    return WaveEnrichResult(
        ledger=updated_ledger,
        step_latencies_ms={"merge": merge_latency_ms, "enrich": enrich_latency_ms},
    )


def _wave_score(
    ledger: EvidenceLedger,
    plan: ResearchPlan,
    config: ResearchConfig,
) -> WaveScoreResult:
    flow = _flow()
    coverage_started = perf_counter()
    coverage = flow.score_coverage.submit(ledger, plan, config).load()
    coverage_latency_ms = int((perf_counter() - coverage_started) * 1000)
    if not isinstance(coverage, CoverageScore):
        coverage = CoverageScore(
            subtopic_coverage=getattr(coverage, "subtopic_coverage", coverage.total),
            plan_fidelity=getattr(coverage, "plan_fidelity", coverage.total),
            source_diversity=getattr(coverage, "source_diversity", coverage.total),
            evidence_density=getattr(coverage, "evidence_density", coverage.total),
            total=getattr(coverage, "total", 0.0),
            uncovered_subtopics=list(getattr(coverage, "uncovered_subtopics", [])),
            unanswered_questions=list(
                getattr(coverage, "unanswered_questions", [])
            ),
        )
    return WaveScoreResult(
        coverage=coverage,
        step_latencies_ms={"coverage": coverage_latency_ms},
    )


_MAX_SYNTHESIS_CLAIM_QUERIES = 5


def _lightweight_synthesis_claims(ledger: EvidenceLedger) -> list[str]:
    """Extract short pseudo-claim strings from the top selected snippets.

    Cheap and deterministic: the first sentence of the first snippet of each of
    the top selected candidates. We treat these as claims needing corroboration.
    """
    claims: list[str] = []
    selected = list(getattr(ledger, "selected", []) or [])
    # Stable ordering by selection_score then key for determinism across replays.
    selected.sort(
        key=lambda candidate: (
            -float(candidate.raw_metadata.get("selection_score", 0.0)),
            candidate.key,
        )
    )
    for candidate in selected[:_MAX_SYNTHESIS_CLAIM_QUERIES]:
        if not candidate.snippets:
            continue
        text = candidate.snippets[0].text.strip()
        if not text:
            continue
        # First sentence, clipped to keep the query short.
        first_sentence = text.split(".")[0].strip()
        if len(first_sentence) < 12:
            continue
        claims.append(first_sentence[:180])
    return claims


def _wave_feedback(
    coverage: CoverageScore,
    *,
    plan: ResearchPlan,
    config: ResearchConfig,
    ledger: EvidenceLedger | None = None,
    feedback_iteration_count: int = 0,
) -> WaveFeedbackResult:
    feedback_queries: list[str] = []
    for subtopic in coverage.uncovered_subtopics[:2]:
        feedback_queries.append(f"{plan.goal} {subtopic}")
    for question in coverage.unanswered_questions[:2]:
        feedback_queries.append(question)

    # Wave 4.5: synthesis-to-unsupported-claims feedback, deep tier only, hard
    # capped by SelectionPolicyConfig.feedback_loop_max_iterations.
    synthesis_claims: list[str] = []
    policy = config.selection_policy
    max_feedback_iters = (
        policy.get("feedback_loop_max_iterations", 0)
        if isinstance(policy, dict)
        else getattr(policy, "feedback_loop_max_iterations", 0)
    )
    synthesis_gate = (
        config.tier == Tier.DEEP
        and ledger is not None
        and max_feedback_iters > 0
        and feedback_iteration_count < max_feedback_iters
    )
    if synthesis_gate:
        synthesis_claims = _lightweight_synthesis_claims(ledger)
        for claim in synthesis_claims[:_MAX_SYNTHESIS_CLAIM_QUERIES]:
            feedback_queries.append(f"{claim} evidence")

    next_feedback_count = feedback_iteration_count + (1 if synthesis_claims else 0)

    carryover_actions = [
        SearchAction(
            query=query,
            rationale="Gap-fill follow-up query generated from coverage feedback",
            max_results=min(5, config.max_results_per_query),
        )
        for query in feedback_queries
    ]
    if not carryover_actions:
        return WaveFeedbackResult(
            carryover_actions=[],
            reason=None,
            feedback_iteration_count=next_feedback_count,
        )
    reason_parts = [f"{len(carryover_actions)} gap-fill queries"]
    if synthesis_claims:
        reason_parts.append(
            f"{len(synthesis_claims)} synthesis claim queries "
            f"(iter {next_feedback_count}/{max_feedback_iters})"
        )
    return WaveFeedbackResult(
        carryover_actions=carryover_actions,
        reason=f"generated {', '.join(reason_parts)}",
        feedback_iteration_count=next_feedback_count,
    )


def run_iteration_loop(
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> IterationLoopOutput:
    """Run the research loop as waves until a composite stop condition hits."""
    flow = _flow()
    config = run_state.config
    ledger = EvidenceLedger()
    iteration_history: tuple[IterationRecord, ...] = ()
    provider_usage_summary: dict[str, int] = {}
    spent_usd = 0.0
    total_tokens = 0
    active_elapsed_ms = 0
    wall_elapsed_seconds = 0
    uncovered_subtopics: list[str] | None = None
    unanswered_questions: list[str] | None = None
    carryover_actions: list[SearchAction] = []
    council_models = (
        [config.supervisor_model] * config.council_size if config.council_mode else []
    )
    stop_reason = StopReason.MAX_ITERATIONS
    _replans_used = 0
    stall_count = 0
    feedback_iteration_count = 0

    with span("iteration_loop", max_iterations=config.max_iterations):
        for iteration in range(config.max_iterations):
            with span("iteration", iteration=iteration):
                wave_search = _wave_search(
                    plan=plan,
                    ledger=ledger,
                    iteration=iteration,
                    run_state=run_state,
                    uncovered_subtopics=uncovered_subtopics,
                    unanswered_questions=unanswered_questions,
                    council_models=council_models,
                    carryover_actions=carryover_actions,
                )
                decision = wave_search.decision

                # Supervisor can signal completion directly via status field.
                if decision.status == "complete":
                    active_elapsed_ms += sum(wave_search.step_latencies_ms.values())
                    wall_clock = flow.snapshot_wall_clock.submit(
                        stamp.started_at,
                    ).load()
                    wall_elapsed_seconds = getattr(
                        wall_clock,
                        "wall_elapsed_seconds",
                        getattr(wall_clock, "elapsed_seconds", 0),
                    )
                    iteration_record = IterationRecord(
                        iteration=iteration,
                        new_candidate_count=0,
                        accepted_candidate_count=len(ledger.selected),
                        rejected_candidate_count=len(ledger.rejected),
                        coverage=iteration_history[-1].coverage
                        if iteration_history
                        else 0.0,
                        coverage_delta=0.0,
                        uncovered_subtopics=uncovered_subtopics or [],
                        unanswered_questions=unanswered_questions or [],
                        estimated_cost_usd=wave_search.supervisor_cost,
                        tool_calls=wave_search.tool_calls,
                        warnings=list(wave_search.warnings),
                        stop_reason=StopReason.SUPERVISOR_COMPLETE,
                        step_costs_usd={
                            "supervisor": wave_search.supervisor_cost,
                            "search": wave_search.search_cost,
                        },
                        step_latencies_ms=wave_search.step_latencies_ms,
                    )
                    if "active_elapsed_ms" in IterationRecord.model_fields:
                        iteration_record = iteration_record.model_copy(
                            update={"active_elapsed_ms": active_elapsed_ms}
                        )
                    iteration_history = (*iteration_history, iteration_record)
                    stop_reason = StopReason.SUPERVISOR_COMPLETE
                    break

                spent_usd = spent_usd + wave_search.supervisor_cost + wave_search.search_cost
                total_tokens += wave_search.total_tokens
                provider_usage_summary = merge_provider_counts(
                    provider_usage_summary, wave_search.raw_results
                )

                previous_ledger_size = len(getattr(ledger, "selected", [])) + len(
                    getattr(ledger, "rejected", [])
                )
                wave_triage = _wave_triage(
                    wave_search.raw_results,
                    plan,
                    config,
                    iteration,
                )
                spent_usd = spent_usd + wave_triage.relevance_cost
                total_tokens += wave_triage.total_tokens
                wave_enrich = _wave_enrich(wave_triage.candidates, ledger, config)
                ledger = wave_enrich.ledger
                wave_score = _wave_score(ledger, plan, config)
                coverage = wave_score.coverage
                previous_coverage = (
                    iteration_history[-1].coverage if iteration_history else 0.0
                )
                coverage_delta = round(coverage.total - previous_coverage, 6)
                if coverage_delta <= 0:
                    stall_count += 1
                else:
                    stall_count = 0
                uncovered_subtopics = list(coverage.uncovered_subtopics)
                unanswered_questions = list(coverage.unanswered_questions)
                feedback = _wave_feedback(
                    coverage,
                    plan=plan,
                    config=config,
                    ledger=ledger,
                    feedback_iteration_count=feedback_iteration_count,
                )
                carryover_actions = feedback.carryover_actions
                feedback_iteration_count = feedback.feedback_iteration_count
                metric("iteration_coverage", coverage.total, iteration=iteration)
                iteration_cost = (
                    wave_search.supervisor_cost
                    + wave_search.search_cost
                    + wave_triage.relevance_cost
                )
                metric("iteration_cost_usd", iteration_cost, iteration=iteration)
                metric("iteration_candidates", len(wave_triage.candidates), iteration=iteration)
                tool_calls = wave_search.tool_calls
                warnings = list(wave_search.warnings)
                if feedback.reason:
                    warnings.append(feedback.reason)
                diversity_warning = detect_source_diversity_warning(ledger)
                if diversity_warning is not None:
                    warnings.append(diversity_warning)
                ledger_payload = (
                    ledger.model_dump(mode="json")
                    if hasattr(ledger, "model_dump")
                    else {
                        "selected": [vars(item) for item in getattr(ledger, "selected", [])],
                        "rejected": [vars(item) for item in getattr(ledger, "rejected", [])],
                    }
                )
                context_budget_used_ratio = round(
                    len(json.dumps(ledger_payload, allow_nan=False))
                    / config.supervisor_context_budget_chars,
                    4,
                )
                iteration_record = IterationRecord(
                    iteration=iteration,
                    new_candidate_count=max(
                        0,
                        (len(getattr(ledger, "selected", []))
                         + len(getattr(ledger, "rejected", [])))
                        - previous_ledger_size,
                    ),
                    accepted_candidate_count=len(ledger.selected),
                    rejected_candidate_count=len(ledger.rejected),
                    coverage=coverage.total,
                    coverage_delta=coverage_delta,
                    uncovered_subtopics=uncovered_subtopics,
                    unanswered_questions=unanswered_questions,
                    estimated_cost_usd=iteration_cost,
                    tool_calls=tool_calls,
                    warnings=warnings,
                    context_budget_used_ratio=context_budget_used_ratio,
                    step_costs_usd={
                        "supervisor": wave_search.supervisor_cost,
                        "search": wave_search.search_cost,
                        "relevance": wave_triage.relevance_cost,
                        "coverage": 0.0,
                    },
                    step_latencies_ms={
                        **wave_search.step_latencies_ms,
                        **wave_triage.step_latencies_ms,
                        **wave_enrich.step_latencies_ms,
                        **wave_score.step_latencies_ms,
                    },
                )
                active_elapsed_ms += sum(iteration_record.step_latencies_ms.values())
                if "active_elapsed_ms" in IterationRecord.model_fields:
                    iteration_record = iteration_record.model_copy(
                        update={"active_elapsed_ms": active_elapsed_ms}
                    )
                if "wall_elapsed_ms" in IterationRecord.model_fields:
                    iteration_record = iteration_record.model_copy(
                        update={"wall_elapsed_ms": wall_elapsed_seconds * 1000}
                    )

                wall_clock = flow.snapshot_wall_clock.submit(
                    stamp.started_at,
                ).load()
                wall_elapsed_seconds = getattr(
                    wall_clock,
                    "wall_elapsed_seconds",
                    getattr(wall_clock, "elapsed_seconds", 0),
                )
                convergence_config = _config_value(
                    config,
                    "convergence",
                    "coverage_threshold",
                    {
                        "coverage_threshold": getattr(
                            config, "convergence_min_coverage", 0.60
                        ),
                        "strong_coverage_threshold": 0.92,
                        "marginal_gain_threshold": getattr(
                            config, "convergence_epsilon", 0.05
                        ),
                        "max_stall_count": 3,
                        "min_considered_floor": 100,
                        "min_selected_for_stop": 12,
                    },
                )
                active_time_limit_seconds = getattr(
                    getattr(config, "active_time", None),
                    "active_time_box_seconds",
                    getattr(config, "time_box_seconds", 0),
                )
                token_budget_total = getattr(
                    getattr(config, "convergence", None),
                    "token_budget",
                    0,
                )
                considered_count = len(getattr(ledger, "considered", []))
                source_group_count = len(
                    {
                        getattr(candidate, "source_group", None)
                        or getattr(getattr(candidate, "source_kind", None), "value", None)
                        for candidate in getattr(ledger, "selected", [])
                    }
                    - {None}
                )
                signal = ConvergenceSignal(
                    coverage_total=getattr(coverage, "total", 0.0),
                    subtopic_coverage=getattr(
                        coverage, "subtopic_coverage", getattr(coverage, "total", 0.0)
                    ),
                    unanswered_questions_count=len(unanswered_questions),
                    considered_count=considered_count,
                    selected_count=len(getattr(ledger, "selected", [])),
                    source_group_count=source_group_count,
                    new_candidate_count=iteration_record.new_candidate_count,
                    marginal_info_gain=max(0.0, coverage_delta),
                    stall_count=stall_count,
                    spent_usd=spent_usd,
                    active_elapsed_seconds=getattr(
                        wall_clock, "active_elapsed_seconds", 0
                    ),
                    wall_elapsed_seconds=wall_elapsed_seconds,
                    total_tokens=total_tokens,
                    token_budget_remaining_ratio=(
                        1.0
                        if token_budget_total <= 0
                        else max(0.0, 1.0 - (total_tokens / token_budget_total))
                    ),
                )
                decision_stop = flow.check_convergence(
                    signal,
                    list(iteration_history),
                    max_iterations=config.max_iterations,
                    coverage_threshold=convergence_config["coverage_threshold"],
                    strong_coverage_threshold=convergence_config[
                        "strong_coverage_threshold"
                    ],
                    marginal_gain_threshold=convergence_config[
                        "marginal_gain_threshold"
                    ],
                    max_stall_count=convergence_config["max_stall_count"],
                    budget_limit_usd=config.cost_budget_usd,
                    active_time_limit_seconds=active_time_limit_seconds,
                    min_considered_floor=convergence_config["min_considered_floor"],
                    min_selected_for_stop=convergence_config["min_selected_for_stop"],
                    elapsed_seconds=wall_elapsed_seconds,
                    spent_usd=spent_usd,
                    epsilon=convergence_config["marginal_gain_threshold"],
                    min_coverage=convergence_config["coverage_threshold"],
                    time_limit_seconds=active_time_limit_seconds,
                    new_candidate_count=iteration_record.new_candidate_count,
                )
                continue_reason = _continue_reason(
                    decision_stop.should_stop,
                    uncovered_subtopics,
                    unanswered_questions,
                    coverage_delta,
                    getattr(decision_stop, "continue_reason", None),
                )
                iteration_record = iteration_record.model_copy(
                    update={
                        "continue_reason": continue_reason,
                        "stop_reason": (
                            decision_stop.reason if decision_stop.should_stop else None
                        ),
                    }
                )
                iteration_history = (*iteration_history, iteration_record)
                log_kwargs = {
                    "iteration": iteration,
                    "coverage": coverage.total,
                    "coverage_delta": coverage_delta,
                    "uncovered_subtopics": uncovered_subtopics,
                    "new_candidate_count": iteration_record.new_candidate_count,
                    "accepted_candidate_count": iteration_record.accepted_candidate_count,
                    "rejected_candidate_count": iteration_record.rejected_candidate_count,
                    "tool_summaries": [tc.summary for tc in tool_calls],
                    "stop_reason": iteration_record.stop_reason,
                    "continue_reason": continue_reason,
                    "spent_usd": round(spent_usd, 6),
                }
                if unanswered_questions:
                    log_kwargs["unanswered_questions"] = unanswered_questions
                diagnostics = getattr(decision_stop, "diagnostics", None)
                if diagnostics:
                    log_kwargs["convergence"] = diagnostics
                flow.log(**log_kwargs)
                if decision_stop.should_stop:
                    # Check replan gate before accepting the stop
                    if (
                        config.max_replans > 0
                        and decision_stop.reason
                        in (StopReason.LOOP_STALL, StopReason.DIMINISHING_RETURNS)
                        and _replans_used < config.max_replans
                    ):
                        replan_decision = flow.evaluate_replan.submit(
                            plan, coverage, list(iteration_history), config
                        ).load()
                        if replan_decision.should_replan:
                            updates: dict[str, object] = {}
                            if replan_decision.updated_subtopics:
                                updates["subtopics"] = replan_decision.updated_subtopics
                            if replan_decision.updated_queries:
                                updates["queries"] = replan_decision.updated_queries
                            if updates:
                                plan = plan.model_copy(update=updates)
                            uncovered_subtopics = list(coverage.uncovered_subtopics)
                            unanswered_questions = list(coverage.unanswered_questions)
                            _replans_used += 1
                            flow.log(
                                replan_triggered=True,
                                replan_number=_replans_used,
                                rationale=replan_decision.rationale,
                                updated_subtopics=bool(
                                    replan_decision.updated_subtopics
                                ),
                                updated_queries=bool(replan_decision.updated_queries),
                                iteration=iteration,
                            )
                            continue
                        else:
                            flow.log(
                                replan_triggered=False,
                                rationale=replan_decision.rationale,
                                iteration=iteration,
                            )
                    # Fall back to MAX_ITERATIONS if the decision didn't name a reason.
                    stop_reason = decision_stop.reason or StopReason.MAX_ITERATIONS
                    break

    return IterationLoopOutput(
        ledger=ledger,
        iteration_history=iteration_history,
        provider_usage_summary=provider_usage_summary,
        spent_usd=spent_usd,
        stop_reason=stop_reason,
        council_models=council_models,
        active_elapsed_seconds=active_elapsed_ms // 1000,
        wall_elapsed_seconds=wall_elapsed_seconds,
        total_tokens=total_tokens,
    )


def render_deliverable(
    selection: SelectionGraph,
    iteration_output: IterationLoopOutput,
    plan: ResearchPlan,
    run_state: RunState,
    stamp,
) -> tuple[list[RenderPayload], float, list[str]]:
    """Render deliverable(s) and return ``(renders, added_cost, degradations)``."""
    flow = _flow()
    config = run_state.config
    deliverable_mode = run_state.preferences.deliverable_mode
    degradations: list[str] = []

    if deliverable_mode == DeliverableMode.RESEARCH_PACKAGE:
        reading_future = flow.write_reading_path.submit(
            selection,
            iteration_output.ledger,
            plan,
            config,
            preferences=run_state.preferences,
        )
        backing_future = flow.write_backing_report.submit(
            selection,
            iteration_output.ledger,
            plan,
            IterationTrace(iterations=list(iteration_output.iteration_history)),
            iteration_output.provider_usage_summary,
            iteration_output.stop_reason,
            config,
            preferences=run_state.preferences,
        )
        reading_result = reading_future.load()
        backing_result = backing_future.load()
        added_cost = (
            reading_result.budget.estimated_cost_usd
            + backing_result.budget.estimated_cost_usd
        )
        return [reading_result.render, backing_result.render], added_cost, degradations

    # Non-default modes fall through to a single full-report render. Reuse the
    # run stamp so the scaffold RunSummary shares run_id with the final one.
    partial_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="rendering",
        estimated_cost_usd=round(iteration_output.spent_usd, 6),
        elapsed_seconds=0,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=None,
    )
    summary_updates: dict[str, object] = {}
    if "active_elapsed_seconds" in RunSummary.model_fields:
        summary_updates["active_elapsed_seconds"] = iteration_output.active_elapsed_seconds
    if "wall_elapsed_seconds" in RunSummary.model_fields:
        summary_updates["wall_elapsed_seconds"] = iteration_output.wall_elapsed_seconds
    if summary_updates:
        partial_summary = partial_summary.model_copy(update=summary_updates)
    partial_package = InvestigationPackage(
        run_summary=partial_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=[],
        preferences=run_state.preferences,
    )
    full_result = flow.write_full_report.submit(
        partial_package, config, preferences=run_state.preferences
    ).load()
    if deliverable_mode not in _FULLY_SUPPORTED_MODES:
        degradations.append(
            f"Deliverable mode '{deliverable_mode.value}' not yet fully supported"
            f" -- rendered as 'final_report' with {deliverable_mode.value} context"
        )
    return [full_result.render], full_result.budget.estimated_cost_usd, degradations


def run_critique_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    selection: SelectionGraph,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> CritiqueBundle:
    """Run reviewer and apply revisions when critique is enabled."""
    flow = _flow()
    if not config.critique_enabled:
        return CritiqueBundle(renders=renders, critique_result=None, spent_usd=0.0)
    critique_checkpoint = flow.critique_reports.submit(
        renders, plan, selection, ledger, config
    ).load()
    critique_result = critique_checkpoint.critique
    revision_checkpoint = flow.apply_revisions.submit(
        renders, critique_result, plan, config
    ).load()
    return CritiqueBundle(
        renders=revision_checkpoint.renders,
        critique_result=critique_result,
        spent_usd=(
            critique_checkpoint.budget.estimated_cost_usd
            + revision_checkpoint.budget.estimated_cost_usd
        ),
    )


def run_judges_if_enabled(
    renders: list[RenderPayload],
    plan: ResearchPlan,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> JudgeBundle:
    """Run grounding + coherence judges in parallel when enabled."""
    flow = _flow()
    if not config.judge_enabled:
        return JudgeBundle(grounding_result=None, coherence_result=None, spent_usd=0.0)
    grounding_future = flow.verify_grounding.submit(renders, ledger, config)
    coherence_future = flow.verify_coherence.submit(renders, plan, config)
    grounding_checkpoint = grounding_future.load()
    coherence_checkpoint = coherence_future.load()
    return JudgeBundle(
        grounding_result=grounding_checkpoint.grounding,
        coherence_result=coherence_checkpoint.coherence,
        spent_usd=(
            grounding_checkpoint.budget.estimated_cost_usd
            + coherence_checkpoint.budget.estimated_cost_usd
        ),
    )


def assemble_final_package(
    *,
    stamp,
    finalization,
    run_state: RunState,
    plan: ResearchPlan,
    iteration_output: IterationLoopOutput,
    selection: SelectionGraph,
    renders: list[RenderPayload],
    spent_usd: float,
    critique_result: CritiqueResult | None,
    grounding_result: GroundingResult | None,
    coherence_result: CoherenceResult | None,
    degradations: list[str],
) -> InvestigationPackage:
    """Build the final RunSummary and hand everything to the assemble checkpoint."""
    flow = _flow()
    config = run_state.config
    run_summary = RunSummary(
        run_id=stamp.run_id,
        brief=run_state.brief,
        tier=config.tier,
        stop_reason=iteration_output.stop_reason,
        status="completed",
        estimated_cost_usd=round(spent_usd, 6),
        elapsed_seconds=finalization.elapsed_seconds,
        iteration_count=len(iteration_output.iteration_history),
        provider_usage_summary=iteration_output.provider_usage_summary,
        council_enabled=config.council_mode,
        council_size=config.council_size if config.council_mode else 1,
        council_models=iteration_output.council_models,
        started_at=stamp.started_at,
        completed_at=finalization.completed_at,
    )
    summary_updates = {}
    if "active_elapsed_seconds" in RunSummary.model_fields:
        summary_updates["active_elapsed_seconds"] = getattr(
            finalization, "active_elapsed_seconds", iteration_output.active_elapsed_seconds
        )
    if "wall_elapsed_seconds" in RunSummary.model_fields:
        summary_updates["wall_elapsed_seconds"] = getattr(
            finalization, "wall_elapsed_seconds", finalization.elapsed_seconds
        )
    if summary_updates:
        run_summary = run_summary.model_copy(update=summary_updates)
    return flow.assemble_package.submit(
        run_summary=run_summary,
        research_plan=plan,
        evidence_ledger=iteration_output.ledger,
        selection_graph=selection,
        iteration_trace=IterationTrace(
            iterations=list(iteration_output.iteration_history)
        ),
        renders=renders,
        render_settings=RenderSettingsSnapshot(writer_model=config.writer_model),
        critique_result=critique_result,
        grounding_result=grounding_result,
        coherence_result=coherence_result,
        preferences=run_state.preferences,
        preference_degradations=degradations,
    ).load()
