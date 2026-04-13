"""Per-iteration wave functions: search, triage, enrich, score, feedback."""

from __future__ import annotations

from time import perf_counter

from deep_research.config import ResearchConfig
from deep_research.enums import Tier
from deep_research.flow._types import (
    RunState,
    WaveEnrichResult,
    WaveFeedbackResult,
    WaveScoreResult,
    WaveSearchResult,
    WaveTriageResult,
    _flow,
    build_tool_call_records,
)
from deep_research.models import (
    CoverageScore,
    EvidenceLedger,
    ResearchPlan,
    SearchAction,
    SearchExecutionResult,
    SupervisorCheckpointResult,
    SupervisorDecision,
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
    """Run supervisor and execute searches; return combined raw results."""
    flow = _flow()
    config = run_state.config
    supervisor_started = perf_counter()

    # --- council vs single supervisor ---
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
        supervisor_result: SupervisorCheckpointResult = flow.aggregate_council_results.submit(
            [f.load() for f in futures],
        ).load()
    else:
        supervisor_result = flow.run_supervisor.submit(
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
    # --- end council vs single supervisor ---

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
        warnings=[*supervisor_result.warnings, *search_result.warnings],
        tool_calls=build_tool_call_records(
            combined_raw_results if decision_search_actions else supervisor_result.raw_results
        ),
        step_latencies_ms={
            "supervisor": supervisor_latency_ms,
            "search": search_latency_ms,
        },
        total_tokens=supervisor_result.budget.total_tokens + search_result.budget.total_tokens,
    )


def _wave_triage(
    raw_results: list,
    plan: ResearchPlan,
    config: ResearchConfig,
    iteration: int,
) -> WaveTriageResult:
    """Extract candidates and score relevance; return scored candidates."""
    flow = _flow()
    candidates = flow.extract_candidates.submit(raw_results).load()
    relevance_started = perf_counter()
    relevance_result = flow.score_relevance.submit(candidates, plan, config).load()
    relevance_latency_ms = int((perf_counter() - relevance_started) * 1000)
    scored_candidates = [
        candidate
        if candidate.iteration_added is not None
        else candidate.model_copy(update={"iteration_added": iteration})
        for candidate in relevance_result.candidates
    ]
    return WaveTriageResult(
        candidates=scored_candidates,
        relevance_cost=relevance_result.budget.estimated_cost_usd,
        step_latencies_ms={"relevance": relevance_latency_ms},
        total_tokens=relevance_result.budget.total_tokens,
    )


def _wave_enrich(
    candidates: list,
    ledger: EvidenceLedger,
    config: ResearchConfig,
) -> WaveEnrichResult:
    """Merge candidates into the ledger and enrich; return updated ledger."""
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
    """Score coverage of the current ledger against the plan."""
    flow = _flow()
    coverage_started = perf_counter()
    coverage = flow.score_coverage.submit(ledger, plan, config).load()
    coverage_latency_ms = int((perf_counter() - coverage_started) * 1000)
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
    selected = list(ledger.selected)
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
    """Generate gap-fill carryover search actions from coverage feedback."""
    feedback_queries: list[str] = []
    for subtopic in coverage.uncovered_subtopics[:2]:
        feedback_queries.append(f"{plan.goal} {subtopic}")
    for question in coverage.unanswered_questions[:2]:
        feedback_queries.append(question)

    # Wave 4.5: synthesis-to-unsupported-claims feedback, deep tier only, hard
    # capped by SelectionPolicyConfig.feedback_loop_max_iterations.
    synthesis_claims: list[str] = []
    max_feedback_iters = config.selection_policy.feedback_loop_max_iterations
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
