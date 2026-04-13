"""Config resolution and request classification helpers."""

from __future__ import annotations

from deep_research.config import ResearchConfig
from deep_research.enums import DeliverableMode, Tier
from deep_research.flow._types import (
    CLARIFY_BRIEF_WAIT_NAME,
    APPROVE_PLAN_WAIT_NAME,
    ClarifyOptions,
    PlanApproval,
    RunState,
    _flow,
)
from deep_research.models import (
    RequestClassification,
    ResearchPlan,
    SearchAction,
    SeededEntities,
    SupervisorDecision,
)


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


def seed_entities_for_brief(
    brief: str,
    classification: RequestClassification,
    config: ResearchConfig,
) -> SeededEntities:
    """Run a lightweight pre-plan search and harvest candidate entity names."""
    flow = _flow()
    focus = classification.preferences.comparison_targets[:3]
    raw_seeds = [
        brief,
        f"{brief} GitHub README architecture benchmark",
        f"{brief} official docs comparison",
    ]
    if focus:
        joined = " ".join(focus)
        raw_seeds.append(f"{joined} deep research benchmark comparison")
        raw_seeds.append(f"{joined} GitHub README")
    seed_queries = list(dict.fromkeys(seed.strip() for seed in raw_seeds if seed.strip()))[:5]
    actions = [
        SearchAction(
            query=query,
            rationale="Seed concrete entities before planning",
            max_results=min(5, config.max_results_per_query),
        )
        for query in seed_queries
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
        title = candidate.title.strip()
        provider = candidate.provider.strip()
        if title:
            if "bench" in title.lower():
                entities.benchmarks.append(title)
            elif provider in {"github", "exa", "brave"}:
                entities.projects.append(title)
            else:
                entities.key_terms.append(title)
        raw_url = str(candidate.url)
        if title and "github.com" in raw_url:
            entities.projects.append(title)
        if title and "docs." in raw_url:
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
        if clarify_options is not None:
            if clarify_options.clarified_brief:
                current_brief = clarify_options.clarified_brief.strip()
            updates: dict[str, object] = {}
            if clarify_options.comparison_targets:
                updates["comparison_targets"] = clarify_options.comparison_targets
            if clarify_options.source_preference:
                updates["cost_bias"] = clarify_options.source_preference
            if clarify_options.deliverable_mode:
                try:
                    updates["deliverable_mode"] = DeliverableMode(clarify_options.deliverable_mode)
                except ValueError:
                    pass
            if updates:
                preferences = classification.preferences.model_copy(update=updates)
                classification = classification.model_copy(update={"preferences": preferences})
            if clarify_options.scope_adjustment:
                current_brief = f"{current_brief}\n\nScope guidance: {clarify_options.scope_adjustment}"
            if clarify_options.depth_preference:
                current_brief = f"{current_brief}\n\nDepth preference: {clarify_options.depth_preference}"
        classification, resolved_config = _classify_once(current_brief, tier, user_config)
    seeded_entities = seed_entities_for_brief(
        current_brief, classification, resolved_config
    )
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
    ) or PlanApproval(approved=True)
    if not approval.approved:
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
        queries = list(plan.queries)
        for entity in seeded[:5]:
            queries.append(f"{entity} GitHub README")
            queries.append(f"{entity} official docs")
        plan = plan.model_copy(update={"queries": list(dict.fromkeys(queries))})
    return await_plan_approval_if_required(plan, run_state.config)
