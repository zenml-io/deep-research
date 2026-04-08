# Research Preferences: User Intent Layer

**Date:** 2026-04-09
**Status:** Approved

---

## Problem Statement

Users can tell the system *what* to research, but they cannot reliably control *how* the system researches, which sources it should favor or avoid, or what final output shape it should optimize for.

Today the brief influences the plan, but much of that intent is lost before execution:

- `plan.py:14` discards the classification (`del classification`)
- The supervisor sees the plan and ledger but not the raw brief or user preferences
- Output format is fixed: always reading_path + backing_report
- Coverage weights are hardcoded (1/3 each)
- `allowed_source_groups` exists in the schema but nothing reads it

## Design Principle

**LLM-driven with hard guardrails.**

- The LLM decides *what* to do: which providers to target, how many queries, what to prioritize, when to stop. The code provides *tools and constraints*, not *decisions*.
- Hard guardrails are few and explicit: budget limits, time limits, user exclusions, convergence thresholds. Everything else is advisory context the LLM reasons about.
- When in doubt, give the LLM information and let it decide, rather than writing an `if/else` in Python.
- Simplify the code. Less deterministic, more agentic, without losing reliability and good practices.

## Chosen Approach

Introduce a `ResearchPreferences` object as a durable artifact of the initial prompt. All downstream stages either honor it or explicitly degrade when unsupported. The existing durable flow checkpoint graph is preserved.

## Scope

**In scope:**
- Parse user intent from the brief via the existing classifier (expanded, no new checkpoint)
- Preserve intent through all checkpoints (plan, supervisor, search, render)
- Enforce provider/source exclusions at runtime (hard constraints)
- Expose provider/source preferences as advisory LLM context (soft preferences)
- Support multiple deliverable modes with free-form LLM-driven output structure
- Persist preferences and degradation metadata in the final package
- Tests proving steerability

**Out of scope:**
- Large provider expansion (no new social/news/repo providers)
- Replacing the convergence loop
- Free-form tool autonomy
- UI/product changes beyond API/package outputs
- Mid-stream user interaction (future work)

---

## Section 1: ResearchPreferences Model

### Enums

```python
class SourceGroup(str, Enum):
    papers = "papers"       # arxiv, semantic_scholar
    web = "web"             # brave, exa
    news = "news"           # maps to web providers until dedicated providers exist
    repos = "repos"         # maps to web providers until dedicated providers exist
    social = "social"       # no providers yet; classifier captures intent, system degrades gracefully

class DeliverableMode(str, Enum):
    research_package = "research_package"      # reading_path + backing_report (current default)
    final_report = "final_report"             # single prose report
    comparison_memo = "comparison_memo"        # comparison-focused output
    recommendation_brief = "recommendation_brief"  # recommendation-focused output
    answer_only = "answer_only"               # concise direct answer

class PlanningMode(str, Enum):
    broad_scan = "broad_scan"
    decision_support = "decision_support"
    comparison = "comparison"
    timeline = "timeline"
    deep_dive = "deep_dive"
```

All enum values are defined and extractable by the classifier from day one. The rendering layer handles what it can and degrades gracefully for what it can't, logging degradations. This way the user's intent is always captured accurately, even if the system can't fully satisfy it yet.

### Preferences Model

```python
class ResearchPreferences(StrictBaseModel):
    audience: str | None = None                # "technical", "executive", "general public"
    freshness: str | None = None               # "last_week", "last_month", "any"
    deliverable_mode: DeliverableMode = DeliverableMode.research_package
    preferred_source_groups: list[SourceGroup] = Field(default_factory=list)    # advisory
    excluded_source_groups: list[SourceGroup] = Field(default_factory=list)     # hard constraint
    preferred_providers: list[str] = Field(default_factory=list)                # advisory
    excluded_providers: list[str] = Field(default_factory=list)                 # hard constraint
    comparison_targets: list[str] = Field(default_factory=list)
    time_window_days: int | None = None
    planning_mode: PlanningMode = PlanningMode.broad_scan
    cost_bias: str | None = None               # "minimize" | "balanced" | "no_limit"
    speed_bias: str | None = None              # "fast" | "balanced" | "thorough"
```

**Key rules:**
- `excluded_*` = hard constraints enforced at the provider registry level
- `preferred_*` = advisory context the supervisor LLM reasons about
- All fields have sensible defaults; empty preferences = current behavior (backward compatible)

---

## Section 2: Classifier Expansion

The existing classifier checkpoint is expanded to extract `ResearchPreferences` alongside classification. No new checkpoint, no new LLM call.

### Model Change

`RequestClassification` gains a `preferences` field:

```python
class RequestClassification(StrictBaseModel):
    audience_mode: str
    freshness_mode: str
    recommended_tier: Tier
    needs_clarification: bool
    clarification_question: str | None = None
    preferences: ResearchPreferences = Field(default_factory=ResearchPreferences)
```

### Prompt Change (`classifier.md`)

The 2-line stub is expanded to instruct the LLM to:

- Parse explicit user directives ("focus on recent papers", "compare X vs Y", "just give me the answer")
- Infer implicit preferences (brief about a GitHub project → `repos` preference; "what do people think" → `social`)
- When the user doesn't express a preference, leave the field at its default -- don't guess
- Map `freshness_mode` into `preferences.freshness` and `preferences.time_window_days`
- Detect comparison targets ("X vs Y", "compare A and B")
- Infer `planning_mode` from query structure

### Redundancy Note

`audience_mode` and `freshness_mode` on `RequestClassification` become redundant with `preferences.audience` and `preferences.freshness`. They are kept for backward compatibility. The rest of the pipeline reads from `preferences`. They can be deprecated later.

---

## Section 3: Threading Preferences Through Plan and Supervisor

### Planner (`plan.py`)

**Before:** `del classification` -- classification discarded, planner sees only raw brief.

**After:** Planner receives `brief + classification + preferences` and sees a structured prompt containing:

- The raw brief (user's words, preserved)
- The classification (audience, freshness, tier reasoning)
- The preferences (planning mode, source preferences, comparison targets, deliverable mode)
- Mode-specific instructions: "If planning_mode is `comparison`, generate balanced subtopics for each comparison target. If `timeline`, organize subtopics chronologically."

### Supervisor (`supervisor.py`)

The supervisor prompt payload gains three new fields:

```python
prompt = {
    # ... existing fields (plan, ledger, uncovered_subtopics, iteration, tier, etc.) ...
    "user_brief": brief,
    "preferences": prefs.model_dump(mode="json"),
    "guidance": build_supervisor_guidance(prefs),
}
```

`build_supervisor_guidance(prefs)` is a pure function that turns structured preferences into natural language. Example:

> "The user prefers web and social sources over academic papers. Freshness is important -- prioritize results from the last 7 days. This is a comparison between React and Svelte -- ensure balanced evidence for both."

This is advisory. The supervisor reasons about it alongside the plan and ledger. It can deviate if it has good reason, but the user's voice is present in every iteration.

### Supervisor Prompt Update (`supervisor.md`)

The 11-line prompt gains a section on interpreting preferences:

- Preferred sources: weight search actions toward these, but don't ignore other sources if clearly relevant
- Planning mode context: if comparison, ensure both targets get searched; if timeline, search with date-range awareness
- Freshness: bias toward recency-constrained queries when the user cares about freshness

### Data Flow

```
brief
  → classifier → RequestClassification (with preferences)
    → planner sees: brief + classification + preferences
      → supervisor sees: plan + ledger + brief + preferences + guidance
```

User intent survives from input to every search iteration.

---

## Section 4: Runtime Enforcement (Search Layer)

### Hard Exclusions at Registry Level

`ProviderRegistry.providers_for()` gains exclusion filtering:

```python
def providers_for(self, action, excluded_providers, excluded_source_groups):
    candidates = self._match(action)
    candidates = [p for p in candidates if p.name not in excluded_providers]
    candidates = [p for p in candidates if p.source_group not in excluded_source_groups]
    return candidates
```

This is the only hard enforcement point. Deterministic, no LLM discretion.

### Provider Source Group Declaration

Each provider declares its `SourceGroup`:

- `ArxivSearchProvider.source_group = SourceGroup.papers`
- `SemanticScholarProvider.source_group = SourceGroup.papers`
- `BraveSearchProvider.source_group = SourceGroup.web`
- `ExaSearchProvider.source_group = SourceGroup.web`

### Search Checkpoint Change

`execute_searches` passes exclusions from preferences to the registry:

```python
def execute_searches(actions, ledger, config, preferences):
    registry = ProviderRegistry(config.enabled_providers)
    for action in actions:
        providers = registry.providers_for(
            action,
            excluded_providers=preferences.excluded_providers,
            excluded_source_groups=preferences.excluded_source_groups,
        )
        if not providers:
            ledger.record_skip(action, reason="all providers excluded")
            continue
        # execute as normal
```

### Soft Preferences Are LLM-Driven

Everything that isn't an explicit exclusion is advisory:

- Which providers the supervisor chooses to target (via SearchAction)
- How many queries go to preferred vs non-preferred sources
- How freshness bias manifests (date filters, query phrasing, ranking)

All of this lives in the supervisor prompt context, not in code.

### Fallback Metadata

When exclusions cause zero-provider situations, the ledger records it. The supervisor sees it in the next iteration and can adjust. Degradations are collected in the final package.

---

## Section 5: Render Mode Selection

### Deliverable Mode Routing

`preferences.deliverable_mode` determines which renders get materialized:

| Mode | Renders produced |
|---|---|
| `research_package` | reading_path + backing_report (current default) |
| `final_report` | full_report only |
| `comparison_memo` | full_report with comparison-tuned context |
| `recommendation_brief` | full_report with recommendation-tuned context |
| `answer_only` | full_report with concise-answer context |

### Free-Form LLM-Driven Output Structure

The report content is **not validated against a section-level Pydantic schema**. The writer system prompt provides a default structure as guidance (e.g., "typically: executive summary, key findings, detailed analysis, sources"), but the LLM adapts based on deliverable mode and user preferences.

`RenderProse` simplifies to:

```python
class RenderProse(StrictBaseModel):
    content: str              # free-form markdown, LLM-structured
    render_label: str         # what the LLM called this output
    deliverable_mode: DeliverableMode
```

### Writer Prompt Selection

`materialization.py` gains a `prompt_for_mode(mode)` function. For v1:

- `research_package` → existing `writer_reading_path.md` / `writer_backing_report.md`
- `final_report` → existing `writer_full_report.md`
- Other modes → `writer_full_report.md` with a preamble injected from preferences

No new prompt files needed for v1. Context injection (deliverable mode + preferences + comparison targets) is sufficient for the writer LLM to adapt its output structure.

### Writer Prompts as Style Guides

The `writer_*.md` files become style guides with defaults, not schemas to enforce. The LLM decides the actual section structure, headings, and format based on the deliverable mode and user preferences.

---

## Section 6: Persistence and Package Output

### InvestigationPackage Changes

```python
class InvestigationPackage(StrictBaseModel):
    # ... existing fields ...
    preferences: ResearchPreferences | None = None
    preference_degradations: list[str] = Field(default_factory=list)
```

- `preferences`: extracted preferences as the classifier produced them. Immutable record.
- `preference_degradations`: plain-language list of where the system couldn't honor a preference.

### Degradation Sources

- **Search checkpoint:** entries when exclusions cause zero-provider situations
- **Render checkpoint:** entry when falling back from an unsupported deliverable mode
- **`assemble_package`:** collects degradations from flow metadata

### Examples

- "Preferred source group 'social' has no available providers -- skipped"
- "Provider 'arxiv' excluded by user -- subtopic 'foundational theory' has thinner coverage"
- "Deliverable mode 'comparison_memo' not yet fully supported -- fell back to 'final_report' with comparison context"

---

## Behavior Rules

1. User exclusions are hard constraints (registry-enforced).
2. User preferences are advisory context (LLM-reasoned).
3. The system can still stop on convergence/budget/time.
4. The durable loop stays fixed; planning, routing, and rendering become user-shaped.
5. When a preference can't be honored, degrade gracefully and record the degradation.
6. Empty preferences = current behavior. Full backward compatibility.

## Failure Handling

- Preferred provider unavailable → supervisor routes to next compatible provider; degradation logged
- Requested source group unsupported → continue if at least one requested group is satisfiable; degrade and log otherwise
- Unsupported deliverable mode → fall back to nearest supported mode with context injection; degradation logged

## Testing Plan

- **Model tests:** `ResearchPreferences` serialization, defaults, validation
- **Classifier tests:** preferences extraction from various brief styles (explicit directives, implicit preferences, no preferences)
- **Planner tests:** planner receives and uses classification + preferences (no more `del classification`)
- **Supervisor tests:** prompt payload includes brief, preferences, guidance
- **Search tests:** provider exclusions are enforced; preferences don't block providers
- **Render tests:** `deliverable_mode` selects correct render path; fallback works
- **Package tests:** preferences and degradations persist in output
- **Steerability integration tests:** two different briefs on the same topic produce materially different plans and provider routing

## Implementation Phases

**Phase 1: Models and classifier expansion**
- Add enums (`SourceGroup`, `DeliverableMode`, `PlanningMode`)
- Add `ResearchPreferences` model
- Expand `RequestClassification` with `preferences` field
- Expand classifier prompt
- Add `source_group` to `SearchProvider` protocol

**Phase 2: Plan and supervisor threading**
- Update `build_plan` to use classification + preferences (remove `del classification`)
- Update planner prompt
- Add `build_supervisor_guidance()` pure function
- Update supervisor prompt payload with brief, preferences, guidance
- Update supervisor prompt file

**Phase 3: Search enforcement**
- Update `ProviderRegistry.providers_for()` with exclusion filtering
- Update `execute_searches` to pass exclusions
- Add ledger skip recording
- Add degradation metadata collection

**Phase 4: Render mode selection**
- Add deliverable mode routing in render checkpoint
- Add `prompt_for_mode()` in materialization
- Simplify `RenderProse` to free-form content
- Update writer prompts to be style guides with context injection

**Phase 5: Package and validation**
- Add `preferences` and `preference_degradations` to `InvestigationPackage`
- Wire degradation collection in `assemble_package`
- Add all tests per testing plan
- Regression tests: no preferences = identical to current behavior
