# Consolidated PR Review — 8-Agent Deep Audit

Eight parallel agents reviewed the 3-agent cleanup pass plus the broader V2 codebase. This consolidation prioritizes findings by real-world impact.

**Agents:** code-reviewer, comment-analyzer, silent-failure-hunter, type-design-analyzer, pr-test-analyzer, code-simplifier, kitaru-validator, pydantic-ai-validator.

---

## Tier A — CORRECTNESS BUGS (replay-breaking / real)

These are not style — they change program behavior.

### A1. Mutable `BudgetConfig.spent_usd` crosses checkpoint boundaries
**Source:** kitaru-validator
**Location:** `research/flows/budget.py:255`, `research/flows/deep_research.py:264`
**Problem:** `BudgetTracker.record_usage()` mutates `self.budget.spent_usd += cost_usd` from inside `BudgetAwareAgent.run_sync()`. On replay, cached LLM checkpoints don't re-invoke the wrapped agent, so `spent_usd` stays 0. The flow body then reads `cfg.budget.soft_budget_usd - cfg.budget.spent_usd` to compute the supervisor's remaining-budget signal and to decide convergence — **replay will diverge from the original run's convergence decisions.** Classic "mutable state crossing @checkpoint boundary" bug.
**Fix:** Persist per-iteration spend into each `IterationRecord` (already returned from checkpoints) and reconstruct cumulative spend from those records in the flow body. Remove in-place mutation.

### A2. Supplemental-loop duplicate-ID draft/critique checkpoints
**Source:** kitaru-validator
**Location:** `research/flows/deep_research.py:410, 417, 443, 447`
**Problem:** `run_draft.submit()` and `run_critique.submit()` are called twice in the same flow execution (Phase 5/6, then again inside `while critique.require_more_research`) with no explicit `id=`. Kitaru's auto-ID derivation may cache-hit the first call's result for the second call, or produce ambiguous replay selectors. The supervisor path already does this right (`id=f"supervisor_{i}_a1"`), so there's a precedent.
**Fix:** Pass `id="draft_primary"` / `id="critique_primary"` on first calls and `id=f"draft_supplemental_{n}"` / `id=f"critique_supplemental_{n}"` on loop calls.

### A3. `list[Callable]` as a checkpoint argument (`run_subagent(tools=...)`)
**Source:** kitaru-validator
**Location:** `research/checkpoints/subagent.py:40-42`
**Problem:** Closures have no stable identity across processes. When Kitaru fingerprints the argument for cache-key derivation, replay in a fresh runner sees different hashes and cache-misses every subagent call. Also blocks serialization if Kitaru ever persists checkpoint args.
**Fix:** Pass the already-existing `ToolProviderManifest` (a Pydantic model, serializable) as the checkpoint argument, and reconstruct tools inside the checkpoint body.

### A4. Outer `try/except` around `run_finalize` hides Kitaru framework errors
**Source:** silent-failure-hunter (contesting review-agent's KEEP call)
**Location:** `research/flows/deep_research.py:456-464`
**Problem:** The inner `run_finalize` checkpoint already returns `None` on LLM failure. The outer `try/except Exception` is redundant for the LLM-failure path but swallows Kitaru transport/replay/serialization errors that should surface. The review-agent kept it because test stubs at `tests/test_v2_flow.py:807-859` raise — but those stubs simulate the contract incorrectly; they should return `None` to match the real checkpoint.
**Fix:** Remove the outer `try/except` in the flow; update the two test stubs to `return None` instead of `raise`.

---

## Tier B — IMPORTANT (real gaps that affect correctness, observability, or evolvability)

### B1. 14 pre-existing test failures have an exact fix
**Source:** pr-test-analyzer
**Location:** `tests/test_v2_checkpoints.py` (9 sites) + `tests/test_v2_invariants.py` (5 sites)
**Problem:** `assemble_package()` added required `tool_provider_manifest: ToolProviderManifest` arg; tests still call the 8-arg signature. `ToolProviderManifest()` has all-default fields.
**Fix:** Add `tool_provider_manifest=ToolProviderManifest()` (plus the import) to all 14 call sites. Mechanical.

### B2. `build_tool_provider_manifest` (255 lines) has zero direct unit tests
**Source:** pr-test-analyzer
**Location:** `research/providers/agent_tools.py:258-354`
**Fix:** Add `TestBuildToolProviderManifest` with 4 cases: empty registry, provider-available-marks-active, provider-raises-on-is-available-recorded-as-reason, surface-none-reports-no-tools.

### B3. Contract types lack `Literal` constraints for documented enumerations
**Source:** type-design-analyzer
**Locations:**
- `research/contracts/package.py:87` — `RunMetadata.tier: str` → `Literal["quick","standard","deep","exhaustive"]`
- `research/contracts/reports.py:50` — `CritiqueDimensionScore.dimension: str` → `Literal["source_reliability","completeness","grounding"]` (docstring at line 69 already declares this)
- `research/contracts/package.py:102` — `RunMetadata.stop_reason: str | None` → `Literal` matching `StopDecision.reason` values
**Fix:** Tighten types. Pydantic enforces the invariant.

### B4. Contract types lack range constraints where docstrings declare them
**Source:** type-design-analyzer
**Locations:**
- `research/contracts/reports.py:54` — `CritiqueDimensionScore.score: float` unbounded; probably 0-1 or 0-10
- `research/contracts/package.py:105-110` — `grounding_density: float | None`; docstring says "0.0–1.0"
- `research/contracts/evidence.py:50` — `iteration_added: int` should be `ge=0`
**Fix:** Wrap in `Annotated[float, Field(ge=0.0, le=1.0)]` etc.

### B5. PydanticAI tool return types are bare `dict` / `list[dict]`
**Source:** pydantic-ai-validator
**Location:** `research/providers/agent_tools.py:165, 193`
**Problem:** Schema derivation gives the LLM an opaque object with no property names → model has to guess field names from examples. Measurable quality impact.
**Fix:** Define small `TypedDict` or Pydantic models: `SearchToolResult`, `CodeExecToolResult`, `FetchToolResult`.

### B6. `fetch` tool collapses errors into empty string
**Source:** pydantic-ai-validator, silent-failure-hunter
**Location:** `research/providers/agent_tools.py:180-183`
**Problem:** Network error, non-text content, PDFs, and valid empty documents all look identical to the LLM. Model cannot distinguish "nothing there" from "failed".
**Fix:** Return `{ok: bool, content: str, reason: str | None}` or raise `ModelRetry` so the model tries a different URL.

### B7. `_record_usage` drops Anthropic cached + OpenAI reasoning token breakdowns
**Source:** pydantic-ai-validator
**Location:** `research/agents/_factory.py:54-69`
**Problem:** `RunUsage.details` carries provider-specific subcategories (Anthropic prompt-cache hits, OpenAI reasoning tokens) that have distinct per-token prices. Currently all merged into the same `input_tokens` bucket → systematic over/under-charging.
**Fix:** Read `usage.details` and surface cached/reasoning token counts separately into `BudgetTracker`.

### B8. `_wait_for_input` `None` branch is untested
**Source:** pr-test-analyzer
**Location:** `research/flows/deep_research.py:82-83`
**Fix:** Add `test_plan_wait_returning_none_raises_flow_timeout` in `TestPlanApproval`.

### B9. `schema: type` is too narrow for Kitaru's `wait()` contract
**Source:** code-reviewer, type-design-analyzer
**Location:** `research/flows/deep_research.py:63`
**Problem:** Kitaru's upstream accepts `schema: Any = None`. Current annotation rejects the `None` default.
**Fix:** `schema: type[Any]` or `schema: Any`. Alternatively rename `_wait_for_input` → `_wait_for_bool` (type-design-analyzer's suggestion) to lock in the single-caller shape.

### B10. Redundant defensive code in `_record_usage`
**Source:** pydantic-ai-validator
**Location:** `research/agents/_factory.py:54-60`
**Problem:** `getattr(result, "usage", None)` + `callable(...)` fallback hides the actual PydanticAI API. `result.usage()` is always a method on successful runs.
**Fix:** `usage = result.usage()`. Let `AttributeError` surface in tests if API shape drifts.

### B11. Finalizer broad `except Exception` + `logger.warning`
**Source:** silent-failure-hunter
**Location:** `research/checkpoints/finalize.py:43-58`
**Problem:** Catches all exceptions at warning severity for the primary deliverable. Hides `AttributeError`, `NameError`, `ValidationError` as if they were LLM failures.
**Fix:** Narrow to `(ValidationError, ModelHTTPError, TimeoutError)`, log with `logger.exception` (traceback). Deferred if A4 is taken — then the entire try/except goes away.

---

## Tier C — POLISH / DESLOP (pure cleanup, zero behavioral risk)

### C1. Decorative banner comments (21+ occurrences)
**Source:** comment-analyzer
**Locations:** `research/providers/agent_tools.py`, `research/flows/budget.py`, `research/config/defaults.py`, `research/ledger/canonical.py`, `research/providers/search.py`, `research/providers/fetch.py` — `# ----- XXX -----` dividers, `# ── Active tier models ──`, etc.
**Fix:** Delete. Python section breaks are inferred from function names and blank lines.

### C2. Phase 1-8 comments in `deep_research.py`
**Source:** comment-analyzer
**Location:** `research/flows/deep_research.py:364, 371, 379, 382, 409, 414, 421, 455, 478`
**Problem:** Duplicate the module docstring's pipeline outline.
**Fix:** Delete or collapse to blank-line separators.

### C3. Stray return-type comment
**Source:** comment-analyzer
**Location:** `research/flows/deep_research.py:185`
**Problem:** `# Return type: tuple[SupervisorDecision, list[Any]]` documents what an annotation should.
**Fix:** Move to real return annotation on `_run_supervisor_with_retry` signature (line 174).

### C4. Inline meta-narrative in supplemental loop
**Source:** comment-analyzer
**Location:** `research/flows/deep_research.py:452-453`
**Fix:** Delete — code is self-evident.

### C5. Redundant `CapturePolicy(tool_capture="full")`
**Source:** kitaru-validator
**Location:** `research/agents/_factory.py:109`
**Problem:** `"full"` is the Kitaru default.
**Fix:** `KitaruAgent(agent, name=name)` — drop the explicit policy.

### C6. `_run_supervisor_with_retry` is 65 lines with two identical submit blocks
**Source:** code-simplifier
**Location:** `research/flows/deep_research.py:163-228`
**Fix:** Collapse to `for attempt in (1, 2): ...` loop with `id=f"supervisor_{i}_a{attempt}"`. Drops to ~25 lines.

### C7. Supplemental draft/critique duplicates Phase 5+6 body
**Source:** code-simplifier
**Location:** `research/flows/deep_research.py:423-453`
**Fix:** Extract `_run_draft_and_critique(...)` helper; call once in Phase 5+6 and once in the supplemental loop. Synergistic with A2 (explicit IDs).

### C8. Nested ternary in `code_exec` reason
**Source:** code-simplifier
**Location:** `research/providers/agent_tools.py:333-347`
**Fix:** Extract `_code_exec_reason(available_tools, config) -> str | None` with explicit `if` branches.

### C9. `_surface_tool_names` is over-defensive for single caller
**Source:** code-simplifier
**Location:** `research/providers/agent_tools.py:241-255`
**Fix:** Simplify to `return list(surface.available_tools()) if surface is not None else []`.

### C10. `run_judge` called directly instead of via `.submit().load()`
**Source:** kitaru-validator
**Location:** `research/flows/council.py:149`
**Fix:** Use `.submit().load()` for consistency with every other checkpoint + DAG edge tracking.

### C11. Council `wait()` uses `schema=str` for generator selection
**Source:** kitaru-validator
**Location:** `research/flows/council.py:63`
**Fix:** `schema=Literal["generator_a", "generator_b"]` or an enum.

### C12. `snapshot_wall_clock.submit()` lacks explicit `id=`
**Source:** kitaru-validator
**Location:** `research/flows/deep_research.py:248`
**Fix:** `id=f"wall_clock_{iteration_index}"`.

### C13. `if approved is not True:` after `-> bool` annotation is verbose
**Source:** code-simplifier
**Location:** `research/flows/deep_research.py:105`
**Fix:** `if not approved:`.

---

## Tier D — ARCHITECTURAL (larger changes, need sign-off)

### D1. Migrate subagent tools to `deps_type=AgentToolSurface` + `RunContext`
**Source:** pydantic-ai-validator
**Scope:** `research/agents/_factory.py`, `research/agents/subagent.py`, `research/providers/agent_tools.py`, `research/checkpoints/subagent.py`
**Benefit:** Idiomatic PydanticAI; stable tool identity; no per-run closure rebuild; Logfire gets structured deps spans.
**Risk:** Medium — refactors the agent/tool boundary. Requires updating tests.

### D2. Add `kitaru.log` / `kitaru.save` breadcrumbs
**Source:** kitaru-validator
**Scope:** Flow body + drafts/reports
**Benefit:** Operator ergonomics in `kitaru executions logs`; named artifact retrieval.
**Risk:** Low — purely additive.

### D3. Wrap `_build_tools_and_manifest` in a `@checkpoint(type="tool_call")`
**Source:** kitaru-validator
**Location:** `research/flows/deep_research.py:380`
**Benefit:** Provider construction runs once per replay instead of every time.
**Risk:** Low-medium.

### D4. Use PydanticAI's `ModelRetry` / `ModelHTTPError` in subagent retry
**Source:** pydantic-ai-validator
**Location:** `research/checkpoints/subagent.py`
**Benefit:** Idiomatic; removes duck-typed `getattr(exc, "status_code", ...)`.
**Risk:** Low.

---

## Strengths observed (what the cleanup + codebase got right)

- Non-determinism isolation in `research/checkpoints/metadata.py` is the **only** place `uuid4`/`datetime.now()` appear in orchestration — excellent discipline.
- `StrictBase(extra="forbid")` applied uniformly; `EvidenceStats` TypedDict is an exemplary boundary tightening.
- Agent/tool split (only subagent gets tools) correctly matches CLAUDE.md's "narrow agent surface" contract.
- Judge uses a provider distinct from generator+reviewer; bias-elimination contract enforced.
- The 3-agent cleanup's conservatism was correct on almost every call — only 1 of 25 try/except decisions was contested (A4).
- `compute_evidence_stats` removal (dead `relevance_threshold`) verified safe — no broken callers.
