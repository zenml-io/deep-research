# Kitaru Usage Audit — deep-research V2

Scope: `research/flows/{deep_research,council}.py`, all `research/checkpoints/*.py`, `research/agents/_factory.py`. Read-only assessment. Compared against `~/work/zenml-io/kitaru/kitaru/src/kitaru/`.

Overall: the codebase is a well-disciplined Kitaru user — checkpoint types are correct, non-determinism is quarantined, checkpoint IDs for the replay-critical supervisor path are stable, and `wait()` is correctly placed in flow scope. The biggest risks are replay correctness around mutable budget state, duplicate checkpoint IDs in the supplemental loop, and a `list[Callable]` passed as a checkpoint argument.

---

## CORRECTNESS BUGS

### 1. Supplemental loop creates duplicate-ID draft/critique checkpoints
`research/flows/deep_research.py:443-451` — inside the `while critique.require_more_research` supplemental loop, `run_draft.submit(...)` and `run_critique.submit(...)` are called a second time **in the same flow execution** with no `id=` override. The initial calls at lines 410 and 417 have no `id=` either. Kitaru auto-derives checkpoint invocation IDs from call order plus argument fingerprint; calling the same checkpoint twice in one execution without an explicit ID risks (a) cache-key collision that serves the first draft on the second call, or (b) ambiguous replay selectors for `kitaru executions replay --from run_draft`. Compare to `_run_supervisor_with_retry` which correctly uses `id=f"supervisor_{iteration_index}_a1"` / `_a2`. The fix: pass `id=f"draft_supplemental_{supplemental_loops}"` and `id=f"critique_supplemental_{supplemental_loops}"`, and also make the initial calls explicit (`id="draft_primary"`, `id="critique_primary"`).

### 2. Mutable `BudgetConfig.spent_usd` crosses checkpoint boundaries
`research/flows/budget.py:255` — `self.budget.spent_usd += cost_usd` mutates a shared `BudgetConfig` in place from inside `BudgetTracker.record_usage()`, which is called from `BudgetAwareAgent.run_sync()` inside every LLM checkpoint. On **replay**, completed checkpoints return cached results without re-invoking the wrapped agent, so `spent_usd` is **not** re-accumulated. The flow body then reads `cfg.budget.soft_budget_usd - cfg.budget.spent_usd` at `deep_research.py:264` and passes that delta into `run_supervisor` and into `check_convergence` — the flow will see `spent_usd == 0` after a replay, diverging from the original run's convergence decisions. This is the textbook "mutable global state crossing checkpoint boundary" antipattern from the Kitaru guardrails. Fix: persist cumulative spend inside the `WallClockSnapshot` or a new `@checkpoint(type="tool_call")` that returns the running total; or stamp per-iteration spend into each `IterationRecord` and reconstruct the total from checkpoint outputs rather than a live mutable field. Also apply to `ContextVar` token plumbing — the ContextVar itself is fine (run-scoped) but the underlying tracker's mutation is the problem.

### 3. `list[Callable]` passed as a checkpoint argument (`run_subagent(tools=...)`)
`research/checkpoints/subagent.py:40-42` — `tools: list[Callable[..., Any]] | None` is a positional checkpoint argument. Kitaru fingerprints arguments to compute the cache key / invocation id; closures/functions do not have a stable identity across processes, so on **replay in a fresh runner** the argument fingerprint will differ from the original run and replay-from-later-checkpoint may cache-miss and re-run all subagents. The tools are also not serializable if Kitaru ever tries to persist the argument. Fix: build the tool surface **inside** the checkpoint body (lazy-import a stable provider config), or accept a serializable `ToolSurfaceSpec` (list of provider names) and reconstruct the callables inside the checkpoint. The `ToolProviderManifest` already exists for this purpose — pass *it* and reconstruct tools inside.

---

## ANTI-PATTERNS

### 4. `run_finalize` returns `None` on internal failure instead of raising
`research/checkpoints/finalize.py:43-58` swallows the exception and returns `None`. The outer flow (`deep_research.py:456-476`) also wraps the call in its own `try/except`, so there are two levels of error hiding. A checkpoint that catches its own `Exception` and returns a sentinel defeats Kitaru's retry/replay semantics — you cannot `kitaru executions retry` a "failed" finalize because the checkpoint completed successfully (with `None`). Either (a) remove the inner try/except and let Kitaru record the failure with `@checkpoint(retries=...)`, or (b) keep only the outer handler. Prefer (a).

### 5. `_build_tools_and_manifest` runs in flow scope outside any checkpoint
`research/flows/deep_research.py:380` — builds the provider registry and tool surface in the flow body. Any provider construction side effects (env-var reads, HTTP client initialization) run on **every replay from scratch**, even when replaying from a late checkpoint. This is not strictly a correctness bug because the output feeds into `run_subagent(tools=...)` which is itself a cache-hit, but it's wasted setup and an observability gap. Wrap in a `@checkpoint(type="tool_call")` returning the `ToolProviderManifest` plus a spec; reconstruct `tools` lazily inside `run_subagent`.

### 6. No `kitaru.log` / `kitaru.save` / `kitaru.load` anywhere
Grep for `kitaru.log` / `kitaru.save` / `kitaru.load` / `kitaru.llm` returned zero hits across `research/`. The iteration loop is the ideal place to `log(iteration_index=..., ledger_size=..., stop_reason=...)` so operators can scan structured breadcrumbs in `kitaru executions logs`. The draft/critique/final reports are ideal `save(name="draft_markdown", value=..., type="response")` candidates — right now they are only recoverable via `kitaru_artifacts_list` on the auto-captured checkpoint response, which is implicit and not guaranteed to have a memorable name. Missed durability + operator ergonomics.

### 7. `snapshot_wall_clock` is not pure across replay (by design) but is re-submitted every iteration without a stable `id=`
`research/flows/deep_research.py:248` — `clock_h = snapshot_wall_clock.submit(started_at)` inside `_run_iteration`. Since `started_at` is the same across iterations, **every iteration's checkpoint gets the same argument fingerprint**. Kitaru's auto-ID derivation usually disambiguates by call-order counter, but this is implicit. Add `id=f"wall_clock_{iteration_index}"` to be explicit.

### 8. `export_package` accepts a non-serializable `InvestigationPackage` as a checkpoint argument
`research/checkpoints/export.py:12-18` — pydantic models serialize fine via pydantic-core, so this is OK in practice, but the checkpoint also takes `output_dir: str` which combined with `assemble_h.load()` being awaited **before** `export_package.submit(..., after=all_handles)` means the flow serializes the entire package twice (once for assemble return, once for export arg). Consider passing the assemble handle (`after=[assemble_h]`) and having export re-load via `kitaru.load()` inside the checkpoint body.

---

## MISSED OPPORTUNITIES

### 9. `CapturePolicy(tool_capture="full")` is redundant
`research/agents/_factory.py:109` — `"full"` is the Kitaru default (see `~/work/zenml-io/kitaru/kitaru/src/kitaru/adapters/pydantic_ai/_policy.py:21`). `KitaruAgent(agent, name=name)` with no explicit `capture=` produces identical behavior. Per-tool overrides would be interesting — e.g. `tool_capture_overrides={"code_exec": "metadata"}` for the sandbox tool to skip potentially-huge stdout from ending up in Kitaru artifacts. Low-priority polish.

### 10. Council's `run_judge` is not invoked via `.submit().load()`
`research/flows/council.py:149` — `comparison = run_judge(packages, ...)` is a direct call, while every other checkpoint in the codebase uses `.submit().load()`. Direct calls work (the decorator handles both), but the inconsistency is a smell and prevents `after=` ordering / DAG edge tracking on the judge node. Make it `run_judge.submit(...).load()` for consistency.

### 11. `wait()` usage is correct but lacks structured metadata for the council selection enum
`research/flows/council.py:63` uses `schema=str` with a free-form string. Prefer `schema=Literal["generator_a", "generator_b"]` or a dedicated pydantic enum contract — makes the wait resolution typed end-to-end and surfaces better in `kitaru executions pending_waits`.

### 12. `_await_plan_approval` converts `wait()` timeout exceptions to `FlowTimeoutError` but Kitaru already has a typed `WaitTimeoutError`
`research/flows/deep_research.py:79-83` catches bare `Exception` and remaps. Narrow the catch to Kitaru's own wait exceptions so genuine bugs (`ValidationError` on `schema=bool`) are not swallowed.

---

## Summary Checklist

| Issue | Severity | Location |
|-------|----------|----------|
| Supplemental draft/critique duplicate IDs | Correctness | deep_research.py:443,447 |
| Mutable `BudgetConfig.spent_usd` across replay | Correctness | budget.py:255, deep_research.py:264 |
| `list[Callable]` as checkpoint arg | Correctness | subagent.py:40 |
| Finalizer swallows + returns None | Anti-pattern | finalize.py:43 |
| Tools built outside checkpoint | Anti-pattern | deep_research.py:380 |
| No `kitaru.log/save/load` breadcrumbs | Anti-pattern | flow-wide |
| `snapshot_wall_clock` no explicit `id=` | Anti-pattern | deep_research.py:248 |
| Redundant `CapturePolicy(tool_capture="full")` | Polish | _factory.py:109 |
| `run_judge` called directly, not `.submit()` | Polish | council.py:149 |
| Council wait uses free-form `schema=str` | Polish | council.py:63 |

Metadata isolation in `checkpoints/metadata.py` is **correct** and is the only place `uuid4` / `datetime.now()` appear in the orchestration path — good discipline.
