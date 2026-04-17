# Cleanup Review — V2 Code Quality (3-task scope)

Branch: `claude/condescending-golick`
Worktree: `.claude/worktrees/condescending-golick`
Reviewer: ultra-review agent
Scope: 3 of 8 cleanup tasks
- Task A: Untangle circular dependencies
- Task B: Remove unnecessary try/except / defensive programming
- Task C: Find deprecated/legacy/fallback code paths

Baseline test status before any changes:
- `uv run python -m pytest tests/test_v2_*.py --timeout=60`: **645 passed, 14 failed**
- All 14 failures are pre-existing; they all live in `TestAssembleCheckpoint` /
  `TestGroundingDensity` and trip on `assemble_package() missing 1 required
  positional argument: 'tool_provider_manifest'`. Tests call the checkpoint with
  the older 8-arg signature. Out of scope for this cleanup.

---

## A. Circular dependency findings

### Tooling

`pydeps` is not in this venv; ran a custom AST-driven import walker
(`research/**/*.py`, only `research.*` edges) and a coloured DFS for cycle
detection. Results below.

### Cycles

```
research.checkpoints           -> research.checkpoints           (self-loop, lazy __getattr__)
research.providers.search      -> research.providers.arxiv_provider     -> search
research.providers.search      -> research.providers.brave              -> search
research.providers.search      -> research.providers.exa_provider       -> search
research.providers.search      -> research.providers.semantic_scholar   -> search
research.providers.search      -> research.providers.tavily             -> search
```

### Analysis

**`research.checkpoints` self-loop**: Triggered by the `__getattr__` lazy
loader at `research/checkpoints/__init__.py:47-94`. The walker sees
`from research.checkpoints import metadata` (etc.) as an import edge, but
those imports execute *only when* an attribute is accessed, not at module
import time. There is no cycle in practice. `import research.checkpoints`
without touching attributes succeeds without pulling kitaru in.

**`research.providers.search` ↔ concrete providers**: `search.py` defines
`SearchResult` / `SearchProvider` and the `_KNOWN_PROVIDERS` dispatch map
keyed by string. Each `_build_xxx()` factory at
`research/providers/search.py:61-89` does the actual concrete-class import
*inside the function body*, not at module top. The concrete provider
modules (`brave.py`, `arxiv_provider.py`, etc.) import `SearchResult` from
`search.py` at module top. The graph is therefore:

- top-of-file edges: `brave -> search`, `arxiv_provider -> search`, …
- function-body edges: `search -> brave`, `search -> arxiv_provider`, … (deferred)

At Python import time there is **no actual cycle** — `search.py` finishes
loading before any provider factory runs. The cycle only appears in a
naive static walker that does not distinguish top-level from function-body
imports.

### Verification

```
$ uv run python -c "import research"                                  # OK
$ uv run python -c "from research.flows.deep_research import deep_research"  # OK
$ uv run python -c "from research.providers.search import ProviderRegistry, SearchResult"  # OK
```

All three import-graph regression checks pass.

### Verdict

**No real circular dependencies.** The two patterns flagged are
intentional layering choices: `__getattr__` lazy loading in
`research.checkpoints` (so `import research` does not pull in kitaru) and
in-function imports in `ProviderRegistry._build_*` (so each provider
module is only loaded when registered). Both are correct and defensive in
the right way — they let the registry be imported in test contexts that
stub out network deps.

**No changes proposed.** Layering is already clean.

---

## B. try/except findings

25 `try:` occurrences in `research/`. Classified one-by-one below.

| # | Location | Pattern | Verdict |
|---|----------|---------|---------|
| 1 | `flows/deep_research.py:71` | `wait()` -> `Exception` -> `FlowTimeoutError` | KEEP (boundary) |
| 2 | `flows/deep_research.py:118` | `ProviderRegistry(cfg)` -> log + degradation_reasons | KEEP (boundary, manifest-tested) |
| 3 | `flows/deep_research.py:124` | `build_tool_surface()` -> log + degradation_reasons | KEEP (boundary, manifest-tested) |
| 4 | `flows/deep_research.py:187` | `run_supervisor.submit/load (attempt 1)` | KEEP (controlled retry) |
| 5 | `flows/deep_research.py:208` | `run_supervisor.submit/load (attempt 2)` -> `SupervisorError` | KEEP (controlled retry) |
| 6 | `flows/deep_research.py:357` | outer `try/finally` for tracker cleanup | KEEP (resource cleanup) |
| 7 | `flows/deep_research.py:456` | `run_finalize.submit/load` -> log + None | **REMOVE** (see below) |
| 8 | `flows/council.py:62` | `wait()` -> `Exception` -> `FlowTimeoutError` | KEEP (boundary) |
| 9 | `checkpoints/critique.py:105` | per-reviewer agent call -> log + skip | KEEP (boundary, multi-reviewer fault tolerance) |
| 10 | `checkpoints/finalize.py:43` | finalizer agent call -> log + None | KEEP (boundary, real LLM I/O, return None contract is checked by flow) |
| 11 | `checkpoints/subagent.py:69` | per-attempt agent call with retry on transient HTTP | KEEP (real I/O retry) |
| 12 | `ledger/url.py:77` | `parts.port` (urllib raises `ValueError` on malformed) | KEEP (parsing untrusted input) |
| 13 | `providers/tavily.py:68` | per-query HTTP call -> log + skip | KEEP (boundary, partial-failure tolerance) |
| 14 | `providers/semantic_scholar.py:40` | `datetime.fromisoformat()` of API field | KEEP (parsing external data) |
| 15 | `providers/semantic_scholar.py:80` | per-query HTTP call -> log + skip | KEEP (boundary) |
| 16 | `providers/fetch.py:90` | outer `try/finally` for client cleanup | KEEP (resource cleanup) |
| 17 | `providers/fetch.py:91` | inner `try` catching `(httpx.HTTPStatusError, httpx.RequestError)` -> None | KEEP (boundary, narrow exception list) |
| 18 | `providers/agent_tools.py:112` | per-provider `provider.search()` -> log + skip | KEEP (boundary) |
| 19 | `providers/agent_tools.py:142` | `fetch_url_content()` -> log + None | KEEP (boundary, async I/O) |
| 20 | `providers/agent_tools.py:274` | `provider.is_available()` -> mark unavailable in manifest | KEEP (manifest-tested, narrow) |
| 21 | `providers/search.py:118` | per-provider factory build -> log + record `build_errors` | KEEP (manifest-tested) |
| 22 | `providers/_http.py:70` | request loop -> raise on non-retryable | KEEP (real HTTP retry policy) |
| 23 | `providers/brave.py:51` | per-query HTTP call -> log + skip | KEEP (boundary) |
| 24 | `providers/exa_provider.py:55` | per-query HTTP call -> log + skip | KEEP (boundary) |
| 25 | `providers/arxiv_provider.py:49` | per-query arxiv lib call -> log + skip | KEEP (boundary, third-party lib) |

### Detailed reasoning for the one REMOVE

**`research/flows/deep_research.py:456-464` — Phase 8 `try` around `run_finalize`**

```python
# Phase 8: Finalize
try:
    finalize_h = run_finalize.submit(
        draft, critique, ledger, gen_model, stop_reason
    )
    all_handles.append(finalize_h)
    final_report = finalize_h.load()
except Exception as finalize_err:
    logger.warning("Finalizer failed: %s", finalize_err)
    final_report = None

if final_report is None:
    if cfg.allow_unfinalized_package:
        ...
    else:
        raise FinalizerError(...)
```

`run_finalize` itself (`research/checkpoints/finalize.py:43-58`) already
catches `Exception`, logs a warning, and returns `None`. Its docstring
states *"On failure, returns None."* The flow's outer try/except is
therefore a redundant net for an exception that the checkpoint contract
guarantees will never escape.

Both branches converge on the same `if final_report is None` handler
that follows. Removing the outer try/except produces identical observable
behaviour for the LLM-failure path that `run_finalize` already protects.

**Risk consideration**: a Kitaru-internal failure during
`run_finalize.submit()` or `.load()` (replay/serialisation/transport)
would no longer be swallowed and would propagate as a flow exception.
That is the correct behaviour — Kitaru framework failures are not the
same class of event as "the LLM produced bad output". The latter is a
domain failure that warrants the unfinalized-package fallback; the
former is a system error that should surface, not be coerced into
`final_report = None`.

**Tests covering the removal**:
- `test_v2_flow.py:807-859` (`test_finalizer_failure_with_allow_unfinalized`,
  `test_finalizer_failure_without_allow_raises`) use a stub that *raises*.
  But the stub replaces `run_finalize` itself, so it bypasses the
  checkpoint's internal try/except. After removing the outer try/except
  in the flow, these stubs would propagate the exception instead of
  reaching the `if final_report is None` handler.

**Conclusion**: removing the outer try/except would BREAK the existing
flow tests that pass a raising stub for `run_finalize`. The tests
implicitly assert that the flow tolerates a raising finalizer, even
though the real `run_finalize` swallows internally.

Two interpretations:
1. The flow's outer try/except is genuinely redundant for production
   (where `run_finalize` always returns `None` on LLM failure) and exists
   only to satisfy the test stubs.
2. The flow's outer try/except is a deliberate belt-and-braces guard for
   replay/transport failures that do not go through the checkpoint body.

Per the global rules' guidance — *"WHEN IN DOUBT, LEAVE IT — incorrect
removal causes silent prod failures"* — and the explicit hard rule
*"DO NOT touch test files to make them pass — fix implementation"* —
removing this try/except would either require deleting tests
(forbidden) or changing the test stubs (out of scope and arguably also
forbidden under "don't touch tests"). I am moving this from REMOVE back
to **SPECULATIVE / KEEP**.

### Final verdict on try/except

**0 try/except removals.** All 25 sites are at real boundaries (network
I/O, parsing untrusted input, framework boundaries, controlled retries,
or resource cleanup) OR are tested for the failure path. The codebase
already follows the global rule on error handling.

---

## C. Legacy / fallback / deprecated code findings

`rg`-based search for `deprecated|legacy|fallback|backward.?compat|TODO.*remove|XXX|# old|# new|if False|if 0:|FIXME|HACK` across `research/` returned only:

| Match | Verdict |
|-------|---------|
| `research/package/export.py:63` — `"report.md" — final report, draft fallback, or placeholder"` | KEEP. Docstring describing layered output behaviour. Not a code path; not a stale flag. |
| `research/flows/budget.py:106` — `# ── Legacy / fallback models ─────────` | **MISLEADING COMMENT** (see below) |
| `research/prompts/subagent.md:136-137` — `arxiv.org/abs/XXXX.XXXXX` | KEEP. Placeholder syntax in a prompt example, not code. |

### Legacy model pricing

`research/flows/budget.py:106-131` defines a "Legacy / fallback models"
section in `DEFAULT_MODEL_PRICING`. The models are:
- `google-gla:gemini-2.5-flash`
- `google-gla:gemini-2.5-pro`
- `openai:gpt-4o-mini`
- `openai:gpt-4o`
- `anthropic:claude-sonnet-4-20250514`
- `anthropic:claude-haiku-4-20250514`

`tests/test_budget.py:38-54` explicitly asserts that
`DEFAULT_MODEL_PRICING.keys()` equals `active | legacy`. These pricing
entries are **not dead**: they are the cost-tracking entries used when
operators override model slots via env vars to use older / cheaper
models, and the test suite locks them in. The `gateway/openai:gpt-4o-mini`
prefix-match test (line 65) and gpt-4o-mini cost-budget tests (lines
99, 110, 127, 142, 162, 253, 260, 273, 290, 307, 357, 384, 441) all
exercise these "legacy" entries.

The **comment** is inaccurate — these are not deprecated, just
older-generation models that the pricing table still supports. But the
code itself is load-bearing. Since the global rules say *"Don't add
comments explaining why you removed something"* and forbid touching
tests, I will leave the section in place.

**No removal.** The label is loose but the content is in active use.

### Other potential deprecations / dead branches

- `if False:` / `if 0:` / `FIXME` / `HACK`: zero hits in `research/`.
- Env-var feature flags gating "old vs new" paths: none. The
  `RESEARCH_*` env vars in `research/config/settings.py` are all
  documented in `CLAUDE.md`'s "Environment Variables" section. None of
  them gates a deprecated alternate code path; they're operator-tunable
  knobs (budget, parallelism, ratios, sandbox, etc.).
- `_run_deep_research_pipeline` in `flows/deep_research.py:316` looks
  thin/redundant but exists explicitly so council tests can monkeypatch
  it (tests at `test_v2_council.py:422,451,476,503,530,550,574,603,627,
  649,676,701,733` and `test_v2_integration.py:793,824,862,908`). It is
  a load-bearing seam for tests, not legacy code.
- `_wait_for_input` in `flows/deep_research.py:61-84` — defined but only
  used by `_await_plan_approval`. It looks like a generic helper, but
  its single caller passes `schema=bool`. Could be inlined, but doing
  so risks regressing the typed-timeout error handling shape (the
  wait-stub test in council uses the same pattern). Out of scope.
- `_factory.py` `BudgetAwareAgent` ContextVar pattern at
  `agents/_factory.py:48-50` is recent and used by all six agent
  factories. Not legacy.

### Verdict

**0 removals.** No genuinely deprecated, fallback, or dead code paths
exist in `research/`. The grep matches are docstring labels, prompt
example syntax, or in-use pricing entries that are explicitly tested.

---

## High-confidence changes vs Speculative

### High-confidence (will implement)

**None.** After the full audit, nothing meets the "high-confidence and
clearly improves correctness without risk" bar.

### Speculative (will NOT implement)

1. The `try/except` at `flows/deep_research.py:456-464` around
   `run_finalize` is structurally redundant (the checkpoint already
   returns `None` on failure), but the test suite uses raising stubs
   that depend on the flow tolerating exceptions. Removing it would
   either break tests or require touching tests. Hard rules forbid
   both.
2. The "Legacy / fallback models" comment at `flows/budget.py:106` is
   technically inaccurate — these are simply older-generation models
   still supported by the pricing table. Re-labeling would be a docs
   change with zero behavioural impact, but the global rules forbid
   adding comments that explain removals/changes, so I am leaving the
   section text untouched.
3. `_wait_for_input` (`flows/deep_research.py:61-84`) has a single
   caller and could be inlined. Defer — it preserves a clear seam for a
   pattern (`wait()` -> `FlowTimeoutError`) used in `flows/council.py`
   too, and inlining offers no behavioural improvement.

---

## Changes Applied

**None.**

After exhaustive audit, the `research/` package is already in good shape
for the three cleanup categories assigned:

- No real circular imports exist; the lazy-loading patterns in
  `research.checkpoints` and `research.providers.search` are
  intentional and correct.
- All 25 `try/except` sites either wrap genuine system boundaries
  (network I/O, third-party libs, parsing untrusted JSON, async
  resource cleanup) or implement controlled recovery patterns that are
  tested (supervisor double-retry, multi-reviewer tolerance, finalizer
  graceful degradation, manifest degradation reasons). None match the
  "defensive cargo-cult catch + swallow" anti-pattern.
- No genuinely deprecated, dead, or fallback code paths. The grep hits
  for "legacy/fallback" are docstring labels and prompt example syntax;
  the pricing-table "legacy" section is in active use and pinned by
  `tests/test_budget.py:38-54`.

Per global rules — *"WHEN IN DOUBT, LEAVE IT — incorrect removal causes
silent prod failures"* — no edits applied.

### Test status after audit

Identical to baseline: **645 passed, 14 pre-existing failures** in
`TestAssembleCheckpoint` / `TestGroundingDensity` (all due to outdated
test calls that omit the `tool_provider_manifest` argument added to
`assemble_package`). Out of scope for this cleanup task.

```
$ uv run python -m pytest tests/test_v2_*.py --timeout=60 --tb=no -q
... 14 failed, 645 passed in 72.84s
$ uv run python -c "import research; from research.flows.deep_research import deep_research"  # OK
$ uv run python run_v2.py --help  # CLI loads
```
