# Deslop Cleanup Assessment

Worktree: `/Users/safoineelkhabich/work/zenml-io/deep-research/.claude/worktrees/condescending-golick`
Scope: `research/` (V2). `deep_research/` (V1) untouched.
Tools used: `vulture --min-confidence 60`, `ruff check --select F401,F811,F841,ARG,B008`, plus `rg` cross-checks.

---

## Unused code findings

### TRULY UNUSED — safe to delete

| Symbol | File:line | Cross-check | Action |
|---|---|---|---|
| `import httpx` | `research/providers/brave.py:11` | `rg "httpx"` in file: only the import. HTTP comes via `_http.build_async_client`. | DELETE |
| `from research.config.slots import ModelSlot` | `research/config/settings.py:14` | `rg "ModelSlot[^C]" research/config/settings.py`: no other uses. | DELETE (keep `ModelSlotConfig`) |
| `from dataclasses import field` | `research/providers/agent_tools.py:17` | Only `dataclass` is used in this file. | DELETE |
| `from research.providers.code_exec import SandboxNotAvailableError` | `research/providers/agent_tools.py:29` | Mentioned only in a docstring (`code_exec` method); never raised or caught here. | DELETE |
| `relevance_threshold` parameter | `research/package/assembly.py:53` | Tests never pass it; only documented as "reserved for future use" — classic AI-slop. | DELETE param + docstring entry |

### USED — keep (vulture false positives)

| Symbol | File:line | Why it's actually used |
|---|---|---|
| `__getattr__` | `research/checkpoints/__init__.py:47` | Dunder — Python invokes it implicitly for lazy imports. |
| All Pydantic field defaults flagged in `contracts/` (audience, freshness_constraint, source_preferences, rationale, gaps, schema_version, generator_scores, council_provider_compromise, etc.) | various | Pydantic `BaseModel` fields accessed via attribute or by serialization; widely used in tests and `prompts/*.md`. |
| `model_config` (3 sites) | `config/settings.py`, `contracts/base.py` | Pydantic config dict — recognized by Pydantic, not directly referenced. |
| `input_cost_per_token`, `output_cost_per_token` | `config/slots.py:35-36` | Pydantic fields on `ModelSlotConfig`. |
| `check_iteration_budget` | `flows/budget.py:272` | Used in `tests/test_budget.py`. Public budget API. |
| `council_research` | `flows/council.py:97` | Heavily used: CLI (`run_v2.py`), README, large test suite (`test_v2_council.py`, `test_v2_integration.py`). Vulture FP because of `@flow` decorator. |
| `get_by_id` | `ledger/ledger.py:197` | Used in tests; trivial public lookup API. |
| `ToolCallResult` | `providers/agent_tools.py:37` | Used in `tests/test_v2_providers.py`. |
| `success`/`error` fields on `ToolCallResult` | same | Pydantic-style dataclass fields, accessed via attribute in tests. |
| `_config` attribute | `providers/agent_tools.py:66` | Set but unused inside class; arguable, but tests may instantiate and the symmetry with `_registry` reads as intentional. Conservative: keep. |
| `success` property on `CodeExecResult` | `providers/code_exec.py:37` | Tested and a documented public API on a stub interface. |
| `code`, `language`, `input_data` args on `SandboxExecutor.execute` | `providers/code_exec.py:78-81` | Documented stub interface; backends will use them. |
| `handle_starttag/endtag/data` | `providers/fetch.py:46/51/55` | `HTMLParser` overrides invoked by `parser.feed()`. |
| `raw_metadata` field | `providers/search.py:28` | Used by every provider adapter and tests. |
| `get_provider` | `providers/search.py:128` | Used in tests; trivial registry lookup. |
| `duration_seconds`, `iteration_added`, etc. on contracts | various | Pydantic fields. |

---

## AI slop / unnecessary comments

`rg` for `# TODO|FIXME|XXX|HACK|Now we|Here we|Previously|Removed|Updated|Added|Changed` returned **zero** matches in `research/`. The codebase is already disciplined about not leaving in-motion-work comments.

Spot-read of large files (`flows/deep_research.py`, `flows/budget.py`, `checkpoints/critique.py`, `checkpoints/assemble.py`, `checkpoints/subagent.py`, `agents/_factory.py`, `providers/agent_tools.py`, `providers/fetch.py`) found no AI-slop comments. Section-divider comments and design-rationale comments (e.g. `# Late imports — resolved against current sys.modules at call time`) are genuinely informative.

The only docstring-level slop is the `relevance_threshold:` "reserved for future use" note in `research/package/assembly.py:61-63`, which goes away when the parameter is deleted.

---

## Stub / larp findings

| Item | File:line | Verdict |
|---|---|---|
| `SandboxExecutor.execute` raises `SandboxNotAvailableError` for any backend | `providers/code_exec.py:100-105` | NOT slop — the module docstring explicitly calls itself an interface stub; tests assert the raise; backends are pluggable. KEEP. |
| `...` in `SearchProvider` Protocol | `providers/search.py:39, 53` | Protocol method bodies — required syntax. KEEP. |
| `pass` in `ledger/url.py:86` | inside an `if/else` branch | Legitimate control flow (the netloc default case). KEEP. |

No fake mock-data or "for testing" branches found in production paths.

---

## High-confidence changes (will implement)

1. Remove `import httpx` from `research/providers/brave.py:11`.
2. Remove `ModelSlot` from the `research/config/settings.py:14` import.
3. Remove `field` from the `research/providers/agent_tools.py:17` dataclass import.
4. Remove `SandboxNotAvailableError` from the `research/providers/agent_tools.py:26-30` import block.
5. Remove the unused `relevance_threshold` parameter (and its docstring entry) from `compute_evidence_stats` in `research/package/assembly.py`.

## Speculative (will NOT implement)

- `_config` attribute on `AgentToolSurface` — currently unread but the class stores it for symmetry with `_registry`; removing risks future regressions and reads as deliberate.
- `check_iteration_budget` is wrapper-around `budget.is_exceeded()` only consumed by tests; it's part of the public flow-budget API and deleting would require deleting tests.
- `get_by_id` on `ManagedLedger` — only test-consumed but trivial, public, documented.
- All Pydantic field "unused" findings — false positives by definition.

---

## Changes Applied

| File | Lines removed | What | Why safe |
|---|---|---|---|
| `research/providers/brave.py:11` | 1 import + 1 blank | `import httpx` | `rg "httpx" research/providers/brave.py` confirmed only the import; HTTP comes via `_http.build_async_client`. |
| `research/config/settings.py:14` | 1 import name | `ModelSlot` (kept `ModelSlotConfig`) | `ModelSlot` symbol not referenced anywhere in the file. ruff F401 confirmed. |
| `research/providers/agent_tools.py:17` | 1 import name | `field` from `dataclasses` (kept `dataclass`) | `field` not used in this file. ruff F401 confirmed. |
| `research/providers/agent_tools.py:29` | 1 import name | `SandboxNotAvailableError` from `code_exec` | Only mentioned in a docstring inside the same file; never raised, caught, or otherwise referenced in this module. The docstring text remains accurate (the inner `_sandbox.execute` does raise it). ruff F401 confirmed. |
| `research/package/assembly.py:50-64` | 14 lines collapsed to 2 | `relevance_threshold` parameter and its "reserved for future use" docstring entry | Tests never pass it; docstring explicitly admits it's a no-op. Pure AI slop. |

**Validation:**
- `uvx ruff check research/ --select F401,F811,F841` — All checks passed.
- `uvx vulture research/ --min-confidence 80` — clean.
- `uv run python -c "import research"` — succeeds.
- `uv run python run_v2.py --help` — CLI works unchanged.
- `uv run pytest tests/test_v2_providers.py tests/test_v2_config.py tests/test_v2_package.py` — 196 passed.
- `uv run pytest tests/test_v2_*.py` — 645 passed, 14 failed; **all 14 failures predate this work** (verified by `git stash` + re-run: same failures with no diff applied). They are caused by an `assemble_package() missing 1 required positional argument: 'tool_provider_manifest'` mismatch between `tests/test_v2_checkpoints.py` / `tests/test_v2_invariants.py` and the current signature in `research/checkpoints/assemble.py`.

**Net:** 4 files modified, 17 lines net removed, 4 unused imports + 1 dead parameter eliminated.
