# Implement — user memory pool

Ordered checklist; verify before each next step.

## Phase 1 — `MemoryStore` component

- [ ] 1.1 Create `research_assistant/retrieval/memory.py` with the schema + methods from `design.md`.
- [ ] 1.2 Add `MemoryStore` + `format_preferences` to `retrieval/__init__.py` `__all__`.
- [ ] 1.3 `py_compile` the new file.

## Phase 2 — tests

- [ ] 2.1 `tests/test_memory.py`: add/get preference round-trip (3 categories); add_qa → recall returns stored Q&A; recall with non-matching query returns empty.
- [ ] 2.2 Run `pytest tests/test_memory.py`.

## Phase 3 — facade integration

- [ ] 3.1 `assistant.py`: add `user_id` param (default `"default"`); create `self.memory = MemoryStore(str(_PROJECT_ROOT / 'data' / 'memory.db'))`.
- [ ] 3.2 `search_papers`: read keyword prefs and merge into the query (before/after query enhancement).
- [ ] 3.3 Add a `format_preferences`-based preamble into one generation path (e.g. `interpret_paper`) as the soft-personalization hook.

## Phase 4 — verification

- [ ] 4.1 `py_compile` all touched files.
- [ ] 4.2 `pytest tests/` — new + existing 26 tests pass.
- [ ] 4.3 `import research_assistant` + a quick `add_preference`/`get_preferences` smoke run against `:memory:`-style temp DB.

## Rollback

- New component only — no existing behavior changes except `search_papers`
  keyword merging; if it misbehaves, remove the merge call and memory still works.
