# Add user memory pool (short/long-term personalization)

## Goal

Add a per-user memory system so the assistant personalizes answers to its
operator. **Short-term** = current-session context; **long-term** = the user's
research preferences (journal / keyword / method) plus historical Q&A.

## Background (user intent)

- This machine deploys the assistant for one operator ("current user"); the
  design is per-`user_id` so it stays single- or multi-user ready.
- Long-term memory stores: **research preferences** (journal / keyword /
  method), **historical Q&A**, and (deferred) successful tasks.
- Purpose: personalize later retrieval and generation.

## Scope (MVP)

- `research_assistant/retrieval/memory.py` — a `MemoryStore` (SQLite, per `user_id`).
- **Explicit** preferences only (no auto-mining) — `add_preference` / `get_preferences`.
- Q&A history — `add_qa` / `recall`.
- Personalization at **both** retrieval and generation:
  - retrieval (hard): keyword preferences expand/refine the search query.
  - generation (soft): preferences injected as a prompt preamble.
- `tests/test_memory.py` (spec-required add/recall round-trip).

## Requirements

- **R1** `MemoryStore` persists preferences and Q&A per `user_id`, following `database-guidelines.md` (parameterized queries, per-operation connection, idempotent DDL).
- **R2** `add_preference(user_id, category, value, weight=1.0)` and `get_preferences(user_id)` return `{journal: [], keyword: [], method: []}`.
- **R3** `add_qa(user_id, query, answer, success=True)` and `recall(user_id, query=None, top_k=5)` return preferences + matching Q&A history (simple keyword match, GPU-free).
- **R4** The facade exposes `self.memory` and a `user_id` (default) so callers read/write memory.
- **R5** `search_papers` applies the user's keyword preferences to the query (observable, testable).
- **R6** A `format_preferences(preferences)` helper renders preferences as a text block for generation prompts.

## Acceptance Criteria

- [ ] `MemoryStore.add_preference` → `get_preferences` round-trips (all three categories).
- [ ] `add_qa` → `recall(query)` returns the relevant stored Q&A.
- [ ] `ResearchAssistant.search_papers(...)` with keyword prefs set returns a query influenced by those prefs.
- [ ] `tests/test_memory.py` passes; the existing 26 tests still pass.
- [ ] No network/GPU required by memory operations.

## Out of scope (deferred)

- Auto-mining/inference of preferences from history.
- `task_history` table (successful-task records).
- Multi-user auth/UI; session persistence of short-term memory (in-memory only for now).
- Semantic (BGE) recall of Q&A — keyword match is sufficient for MVP.
