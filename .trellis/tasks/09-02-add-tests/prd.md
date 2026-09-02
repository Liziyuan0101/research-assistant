# Add pytest smoke tests

## Goal

Add a first, network-free and GPU-free pytest test suite that locks down the
behavior of the components refactored in `09-02-refactor-package`, so future
changes have a safety net. Scope follows `.trellis/spec/backend/quality-guidelines.md`.

## Scope

Four test files under `tests/`:

| File | What it covers |
|------|----------------|
| `tests/test_metadata_store.py` | `MetadataStore` SQLite CRUD against `:memory:` — `add_paper`/`get_paper` round-trip, `add_chunk`/`get_chunk_by_id`, `search_papers_by_title`, `get_stats`. |
| `tests/test_hybrid_retriever.py` | BM25 indexing + `_bm25_search` over a few synthetic papers (no BGE/FAISS, no GPU). |
| `tests/test_paper_retriever.py` | Pure helpers only: `_reconstruct_abstract`, `_deduplicate_papers`, `_extract_keyphrases_for_search`. No network. |
| `tests/test_utils.py` | `load_config` (yaml/json + error cases), `save_json`/`load_json` round-trip, `llm.resolve_api_key`/`create_openai_client` (env monkeypatching). |

## Requirements

- **R1** Tests run with plain `pytest` from the repo root, no network, no GPU, no model download.
- **R2** Tests import from `research_assistant.*` (the installed package).
- **R3** `pytest` is declared as a dev dependency (add a `dev` extra in `pyproject.toml`).
- **R4** Dense-retrieval (BGE-M3) and any network/API path are skipped, not exercised.

## Acceptance Criteria

- [ ] `pytest` passes (0 failures) in the `research_assistant` conda env.
- [ ] Running `pytest` touches no network and no GPU (verified by using only `:memory:` DB and synthetic data).
- [ ] Each of the four test files has ≥ 3 meaningful assertions against real (not trivially-tautological) behavior.

## Out of scope (deferred)

- `tests/test_memory.py` — `memory.py` does not exist yet; it lands with the memory-pool task.
- Dense/BGE and reranker coverage — requires GPU/model; separate from smoke tests.
- Property-based / fuzz tests, coverage thresholds, CI wiring.
