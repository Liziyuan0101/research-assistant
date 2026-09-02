# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

This is a research/RAG codebase with heavy ML dependencies. Quality means: public API is small and clean, optional dependencies degrade gracefully, and there are no silent failures. Style follows PEP 8 with the additions below.

---

## Forbidden Patterns

- **Bare `except: pass`** (currently in `paper_retriever.py`) — see error-handling.md.
- **`print()` for status/errors** — use the logger (see logging-guidelines.md).
- **Hardcoding API keys** — read from `.env` / `os.getenv` only; commit only `.env.example`.
- **Importing heavy deps at module top without a `HAS_X` guard** — `torch`/`FlagEmbedding`/`PyMuPDF` must be optional so `core` installs without them.
- **Duplicated LLM-client construction** — the `OpenAI(api_key=..., base_url=...)` init is repeated across modules; centralize into one helper during the refactor.

---

## Required Patterns

- **Chinese docstrings** on every module (top) and every public method, describing purpose and args/returns.
- **Type hints everywhere** using `typing` (`List`, `Dict`, `Optional`).
- **`__all__` in `__init__.py`** listing the public API, so `from research_assistant import ResearchAssistant` works.
- **`HAS_X` capability flags** for optional dependencies (see error-handling.md).
- **`if __name__ == "__main__":` demo block** at the bottom of each module, showing a minimal working example.
- Config access via `config.get("section", {})` with a sensible default, never bare `config["key"]`.

Canonical module header:

```python
"""
Hybrid Retrieval System
混合检索系统：BM25 + BGE-M3 + Reranker
"""

class HybridRetriever:
    """融合 BM25 与稠密向量检索，二阶段精排。"""

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """执行混合检索。"""
        ...
```

---

## Testing Requirements

There are currently **no tests**. During the refactor, add minimal `pytest` smoke tests under `tests/`:

- `tests/test_memory.py` — memory pool add/recall round-trip.
- `tests/test_hybrid_retriever.py` — index a few synthetic papers and search (skip if BGE not installed).
- `tests/test_metadata_store.py` — SQLite CRUD against `:memory:`.

Tests must run without network access and without GPU.

---

## Code Review Checklist

- [ ] No `except: pass`, no `print()` for errors.
- [ ] Optional deps behind `HAS_X`; `core` installs without torch/GPU.
- [ ] Docstrings + type hints present.
- [ ] API key not committed; `.env` ignored.
- [ ] New heavy code placed in `optional/` if GPU/optional.
