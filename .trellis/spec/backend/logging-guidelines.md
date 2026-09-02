# Logging Guidelines

> How logging is done in this project.

---

## Overview

Use the `logging` module via the shared `setup_logger()` in `research_assistant/utils/logger.py`. **Do not use `print()` for status/errors** — `print()` is reserved for the CLI's final human-readable output only.

The current codebase logs via emoji `print()` statements; the refactor migrates these to `logger` calls. Keep emoji in CLI-facing `print()` output where it aids readability, but route errors/status through the logger.

---

## Log Levels

- `DEBUG` — per-chunk/per-query detail (tokenization, embedding shapes).
- `INFO` — workflow milestones (index built, N papers fetched, model loaded).
- `WARNING` — degraded but acceptable (external API down → empty result, optional dep missing, cache miss).
- `ERROR` — a real failure that produced a fallback or aborted an operation.

---

## Structured Logging

Format is fixed in `setup_logger()`:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

- Use `logger = setup_logger(__name__)` per module so logs carry the module name.
- Prefer lazy formatting: `logger.warning("failed: %s", e)` over f-strings (avoids formatting cost on suppressed levels).
- Suppress noisy third-party loggers at startup (already done in `hybrid_retriever.py`):

```python
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
```

---

## What to Log

- Retrieval: query, top-k, result count, index stats after rebuild.
- External API: request failure reason and the source (arxiv/openalex/...).
- LLM: which operation failed and a truncated error; not the full prompt unless debugging.
- Lifecycle: model load success/failure, index load from disk.

---

## What NOT to Log

- **API keys or tokens** (DeepSeek/OpenAI/CORE keys) — never log config values that may contain secrets.
- Full user prompts that may contain sensitive text.
- Full paper PDFs / large bodies.
