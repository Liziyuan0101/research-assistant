# Error Handling

> How errors are handled in this project.

---

## Overview

The project talks to external services (arXiv, OpenAlex, LLM APIs) and optional heavy dependencies (FlagEmbedding, PyMuPDF). The rule:

- **Optional dependency missing** → feature disabled gracefully (`HAS_X = False`).
- **External API/LLM failure** → log a warning and return an empty/fallback result (degrade, don't crash).
- **Real bug / programmer error** → let it raise; never swallow silently.

The current codebase has a widespread anti-pattern of bare `except: pass` (notably in `paper_retriever.py`). The refactor must eliminate these.

---

## Error Types

No custom exception classes yet. Use built-in and standard-library exceptions (`ImportError`, `requests.exceptions.RequestException`, `sqlite3.Error`). Add a small set of custom exceptions only if a caller needs to catch a specific failure class.

---

## Error Handling Patterns

**1. Optional import → capability flag (keep this pattern):**

```python
try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGE = True
except ImportError:
    HAS_BGE = False
```

Later, check `if not HAS_BGE: raise ImportError(...)` at the point of use.

**2. External API call → log + graceful empty result (do NOT use `except: pass`):**

```python
try:
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    logger.warning("OpenAlex request failed: %s", e)
    return []
```

**3. LLM call → log + fallback:**

```python
try:
    response = self.client.chat.completions.create(...)
    return response.choices[0].message.content.strip()
except Exception as e:
    logger.error("LLM generation failed: %s", e)
    return self._fallback_summary(paper)
```

**4. Database write → log + sentinel return:**

```python
try:
    cursor.execute(...)
    conn.commit()
    return cursor.lastrowid
except sqlite3.Error as e:
    logger.error("Failed to insert chunk: %s", e)
    return -1
finally:
    conn.close()
```

---

## API Error Responses

This is a library, not an HTTP service — there are no API error responses. "Clients" are the facade methods on `ResearchAssistant`; on failure they return a sensible empty value (`[]`, `{}`, `None`) rather than raising, and log the cause.

---

## Common Mistakes

- **Bare `except: pass`** — hides every failure and makes debugging impossible. Forbidden.
- Catching `Exception` and returning success-shaped data without logging → the caller cannot tell "empty because nothing matched" from "empty because it crashed". Always log on non-`ImportError` catches.
- Swallowing `ImportError` for a *required* dependency (vs an optional one).
