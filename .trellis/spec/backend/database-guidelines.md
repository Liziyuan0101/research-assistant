# Database Guidelines

> Database patterns and conventions for this project.

---

## Overview

The project uses **raw `sqlite3` from the Python standard library** — not SQLAlchemy, even though `sqlalchemy` appears in `requirements.txt` (it is an unused leftover and should be removed during the refactor).

The single metadata database lives at `data/metadata/papers.db` and is managed by the `MetadataStore` class in `research_assistant/retrieval/hybrid_retriever.py`.

There is **no ORM and no migration framework**. Schema is created idempotently on startup.

---

## Query Patterns

- Create schema with idempotent DDL — always `CREATE TABLE IF NOT EXISTS`, so re-running never fails:

```python
cursor.execute('''
    CREATE TABLE IF NOT EXISTS papers (
        paper_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        ...
    )
''')
```

- Upsert with `INSERT OR REPLACE` for metadata that may be re-fetched:

```python
cursor.execute('''
    INSERT OR REPLACE INTO papers (paper_id, title, ...)
    VALUES (?, ?, ...)
''', (paper.paper_id, paper.title, ...))
```

- **Always use parameterized queries** (`?` placeholders), never f-strings/`%` for values.
- Serialize list/dict columns to JSON with `json.dumps` on write and `json.loads` on read (SQLite has no native array column).
- Open/close a connection **per operation** (single-user, low-write; no pooling). Use `try/finally` so the connection always closes.
- Create indexes with `CREATE INDEX IF NOT EXISTS` for columns used in `WHERE`/`JOIN` lookups (e.g. `papers.title`, `chunks.paper_id`).

---

## Migrations

No migration tool. When schema changes:

1. Edit the `CREATE TABLE IF NOT EXISTS` statements in `MetadataStore._init_db()`.
2. For columns added to an existing table, add a guarded `ALTER TABLE ... ADD COLUMN` behind a `PRAGMA table_info` check, or bump a schema version and rebuild.
3. Test against a fresh `:memory:` DB **and** an existing DB before committing.

---

## Naming Conventions

- Tables: plural snake_case (`papers`, `chunks`).
- Columns: snake_case; primary key is `<entity>_id` (`paper_id`); foreign keys are `<ref>_id` (`paper_id` in `chunks`).
- Indexes: `idx_<table>_<column>` (`idx_paper_title`, `idx_chunk_paper`).
- Auto-increment surrogate key: `chunk_id INTEGER PRIMARY KEY AUTOINCREMENT`.

---

## Common Mistakes

- Mixing SQLAlchemy into a codebase that is otherwise raw `sqlite3` — pick one.
- Forgetting `json.dumps` on list fields → `sqlite3` errors or stores a Python repr.
- Not using `try/finally` around `conn.close()` → leaked connections.
- Building SQL with string concatenation → injection risk.
