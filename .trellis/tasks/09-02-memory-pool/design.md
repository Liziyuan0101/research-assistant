# Design — user memory pool

## 1. Data model (SQLite, per `user_id`)

```sql
CREATE TABLE IF NOT EXISTS preferences (
    user_id    TEXT NOT NULL,
    category   TEXT NOT NULL,          -- 'journal' | 'keyword' | 'method'
    value      TEXT NOT NULL,
    weight     REAL DEFAULT 1.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, category, value)
);

CREATE TABLE IF NOT EXISTS qa_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    query      TEXT NOT NULL,
    answer     TEXT,
    success    INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 2. Component & API

`MemoryStore` mirrors the existing `MetadataStore` conventions (raw `sqlite3`,
`Path` db path, per-operation connection, `CREATE TABLE IF NOT EXISTS`).

```python
class MemoryStore:
    def __init__(self, db_path: str): ...
    def add_preference(self, user_id, category, value, weight=1.0) -> bool
    def get_preferences(self, user_id) -> Dict[str, List[str]]   # {journal, keyword, method}
    def add_qa(self, user_id, query, answer, success=True) -> int
    def recall(self, user_id, query=None, top_k=5) -> Dict       # {'preferences', 'qa_history'}
```

- `recall(query)` ranks `qa_history` by token-overlap of `query` against the
  stored `query` (case-insensitive; no GPU), returns top-k.
- `add_preference` upserts; repeated value bumps `weight`/`updated_at`.

## 3. Personalization — hard vs soft

- **Hard (deterministic, testable)** — `search_papers` reads the user's
  `keyword` preferences and appends them to the search query, so retrieval is
  measurably influenced.
- **Soft (prompt-level)** — `format_preferences(preferences)` returns a short
  preamble ("用户偏好期刊: …; 关键词: …; 方法: …") that generation call-sites
  can prepend to their prompt. MVP wires it into at least the facade's
  interpretation/answer path; it is advisory, not guaranteed.

## 4. Integration

- `ResearchAssistant.__init__` gains `user_id: str = "default"` and creates
  `self.memory = MemoryStore(data_dir / "memory.db")`.
- `search_papers` calls `self.memory.get_preferences(user_id)` and, if keyword
  prefs exist, merges them into the enhanced query.
- `format_preferences` lives in `memory.py` and is imported where prompts are built.

## 5. Storage location

`data/memory.db` (runtime, gitignored), consistent with `MetadataStore` at
`data/metadata/papers.db`. Uses `_PROJECT_ROOT / 'data'` as the anchor.
