# Implement — Refactor to installable `research_assistant` package

Ordered checklist. Each step is verified before the next. All destructive
moves use `git mv`; the git baseline from step 1 is the universal rollback.

## Phase 0 — Safety baseline

- [ ] 0.1 `git init` at project root.
- [ ] 0.2 Confirm `.gitignore` covers `data/`, `output/`, `logs/`, `__pycache__/`, `.env`.
- [ ] 0.3 `git add -A && git commit -m "baseline before package refactor"`.
- [ ] 0.4 Verify baseline: `git status --short` is clean.

## Phase 1 — Create the package skeleton

- [ ] 1.1 Create `research_assistant/` with `__init__.py`, `assistant.py` (empty stub first), `cli.py`, `config/`, `retrieval/`, `agents/`, `experiment/`, `writing/`, `tools/`, `utils/`.
- [ ] 1.2 `git mv modules/paper_retrieval/* research_assistant/retrieval/` (preserve `__init__.py`).
- [ ] 1.3 `git mv modules/agents/* research_assistant/agents/`.
- [ ] 1.4 `git mv modules/experiment_agent/* research_assistant/experiment/`.
- [ ] 1.5 `git mv modules/writing_assistant/* research_assistant/writing/`.
- [ ] 1.6 `git mv tools/* research_assistant/tools/`.
- [ ] 1.7 `git mv utils/* research_assistant/utils/`.
- [ ] 1.8 `git mv config/* research_assistant/config/`.
- [ ] 1.9 `git mv modules/training optional/finetune` (create `optional/__init__.py`).

## Phase 2 — Fix imports

- [ ] 2.1 `research_assistant/retrieval/paper_retriever.py`: `from utils.query_enhancer import QueryEnhancer` → `from ..utils.query_enhancer import QueryEnhancer`.
- [ ] 2.2 Move facade: copy `main.py` → `research_assistant/assistant.py`; rewrite its imports to relative (`from .retrieval import …`, `from .tools import …`, `from .utils.query_enhancer import …`, guarded `from .agents` / `from optional.finetune`).
- [ ] 2.3 Fix facade path resolution: `Path(__file__).parent / 'config'` stays; `data` and `output` paths → project root via `Path(__file__).resolve().parent.parent`.
- [ ] 2.4 `research_assistant/cli.py`: move `main.py`'s `main()`/`argparse` here; expose `def main()`.
- [ ] 2.5 `research_assistant/__init__.py`: `from .assistant import ResearchAssistant`; `__all__ = ["ResearchAssistant"]`.
- [ ] 2.6 `research_assistant/tools/__init__.py` and each subpackage `__init__.py`: keep `__all__` intact (relative imports already correct).
- [ ] 2.7 `examples/01_paper_retrieval.py`: `from modules.paper_retrieval.evaluation` → `from research_assistant.retrieval.evaluation`.
- [ ] 2.8 `optional/finetune/lora_trainer.py` string template: `from modules.training import …` → `from optional.finetune import …`.

## Phase 3 — Packaging

- [ ] 3.1 Write `pyproject.toml` (setuptools/hatchling, `requires-python >=3.10`, `[project.scripts]`, extras core/agent/finetune per design §7).
- [ ] 3.2 Remove `sqlalchemy>=2.0.0` from `requirements.txt` (or regenerate the core/agent/finetune split from it).
- [ ] 3.3 `pip install -e ".[core]"` in the project venv (or note for user to run).

## Phase 4 — Debt cleanup (B)

- [ ] 4.1 `research_assistant/retrieval/paper_retriever.py`: replace bare `except: pass` with `except XxxError as e: logger.warning(...)`. Grep `except:\s*pass` → none.
- [ ] 4.2 Migrate status/error `print()` → `logger` across the package; keep CLI-facing final output in `cli.py`.
- [ ] 4.3 Centralize repeated `OpenAI(api_key=…, base_url=…)` construction into one helper (e.g. `research_assistant/utils/llm.py`); replace call sites.
- [ ] 4.4 Confirm optional deps remain behind `HAS_X` flags; core import path doesn't require torch/FlagEmbedding/PyMuPDF at module top.

## Phase 5 — Validation (gate before delete)

- [ ] 5.1 `python -c "import research_assistant; from research_assistant import ResearchAssistant"`.
- [ ] 5.2 Smoke-import every symbol in design §3 from its new path (small script or `python -c` loop).
- [ ] 5.3 CLI smoke: `python -m research_assistant.cli --help` and `research-assistant --help` (if installed).
- [ ] 5.4 `grep -rn "from modules\|import modules\|from tools\|from utils.query_enhancer" --include=*.py` outside `.trellis/` → only `examples/`/docs if updated, none in package.
- [ ] 5.5 `grep -rn "except:\s*pass" --include=*.py` → none.

## Phase 6 — Clean migration (delete old)

- [ ] 6.1 `git rm modules/__init__.py` (now empty after subpackage moves) and remove empty `modules/` dir.
- [ ] 6.2 `git rm main.py` (superseded by `assistant.py` + `cli.py`).
- [ ] 6.3 Remove now-empty `tools/`, `utils/`, `config/` root dirs (files already `git mv`'d).
- [ ] 6.4 `git add -A && git commit -m "refactor: flat modules/ → installable research_assistant package"`.

## Rollback points

- Before Phase 6: `git reset --hard` reverts everything (moves are reversible; files still exist at old paths until `git rm`).
- After any phase: `git status` should show only intended changes; each phase can be committed independently.
- If `pip install -e .` reveals a packaging issue, fix `pyproject.toml`/`__init__` and re-run Phase 5 — no code paths are lost.

## Follow-ups (separate tasks, out of scope here)

- `memory.py` memory pool (spec target).
- `tests/` pytest smoke tests.
- Backfill `tools/` subpackage into `.trellis/spec/backend/directory-structure.md`.
