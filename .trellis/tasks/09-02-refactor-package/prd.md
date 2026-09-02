# Refactor to installable `research_assistant` package

## Goal

Migrate the flat `modules/` + `tools/` + `utils/` + `main.py` layout into an
installable `research_assistant/` Python package matching the target layout in
`.trellis/spec/backend/directory-structure.md`, while preserving all current
functionality (paper retrieval, hybrid search, multi-agent, experiment design,
academic writing, LoRA fine-tuning).

## Scope decision

This task covers **A (mechanical restructure) + B (debt cleanup)** only.
Behavior must stay equivalent to today.

- **In scope (A)** — move/rename modules into the package, add `pyproject.toml`, fix `__init__` exports, move the `ResearchAssistant` facade out of `main.py`.
- **In scope (B)** — eliminate bare `except: pass`, migrate `print()` → logger, centralize repeated `OpenAI(...)` client construction, remove unused `sqlalchemy` from `requirements.txt`.
- **Out of scope (deferred)** — `memory.py` (new memory-pool component) and `tests/` (pytest smoke tests). Each will be a separate follow-up task.

## Background — confirmed facts (from code inspection)

### Current layout (actual)

- `main.py` — holds the `ResearchAssistant` facade class **and** a CLI `main()`. This is the current public entry point.
- `modules/paper_retrieval/` — `PaperRetriever`, `PaperInterpreter`, `HybridRetriever`, `MetadataStore`, `PDFMarkdownProcessor`, `RetrievalEvaluator`, `RAGASTestsetGenerator`, `RAGEvaluator`, `EvalSample`
- `modules/experiment_agent/` — `ExperimentPlanner`
- `modules/writing_assistant/` — `AcademicWriter`, `CitationManager`
- `modules/agents/` — `ResearchAgentGraph`, `AgentState`, `PythonExecutorTool`, `StatisticalAnalysisTool`, `DataVisualizationTool`, `PaperRetrievalTool`
- `modules/training/` — `LoRATrainer`, `TrainingConfig`, `AcademicDataProcessor`, `WritingEvaluator`
- `tools/` — `CodeGenerator`, `DataAnalyzer`, `Visualizer`
- `utils/` — `helpers`, `logger`, `query_enhancer`
- `config/` — `config.yaml`, `prompts.yaml`
- No `pyproject.toml`, no `tests/`, no `research_assistant/` package, no `memory.py`, no `.git` directory (only `.gitignore` / `.gitattributes`).

### Spec target layout (`.trellis/spec/backend/directory-structure.md`)

- `research_assistant/` package — `assistant.py` (facade), `config/`, `retrieval/`, `agents/`, `experiment/`, `writing/`, `utils/`
- `optional/finetune/` (training), `examples/`, `tests/`, `pyproject.toml` (core/agent/finetune extras)

### Gaps between spec target and reality

1. **`tools/` placement** — `CodeGenerator`/`DataAnalyzer`/`Visualizer` are NOT in the spec target layout. They carry LLM/domain logic (used by the experiment agent and the facade), so they do not fit the spec's "pure helpers only" definition of `utils/`. → resolved in `design.md`.
2. **`main.py` split** — facade → `research_assistant/assistant.py`; CLI entry needs a home (console entry point and/or `examples/`). → resolved in `design.md`.
3. **`memory.py`** — NEW component. **Deferred** (out of scope this task).
4. **`tests/`** — NEW. **Deferred** (out of scope this task).
5. **Debt cleanup** — in scope (B), listed above.

## Requirements

- **R1.** `research_assistant/` is an installable package (`pip install -e .`), with `pyproject.toml` exposing core/agent/finetune extras per spec.
- **R2.** Every current public symbol (all `__all__` entries listed under "current layout") remains importable from its new package path, with no behavior change.
- **R3.** `from research_assistant import ResearchAssistant` works; the facade is `research_assistant/assistant.py`.
- **R4.** The CLI that currently lives in `main.py` still runs (`--query`, `--workflow`), via a console entry point.
- **R5.** Debt cleanup applied where spec demands: no bare `except: pass` (notably `paper_retriever.py`), status/errors via logger not `print()`, one centralized LLM-client constructor, `sqlalchemy` removed from `requirements.txt`.
- **R6.** Heavy/optional deps stay behind `HAS_X` capability flags; core install does not require torch/GPU.

## Acceptance Criteria

- [ ] `pip install -e .` succeeds and `import research_assistant` works from a fresh interpreter.
- [ ] Every public symbol in the "current layout" table is importable from its new path (a smoke script importing each symbol passes).
- [ ] `research_assistant.assistant.ResearchAssistant` initializes and the CLI (`--query`) runs end-to-end against the existing config.
- [ ] `grep` for `except:\s*pass` returns nothing in the new package.
- [ ] Status/error output goes through `logging` (no `print()` for errors/status outside the CLI's final human output).
- [ ] One centralized LLM-client constructor is used everywhere; `sqlalchemy` is gone from `requirements.txt`.
- [ ] Old `modules/`, `tools/`, `utils/`, `main.py` are handled per the transition decision below (removed or shimmed).

## Out of scope (deferred follow-ups)

- `memory.py` (memory pool) — new component.
- `tests/` (pytest smoke tests).
- `optional/finetune/` content changes beyond moving `modules/training/`.

## Decisions

- **Scope** = A (mechanical restructure) + B (debt cleanup) only. `memory.py` and `tests/` are deferred to separate follow-up tasks.
- **Transition & rollback** = initialize a git repo and commit a baseline **first**, then a clean migration (delete old `modules/`/`tools/`/`utils/`, new package is the only structure). No backward-compat shims.
