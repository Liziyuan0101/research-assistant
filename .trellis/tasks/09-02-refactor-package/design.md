# Design — Refactor to installable `research_assistant` package

## 1. Target architecture

Move from a flat, root-level layout to an installable package. The new layout is
the package plus an out-of-package `optional/finetune/` and `examples/`.

```
research_assistant/            # the installable package
├── __init__.py                # re-exports ResearchAssistant (public API)
├── assistant.py               # ResearchAssistant facade (moved from main.py)
├── cli.py                     # CLI main() (moved from main.py) + console entry point
├── config/
│   ├── config.yaml            # moved from config/
│   └── prompts.yaml
├── retrieval/                 # ← modules/paper_retrieval/
├── agents/                    # ← modules/agents/
├── experiment/                # ← modules/experiment_agent/
├── writing/                   # ← modules/writing_assistant/
├── tools/                     # ← tools/  (spec extension — see §4)
└── utils/                     # ← utils/
optional/
├── __init__.py                # NEW: makes `optional.finetune` importable
└── finetune/                  # ← modules/training/
examples/                      # existing (update its import path)
pyproject.toml                 # NEW
```

## 2. Boundary & ownership

- **`research_assistant/`** is the only thing `pip install` packages. Its public
  surface is `research_assistant.ResearchAssistant` (re-exported from
  `assistant.py`) plus each subpackage's existing `__all__`.
- **`optional/finetune/`** is deliberately outside the package (matches spec:
  "NOT part of core install"). It becomes importable as `optional.finetune`
  (new `optional/__init__.py`). The facade keeps its `try/except ImportError`
  guard around `LoRATrainer`/`TrainingConfig`/`WritingEvaluator`.
- **`examples/`** stays at root and is updated to import from `research_assistant`.
- **`data/`, `output/`, `logs/`** stay at project root (runtime, gitignored).

## 3. Contracts (public API — must be preserved verbatim)

Every `__all__` symbol in the current `__init__.py` files is a contract; each
must remain importable from its new path with no behavior change:

- `retrieval/`: `PaperRetriever`, `PaperInterpreter`, `HybridRetriever`, `MetadataStore`, `PDFMarkdownProcessor`, `RetrievalEvaluator`, `RAGASTestsetGenerator`, `RAGEvaluator`, `EvalSample`
- `agents/`: `ResearchAgentGraph`, `AgentState`, `PythonExecutorTool`, `StatisticalAnalysisTool`, `DataVisualizationTool`, `PaperRetrievalTool`
- `experiment/`: `ExperimentPlanner`
- `writing/`: `AcademicWriter`, `CitationManager`
- `tools/`: `CodeGenerator`, `DataAnalyzer`, `Visualizer`
- `utils/`: `setup_logger`, `load_config`, `save_json`, `load_json`
- `optional.finetune/`: `LoRATrainer`, `TrainingConfig`, `AcademicDataProcessor`, `WritingEvaluator`

## 4. Key decision — `tools/` placement (spec deviation)

The spec target layout omits `tools/`. `CodeGenerator`/`DataAnalyzer`/
`Visualizer` carry LLM/domain logic (they call the LLM to generate code /
analyze data / plot), so they do not fit the spec's "pure helpers" `utils/`.
They are used by the experiment agent **and** instantiated directly by the
facade.

**Decision:** create `research_assistant/tools/` as a first-class subpackage.
This is a minimal, honest deviation from the spec target; it should be
backfilled into `.trellis/spec/backend/directory-structure.md` afterwards
(via `trellis-update-spec`).

## 5. Import migration

Cross-package absolute imports that must change:

| Current | New |
|---|---|
| `main.py` → `from modules.paper_retrieval import …`, `from tools import …`, `from utils.query_enhancer import …`, guarded `from modules.agents` / `from modules.training` | `assistant.py` uses relative imports (`from .retrieval import …`, `from .tools import …`, `from .utils.query_enhancer import …`, guarded `from .agents` / `from optional.finetune`) |
| `examples/01_paper_retrieval.py` → `from modules.paper_retrieval.evaluation import …` | `from research_assistant.retrieval.evaluation import …` |
| `paper_retriever.py` → `from utils.query_enhancer import QueryEnhancer` | `from ..utils.query_enhancer import QueryEnhancer` |
| `lora_trainer.py` string template → `from modules.training import LoRATrainer, TrainingConfig` | template emits `from optional.finetune import LoRATrainer, TrainingConfig` |

All in-package relative imports (`.paper_retriever`, `.tools`, etc.) survive the
move unchanged because the subpackage structure is preserved.

## 6. Facade & config/data path resolution (risky area)

Today `main.py` resolves paths with `Path(__file__).parent` (project root):

- config: `Path(__file__).parent / 'config'` → after the move, `assistant.py` is
  `research_assistant/assistant.py`, and `config/` moves to
  `research_assistant/config/`, so `Path(__file__).parent / 'config'` keeps
  working.
- data: `Path(__file__).parent / 'data'` (used for `HybridRetriever(data_dir=…)`)
  must keep pointing at **project root** `data/`, not `research_assistant/data/`.
  → change to `Path(__file__).resolve().parent.parent / 'data'`.

This is behavior-preserving under `pip install -e .` (editable install resolves
`__file__` into the source tree). Non-editable install is out of scope.

## 7. Packaging

`pyproject.toml` (setuptools or hatchling) with:

- `requires-python >= 3.10`
- `[project.scripts] research-assistant = "research_assistant.cli:main"`
- `[project.optional-dependencies]`:
  - `core` — light, GPU-free base deps (openai, langchain*, requests, bs4,
    arxiv, scholarly, semanticscholar, pypdf, python-docx, pdfplumber,
    pymupdf4llm, langchain-text-splitters, pandas, numpy, scikit-learn,
    matplotlib, seaborn, plotly, pyyaml, python-dotenv, tqdm, tenacity)
  - `agent` — LangGraph agent deps (langgraph, langchain-core)
  - `finetune` — heavy/GPU deps (torch, accelerate, bitsandbytes, peft, trl,
    datasets, evaluate, rouge-score, ragas, FlagEmbedding, rank-bm25,
    sentence-transformers, transformers, faiss-gpu, chromadb)
- `sqlalchemy` is **removed** (spec: unused leftover).

Exact grouping is refined in `implement.md`; final placement is reviewable.

## 8. Trade-offs

- **`tools/` as new subpackage** vs merging into `experiment/`: kept separate
  because the facade instantiates them as top-level members independent of the
  experiment planner; merging would blur the `experiment/` boundary.
- **`optional.finetune` (new `optional/__init__.py`)** vs a flat `finetune/`
  package: `optional/` mirrors the spec's intent that it is not core and keeps
  the door open for future optional modules.
- **Clean migration (no shims)** vs backward-compat shims: clean per spec and
  per the user's decision; the git baseline (§9) provides the safety net.

## 9. Rollback / operational safety

- First step of implementation is `git init` + commit the current tree as a
  baseline. Every subsequent move is a `git mv`, so history is clean and any
  step is revertible via `git checkout`.
- Final validation (see `implement.md`) must pass before deleting anything;
  moves are done with `git mv` (not copy+delete) so nothing is lost even on
  abort.
