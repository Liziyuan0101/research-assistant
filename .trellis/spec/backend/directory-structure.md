# Directory Structure

> How backend code is organized in this project.

---

## Overview

This is a single Python package (`research_assistant`), not a web service. There are no HTTP endpoints. The public API is the `ResearchAssistant` facade class in `research_assistant/assistant.py`; everything else is internal.

The project is being migrated from a flat `modules/` layout to an installable package. The layout below is the **target** that new code must follow.

---

## Directory Layout

```
research-assistant/
├── pyproject.toml              # package metadata + extras (core/agent/finetune)
├── README.md
├── .env.example                # placeholder API keys (never commit real keys)
├── research_assistant/         # the installable package
│   ├── __init__.py             # re-exports public API (ResearchAssistant)
│   ├── assistant.py            # ResearchAssistant facade class
│   ├── config/                 # config.yaml + prompts.yaml
│   ├── retrieval/              # paper retrieval, hybrid search, PDF, memory
│   │   ├── paper_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── pdf_markdown_processor.py
│   │   ├── paper_interpreter.py
│   │   ├── evaluation.py
│   │   └── memory.py           # memory pool (short/long-term)
│   ├── agents/                 # LangGraph multi-agent
│   │   ├── multi_agent.py
│   │   └── tools.py
│   ├── experiment/             # experiment planning
│   │   └── experiment_planner.py
│   ├── writing/                # academic writing + citations
│   │   ├── academic_writer.py
│   │   └── citation_manager.py
│   └── utils/                  # cross-cutting helpers
│       ├── query_enhancer.py
│       ├── helpers.py
│       └── logger.py
├── optional/
│   └── finetune/               # LoRA fine-tuning (NOT part of core install)
├── examples/                   # runnable demo scripts (01/02/03)
└── tests/                      # pytest smoke tests
```

---

## Module Organization

Each top-level subpackage has one responsibility:

- `retrieval/` — everything about finding and indexing papers (API search, hybrid retrieval, PDF parsing, LLM interpretation, memory).
- `agents/` — LangGraph multi-agent orchestration and agent tools.
- `experiment/` — experiment plan design + code generation.
- `writing/` — academic writing generation + citation management.
- `utils/` — shared helpers with no LLM/domain logic (query enhancement, logging, small utilities).

Rules:

- A new feature goes in the subpackage that owns its domain. Search-related code goes in `retrieval/`; `utils/` is reserved for pure helpers only.
- Heavy, GPU-optional code (fine-tuning) goes in `optional/finetune/`, never in the core package.
- Runtime data (`data/`, `output/`) is generated and never committed; see `.gitignore`.

---

## Naming Conventions

- **Packages/files**: snake_case (`hybrid_retriever.py`, `query_enhancer.py`).
- **Classes**: CamelCase (`HybridRetriever`, `PaperInterpreter`).
- **Methods/variables**: snake_case (`add_papers`, `hybrid_search`).
- One primary class per module, named after the module.

---

## Examples

- `research_assistant/retrieval/hybrid_retriever.py` — canonical example: `MetadataStore` (storage) and `HybridRetriever` (logic) separated cleanly.
- `research_assistant/utils/logger.py` — pure helper with no domain dependencies.
- `examples/01_paper_retrieval.py` — how the public API is meant to be consumed.
