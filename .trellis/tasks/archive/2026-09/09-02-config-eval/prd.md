# Config robustness + evaluation CLI

## Goal

1. Fix config robustness: resolve `${VAR}` placeholders and stop hardcoding `device: cuda`.
2. Add `research-assistant --eval` to run the retrieval-evaluation pipeline and print a report.

## Scope

### Config robustness
- Resolve `${VAR}` environment placeholders in loaded config (recursive, e.g. `${DEEPSEEK_API_KEY}` → `os.getenv`).
- Change `device: "cuda"` → `device: "auto"` in `config.yaml`; resolve `auto` → `cuda` if `torch.cuda.is_available()` else `cpu`.

### Evaluation CLI
- `research-assistant --eval` runs `RetrievalEvaluator.evaluate_retriever` against the hybrid retriever and prints the report (Precision/Recall/MRR/Hit@k).
- Uses sample eval data when no real `data/eval/retrieval_eval.json` exists (numbers are ~0 with demo data; real data yields real numbers).

## Requirements

- **R1** `_load_config` resolves `${VAR}` placeholders recursively after YAML load.
- **R2** `device: auto` resolves to cuda/cpu; a `resolve_device()` helper handles it (torch optional, fallback cpu).
- **R3** `--eval` flag in `cli.py` runs the evaluation and prints a report; missing/empty index degrades to a warning, not a crash.
- **R4** Existing 33 tests still pass; add tests for env-placeholder resolution and device resolution.

## Acceptance Criteria

- [ ] `${DEEPSEEK_API_KEY}` in config resolves to the env value (or empty when unset).
- [ ] `device: auto` returns `cpu` when torch absent / no CUDA.
- [ ] `research-assistant --eval` prints an evaluation report without crashing.
- [ ] `pytest tests/` passes (existing + new).

## Out of scope

- A real evaluation dataset (requires real paper ids) — the README's metrics are not reproduced here.
- RAGAS / writing (ROUGE) evaluation — LLM-dependent.
- Full config schema validation.
