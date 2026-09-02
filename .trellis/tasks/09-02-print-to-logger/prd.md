# Migrate print() to logging

## Goal

Route status/error output through the `logging` module instead of `print()`,
per `.trellis/spec/backend/logging-guidelines.md`. `print()` remains only for
the CLI's final human-readable output and `__main__` demo blocks.

## Scope

Migrate `print()` calls in the library modules (all of `research_assistant/`
except `cli.py`) to `logger` calls, mapped by intent:

| Current pattern | Example | Target level |
|---|---|---|
| `print(f"❌ Error …: {e}")` | academic_writer, experiment_planner, paper_interpreter, code_generator, data_analyzer, evaluation | `logger.error` |
| `print("⚠️ …")` | facade "not available", "index failed", "config load failed" | `logger.warning` |
| `print("✅ …")`, `print("🔬 …")` milestones | "found N papers", "experiment plan generated" | `logger.info` |

**Stays as `print()`**: `cli.py` (final results / usage text) and each module's
`if __name__ == "__main__":` demo block.

## Requirements

- **R1** Each migrated module gets `logger = logging.getLogger(__name__)` (4 modules already have it from the `except:pass` cleanup — reuse, don't duplicate).
- **R2** Logging is configured once at the CLI entry point (`cli.py main()`) so status/errors are actually visible (currently nothing configures the root logger).
- **R3** Message content and emoji are preserved where they aid readability; only the delivery channel (`print` → `logger`) changes.

## Acceptance Criteria

- [ ] `grep -rn "print(" research_assistant/ --include="*.py"` shows no error/status `print()` outside `cli.py` and `__main__` blocks.
- [ ] All `❌`/`⚠️`/status messages go through `logger.error/warning/info`.
- [ ] `research-assistant --query …` still prints its final result lines.
- [ ] `pytest tests/` still passes (26 tests).

## Out of scope

- Config-driven log-file output (`logs/research_assistant.log`) wiring beyond the console.
- Changing log *message wording* (only the channel changes).
- Debug-level logging additions.
