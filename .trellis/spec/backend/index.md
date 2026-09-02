# Backend Development Guidelines

> Best practices for backend development in this project.

---

## Overview

This directory contains the project's coding conventions. These are the **target** conventions the refactor is migrating toward; every future `trellis-implement` / `trellis-check` sub-agent loads these before writing or reviewing code.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Module organization and file layout | Done |
| [Database Guidelines](./database-guidelines.md) | sqlite3 patterns, queries, naming | Done |
| [Error Handling](./error-handling.md) | Error types, handling strategies | Done |
| [Quality Guidelines](./quality-guidelines.md) | Code standards, forbidden patterns | Done |
| [Logging Guidelines](./logging-guidelines.md) | Structured logging, log levels | Done |

---

## How These Guidelines Are Used

Each AI coding task spawns two sub-agents — `trellis-implement` (writes code) and `trellis-check` (verifies quality). The platform hook injects these spec files plus the task's `prd.md` into every sub-agent prompt, so code is written and reviewed against these conventions automatically.

---

**Language**: All documentation should be written in **English**.
