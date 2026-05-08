# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI image segmentation API (FastAPI + Python 3.13) using SAM 2 for segmentation and LaMa for inpainting. Frontend stack TBD.

## Repository Structure

```
backend/    # FastAPI service — Poetry-managed Python env
frontend/   # Client-side code (stack TBD)
```

## Backend Commands

All backend commands must be run from `backend/`.

```bash
# Install dependencies
poetry install

# Run dev server (once fastapi app exists)
poetry run uvicorn app.main:app --reload

# Linting & formatting
poetry run ruff check .
poetry run ruff format .
poetry run black .
poetry run isort .
poetry run mypy .

# Pre-commit (runs black → isort → ruff → ruff-format → mypy)
poetry run pre-commit run --all-files
```

## Tooling Notes

- **perflint** was intentionally excluded — ruff bundles its rules as the `PERF` ruleset, making the package redundant and it conflicts with `isort ^8`.
- **`language_version: python3`** in `.pre-commit-config.yaml` (not `python3.12`) — only Python 3.13 is installed on this machine.
- `B008` is ignored in ruff — FastAPI's `Depends()` pattern triggers it by design.
- `ignore_missing_imports = true` in mypy — torch/cv2/sam2 lack complete type stubs.
- CV model weight files (`*.pt`, `*.pth`, `*.safetensors`, `models/`) are gitignored — SAM 2 checkpoints are ~900 MB.
