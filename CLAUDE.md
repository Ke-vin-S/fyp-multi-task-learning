# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Analyzing the Impact of Multi-task Learning Strategies on Speech Task Performance** — target venue ACL 2027.

### Research Problem

There are few MTL models in speech processing, and no models that combine all four task types (content, speaker, emotion, intent) together. wavCSE covers only three. This project builds the **first parameter-based MTL architectures for speech**, extending beyond the feature-based approach of wavCSE.

### Approach

We take **parameter-based MTL** methods (developed primarily for computer vision) and adapt them for speech processing. Each team member develops one architecture. We compare against two benchmarks:

1. **Pass wavCSE** — parameter-based must beat the feature-learning baseline
2. **Pass single-task models** — prove training tasks together is better than training them separately

### Tasks

Content (linguistic), Speaker (identity), Emotion (paralinguistic), plus Intent Classification as an expansion target.

### Deliverables

A Python MTL library with innovative task-imbalance handling, plus a systematic experimental analysis. Target publication: ACL 2027.

Full project scope at [`docs/project-description.md`](docs/project-description.md). FYP extension context at [`wavCSE/docs/03-fyp-project-context.md`](wavCSE/docs/03-fyp-project-context.md).

### People

- **Supervisor**: Dr. Uthayasanker Thayasivam (Udaya sir)
- **Co-supervisor**: Braveenan Sritharan (author of wavCSE, IJCNLP 2025)
- **Team**: 4 members, each developing one parameter-based MTL architecture

## Repository Structure

Sub-projects follow a two-level pattern: **top-level folder** manages everything, **second-level folder** is the actual git repo. Each sub-project has its own `CLAUDE.md` with specific architecture docs, commands, and gotchas.

```
fyp/
├── docs/                     # Global project docs: project description, paper list
├── sessions/                 # Saved Claude chat transcripts (global/architectural)
├── weekly/                   # Weekly supervisor meetings
│   └── week-NN/
│       ├── meeting-summary.md    # Transcript + supervisor feedback
│       ├── my/                   # Personal work: slides, analysis, paper notes
│       └── assets/               # Diagrams, images
├── wavCSE/                   # Management wrapper for the wavCSE baseline
│   ├── CLAUDE.md             # ← wavCSE-specific instructions (architecture, commands, gotchas)
│   ├── wavCSE/               # Actual git repo (upstream/ + downstream/ pipeline)
│   ├── docs/                 # Paper summary, architecture docs, FYP context
│   ├── improvements/         # Our new MTL architectures (PMR, TSM, GBC)
│   └── WAVCSE_ANALYSIS.md    # Deep-dive reference for wavCSE internals
├── LibMTL/                   # (to be added) Management wrapper for LibMTL
│   └── LibMTL/               # Actual git repo
```

### Sub-Project CLAUDE.md Files

| Sub-project | CLAUDE.md | When loaded |
| --- | --- | --- |
| wavCSE | [`wavCSE/CLAUDE.md`](wavCSE/CLAUDE.md) | Automatically when working in `wavCSE/` |
| LibMTL | `LibMTL/CLAUDE.md` | (to be created) |

Each sub-project CLAUDE.md is self-contained — it can be used directly when running Claude Code from that folder.

## Working Conventions

- **Weekly meetings**: Thursday mornings, 8:30 AM. Each week gets `weekly/week-NN/`. Meeting notes in `meeting-summary.md`. Personal work (slides, analysis) in `my/`. Shared assets (diagrams) in `assets/`. Use the `academic-pptx` skill (`.claude/skills/academic-pptx/`) for weekly presentation slides — it enforces action titles, communication-first design, and academic slide standards.
- **Transcript hygiene**: When extracting or saving meeting transcripts, strip anything that could affect people — personal schedules, private conversations, travel plans, off-topic remarks. Only keep research decisions, technical guidance, action items, and benchmark definitions.
- **Saving Claude sessions**: Use `/copy` to capture transcripts as `.md` files. Global/architectural chats → `sessions/`. Week-specific → `weekly/week-NN/my/sessions/`.
- **Papers**: PDFs are git-ignored. Track every paper in [`docs/papers.md`](docs/papers.md) — when you add a PDF to any folder, update the list with title, arXiv link, and status (to-read/reading/done). Paper summaries go in `weekly/week-NN/my/papers/summaries/`.
- **Sub-projects** (`wavCSE`, `LibMTL`): Each has its own venv and git repo. Always `cd` into the inner repo before running code.
- **Python version**: Python 3.9 is used by wavCSE. Be aware of version constraints when writing new code — prefer compatibility with 3.9+.
- **Commits**: Stage selectively — never `git add -A` or `git add .`. Verify with `git diff --cached --name-only` before committing. Keep out: PDFs, node_modules, .venv, .ppt/.pptx (unless weekly slides), settings.local.json. Commit messages: imperative mood, explain why not what. One logical change per commit.

## Gotchas

- wavCSE uses Python 3.9 — no `match` statements, no `X | Y` type unions. See [`wavCSE/CLAUDE.md`](wavCSE/CLAUDE.md) for more wavCSE-specific gotchas.
- The `.gitignore` has `*.pdf` but `.ignore` has `!*.pdf` — the `.ignore` file may override gitignore behavior for PDFs.
- Sub-project git repos live one level deep (`wavCSE/wavCSE/`, `LibMTL/LibMTL/`) — run git commands from the inner folder.
