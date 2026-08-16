# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

wavCSE is the baseline for the FYP (Multi-Task Learning for Speech, target ACL 2027). This folder is the **management wrapper** — the actual git repo lives at `wavCSE/wavCSE/`.

## Directory Structure

```
wavCSE/
├── wavCSE/                   # Git repo: upstream + downstream pipeline
│   ├── environment.yml       # Conda env (Python 3.9, PyTorch 2.7.1, CUDA 12)
│   ├── readme.md
│   ├── upstream/             # Stage 1: Embedding extraction
│   │   ├── main.py
│   │   ├── configs/
│   │   ├── dataset/          # build_embedding.py, extract_embedding.py
│   │   ├── model/            # load_upstream.py, wavlm/WavLM.py
│   │   ├── pooling/          # 6 frame-pooling strategies
│   │   └── utils/            # Config, logging, device, I/O helpers
│   ├── downstream/           # Stage 2: Multi-task training + evaluation
│   │   ├── main.py
│   │   ├── configs/
│   │   ├── dataset/          # Embedding loaders, custom DataLoader
│   │   ├── model/            # DownstreamMultiTaskModel
│   │   ├── pooling/          # 10 layer-pooling strategies
│   │   ├── trainer/          # MultiTasksModelTrainer
│   │   ├── evaluator/        # MultiTasksModelEvaluator
│   │   └── utils/
│   └── improvements/         # Our new MTL architectures — lives INSIDE the git
│       ├── run_improvements.py   #   repo (sibling of upstream/downstream), not
│       ├── taskrelation/         #   in this wrapper folder. One folder per
│       │   ├── models/           #   owner (see "Our MTL Improvements" below).
│       │   ├── trainers/
│       │   └── configs/
│       ├── lowrank/          # Chehan — not yet implemented
│       ├── clustering/       # Induwara — not yet implemented
│       └── decomposition/    # Pathumi — not yet implemented
├── docs/                     # Paper summary, architecture docs, FYP context
│   ├── 01-paper-summary.md
│   ├── 02-codebase-architecture.md
│   ├── 03-fyp-project-context.md
│   ├── 04-presentation-plan.md
│   └── 05-quick-reference.md
├── improvements/docs/        # Presentation assets only (slides, diagrams) —
│                              # the code itself lives in wavCSE/improvements/
│                              # above, not here. See gotcha below.
├── papers/                   # Published PDFs (not tracked)
└── WAVCSE_ANALYSIS.md        # Deep-dive reference for wavCSE internals
```

## Reference Docs

Start with these (in `docs/`):

| File | Content |
|------|---------|
| `01-paper-summary.md` | Full paper breakdown: motivation, method, experiments, results |
| `02-codebase-architecture.md` | Directory tree, key classes, data flow, MTL implementation details |
| `03-fyp-project-context.md` | How our FYP extends wavCSE — strategy comparison, task imbalance, deliverables |
| `04-presentation-plan.md` | Slide-by-slide breakdown for group presentation |
| `05-quick-reference.md` | Cheat sheet: acronyms, key numbers, results tables |

The most comprehensive internal reference is [WAVCSE_ANALYSIS.md](WAVCSE_ANALYSIS.md) — covers config schema, pooling strategies, data flow, the pipeline contract between stages, and a complete problem inventory.

## Architecture: Two-Stage Pipeline

```
Raw Audio  ──→  WavLM Large (frozen)  ──→  Frame Pooling  ──→  Layer Pooling  ──→  .pt Embeddings
                                                                                        │
                                              DownstreamMultiTaskModel  ←────────────────┘
                                                   │
                                              Per-Task Heads  ──→  Metrics (KS, SID, ER)
```

### Stage 1: Upstream (embedding extraction)

Frozen WavLM Large extracts all 25 transformer layer outputs. Frame pooling reduces the time dimension (6 strategies). Layer concatenation + FC projects to a fixed-size embedding. Saved as `.pt` files.

### Stage 2: Downstream (training)

Loads `.pt` embeddings from disk → shared backbone (2 FC layers) → 3 task-specific classification heads. Trains jointly with equal-weighted loss. Evaluates on keyword spotting, speaker identification, and emotion recognition.

## Environment

```bash
# Conda env (from the git repo's environment.yml)
conda env create -f wavCSE/environment.yml

# Or use the bundled .venv
source .venv/bin/activate
```

Python 3.9, PyTorch 2.7.1, CUDA 12.

## Running

```bash
# Stage 1: Extract embeddings
cd wavCSE/upstream
python main.py --config configs/extract_embedding.yml

# Stage 2: Train + evaluate downstream model
cd wavCSE/downstream
python main.py --config configs/build_model.yml
```

## Our MTL Improvements

`wavCSE/wavCSE/improvements/` holds our parameter-based MTL architectures, extending the downstream model. Structured **by owner**, not by file type — each of the 4 team members gets their own folder (`taskrelation/`, `lowrank/`, `clustering/`, `decomposition/`) containing their own `models/`, `trainers/`, `configs/`. This keeps each person's branch touching only their own directory, so branches merge cleanly instead of colliding on shared `models/`/`configs/`/`trainers/` folders. See `structure/improvements-by-owner` branch for the migration.

**`taskrelation/` (Kevin)** — three exploratory architectures, all task-relation-learning variants:

| Model | File | Description |
| --- | --- | --- |
| **GBC** | `taskrelation/models/gbc_model.py` | Global Bias Coupling (Zhang et al. 2010) — shared global bias couples all task heads |
| **TSM** | `taskrelation/models/tsm_model.py` | Task Structure Matrix (Ciliberto et al. 2015) — latent classifier basis + learned sparse structure matrix |
| **PMR** | `taskrelation/models/pmr_model.py` | Precision Matrix Regularization (Gonçalves et al. 2016) — sparse precision matrix over task parameters |

TSM and PMR have matching trainers in `taskrelation/trainers/` (alternating-minimization training loops); GBC uses the standard trainer. Configs in `taskrelation/configs/`. Entry point: `improvements/run_improvements.py --model [gbc|tsm|pmr|original|all]`.

`lowrank/` (Chehan), `clustering/` (Induwara), `decomposition/` (Pathumi) are scaffolded (folder + README) but not yet implemented.

## Gotchas

- **Python 3.9** — no `match` statements, no `X | Y` type unions. Write backwards-compatible code.
- **WavLM Large is ~1.2 GB** on disk. Needs a GPU with sufficient VRAM.
- **Upstream/downstream are decoupled by `.pt` files** — if you change pooling config, you must re-extract embeddings before downstream sees the change.
- **Two environments**: conda env from `environment.yml` and the `.venv/` at the root — verify which one is active.
- **`wavCSE/` is the git repo** — run git commands from there, not from this management folder.
- **`improvements/` code lives in the git repo (`wavCSE/wavCSE/improvements/`), not in this wrapper folder.** `wavCSE/improvements/` (this folder, one level up) holds only presentation assets (`docs/` — slides, diagrams, transcripts). It's easy to edit the wrong copy — check which `improvements/` you're in.
- **Running `run_improvements.py` requires `cwd` = `wavCSE/wavCSE/`** (the git repo root) — its `from improvements.taskrelation....` imports only resolve with `improvements/`'s parent on `sys.path`.
