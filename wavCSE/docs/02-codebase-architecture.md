# wavCSE Codebase Architecture

> Full exploration of the wavCSE implementation at `/home/kevin/projects/fyp/wavCSE/`

---

## 1. Directory Structure

```
wavCSE/
├── .git/                        # Git repository
├── .gitignore
├── environment.yml              # Conda environment (Python 3.9, PyTorch 2.7.1, CUDA 12)
├── readme.md                    # Official documentation
├── upstream/                    # Stage 1: Embedding extraction
│   ├── main.py                  # Entry point for embedding extraction
│   ├── configs/
│   │   └── extract_embedding.yml
│   ├── dataset/
│   │   ├── build_embedding.py   # Dataset builder factory
│   │   └── extract_embedding.py # Per-dataset embedding extractors
│   ├── model/
│   │   ├── load_upstream.py     # Upstream model loader factory
│   │   └── wavlm/
│   │       ├── WavLM.py         # Core WavLM model (Microsoft UniLM)
│   │       ├── wavlm_loader.py  # Checkpoint loading
│   │       └── modules.py       # Transformer/MHA/norm modules
│   ├── pooling/
│   │   └── pooling.py           # 6 frame-level pooling strategies
│   └── utils/
│       ├── constant_mapping.py  # Label-to-index mappings
│       ├── embedding_io.py      # .pt file I/O + CSV writing
│       ├── get_size.py          # File size formatter
│       ├── load_config.py       # YAML config loader
│       ├── pooling_id.py        # Pooling identifier strings
│       ├── setup_device.py      # CUDA/CPU device setup
│       └── setup_logging.py     # Logging configuration
└── downstream/                  # Stage 2: Multi-task training & evaluation
    ├── main.py                  # Entry point for training + evaluation
    ├── configs/
    │   └── build_model.yml
    ├── dataset/
    │   ├── load_embedding.py    # Multi-task embedding loader from disk
    │   ├── preprocess_embedding.py # Per-dataset offline dataloaders
    │   └── custom_emb_dataloader.py # Custom DataLoader with collate_fn
    ├── model/
    │   └── downstream_model.py  # DownstreamMultiTaskModel
    ├── pooling/
    │   └── pooling.py           # 10 layer-level pooling strategies
    ├── trainer/
    │   ├── trainer_model.py     # MultiTasksModelTrainer
    │   └── trainer_utils.py     # Loss, metrics, checkpointing, plotting
    ├── evaluator/
    │   ├── evaluator_model.py   # MultiTasksModelEvaluator
    │   └── evaluator_utils.py   # Evaluation stats, CSV export
    └── utils/
        ├── constant_mapping.py
        ├── load_config.py
        ├── parse_transformer_layers.py
        ├── pooling_id.py
        ├── setup_device.py
        └── setup_logging.py
```

---

## 2. Two-Stage Pipeline

### Stage 1: Upstream (Embedding Extraction)

**Purpose:** Convert raw audio → fixed-dimensional embeddings using pre-trained WavLM Large.

```
Raw Audio (.wav)
    │
    ▼
WavLM Large (frozen, eval mode)
    │  └── extract_features(): returns all 25 transformer layer outputs
    ▼
Layer Concatenation (cat all layer reps along layer dim)
    │  └── Shape: [num_layers, T, 1024] → [T, num_layers*1024]
    ▼
Frame-level Pooling (temporal dim)
    │  └── mean/max/mix/lnp/smp/lse → fixed-size vector
    ▼
Save to Disk (.pt files + CSV metadata)
```

**Run command:**
```bash
cd upstream
python main.py --dataset_name speechcommand --config configs/extract_embedding.yml --device_index 0
```

**Supported datasets:** `speechcommand`, `voxceleb`, `iemocap`, `fluentspeechcommand`

### Stage 2: Downstream (Multi-Task Training)

**Purpose:** Load embeddings from disk → train multi-task model → evaluate.

```
Saved .pt Embeddings (on disk)
    │  └── Select specific transformer layers
    ▼
Projector (Linear: 1024 → 512)
    │
    ▼
Layer Pooling (across selected layers, 10 strategies)
    │  └── learned-norm is the paper's best
    ▼
Dropout (p=0.4) → Hidden Layer (512 → 2000) → Dropout (p=0.6)
    │
    ▼
┌─────────────┬─────────────┬─────────────┐
│  KS Head    │  SID Head   │   ER Head   │
│  (→12)      │  (→1251)    │   (→4)      │
└─────────────┴─────────────┴─────────────┘
```

**Run command:**
```bash
cd downstream
python main.py --task_type ks_si_er_ic --config configs/build_model.yml --device_index 0
```

---

## 3. Key Classes & Their Roles

### 3.1 Upstream

| Class | File | Purpose |
|-------|------|---------|
| `WavLM` | `upstream/model/wavlm/WavLM.py` | Full WavLM model: CNN feature extractor (7 conv layers) + Transformer encoder (24 layers for Large) |
| `WavLMConfig` | same | 30+ model configuration parameters |
| `ConvFeatureExtractionModel` | same | 7-layer CNN: `[(512,10,5)] + [(512,3,2)]*4 + [(512,2,2)]*2` |
| `TransformerEncoder` | same | Stack of TransformerSentenceEncoderLayer with layerdrop |
| `MultiheadAttention` | `modules.py` | Full MHA with relative position bias, gated relative position |
| `Pooling` | `upstream/pooling/pooling.py` | 6 frame-pooling strategies (mean, max, mix, lnp, smp, lse) |
| `BuildEmbedding` | `upstream/dataset/build_embedding.py` | Maps dataset names → embedding extractor classes |
| `SPEECHCOMMANDSEmbedding` | `upstream/dataset/extract_embedding.py` | Speech Commands: audio trim, layer norm, WavLM extraction, pooling |
| `VoxCeleb1Embedding` | same | VoxCeleb1 extraction |
| `IEMOCAPEmbedding` | same | IEMOCAP: filters "fru", maps "exc"→"hap", session-based splits |
| `FluentSpeechCommandEmbedding` | same | Fluent Speech Commands: resampling, action/object/location labels |

### 3.2 Downstream

| Class | File | Purpose |
|-------|------|---------|
| `DownstreamMultiTaskModel` | `downstream/model/downstream_model.py` | Shared backbone + task-specific classifier heads |
| `MultiClassifierOutput` | same | Named tuple: `(logits_tuple, predictions_tuple)` |
| `Pooling` | `downstream/pooling/pooling.py` | 10 layer-pooling strategies including weighted, gated, auto, sap, LNP |
| `LoadEmbedding` | `downstream/dataset/load_embedding.py` | Loads .pt files + builds multi-task index tuples |
| `CombinedDataset` | `downstream/dataset/preprocess_embedding.py` | Merges multiple datasets into one |
| `PercentageSubset` | same | Random subset for quick experiments |
| `CustomEmbDataLoader` | `downstream/dataset/custom_emb_dataloader.py` | DataLoader with custom collation |
| `MultiTasksModelTrainer` | `downstream/trainer/trainer_model.py` | Full training loop with AdamW + ReduceLROnPlateau |
| `CheckpointManager` | `downstream/trainer/trainer_utils.py` | Saves epoch/best/opt checkpoints |
| `MultiTasksModelEvaluator` | `downstream/evaluator/evaluator_model.py` | Loads checkpoints, evaluates on test set |
| `CsvPredictionWriter` | `downstream/evaluator/evaluator_utils.py` | Exports true/pred pairs to CSV |

---

## 4. Multi-Task Learning Implementation Details

### 4.1 The Index Pattern (Core MTL Mechanism)

This is the most important implementation detail. Since no single dataset has joint annotations for all tasks, wavCSE builds a **composite dataset** with a "one-hot index pattern":

```python
# For task_type = "ks_si_er_ic" (4 tasks)
# A Speech Commands sample (KS task):
label_tensor = [12, -1, -1, -1]    # Only position 0 has a real label
# A VoxCeleb sample (SI task):  
label_tensor = [-1, 1251, -1, -1]   # Only position 1 has a real label
# An IEMOCAP sample (ER task):
label_tensor = [-1, -1, 4, -1]      # Only position 2 has a real label
```

During training, `masked_ce_loss()` computes cross-entropy **only on positions where label ≠ -1** (ignore_index). This allows samples from different datasets to flow through the same batch with per-task loss computation.

### 4.2 Loss Computation

```python
# Per-task masked loss
task_losses = [masked_ce_loss(logits_i, labels_i, ignore_index=-1) for i in range(num_tasks)]

# Equal-weighting: all tasks contribute equally
total_loss = sum(task_losses) / num_tasks

# Regularization
total_loss += l1_lambda * L1_norm + l2_lambda * L2_norm
```

### 4.3 Three Checkpoint Types

| Checkpoint | Criterion | Purpose |
|------------|-----------|---------|
| **epoch** | Every epoch (rolling window) | Reproduce any training point |
| **best** | Max overall validation accuracy (`accuracy_all`) | Standard best model |
| **opt** | Max average of per-task validation accuracies | Balances multi-task performance |

The `opt` checkpoint is the paper's novel contribution — it prevents one task from dominating at the expense of others.

---

## 5. Pooling Strategies

### 5.1 Frame Pooling (Temporal Dimension — Upstream)

| Strategy | Description |
|----------|-------------|
| `mean` | Average across time frames |
| `max` | Maximum across time frames |
| `mix` | `α * max + (1-α) * mean` |
| `lnp` | Learned Lp-norm across frames |
| `smp` | Softmax-weighted combination |
| `lse` | Log-Sum-Exp with temperature |

### 5.2 Layer Pooling (Transformer Layer Dimension — Downstream)

The 6 above plus:

| Strategy | Description | Paper Finding |
|----------|-------------|---------------|
| `weighted` | Learned per-layer weights via `nn.Parameter`, softmax-normalized | Baseline |
| `gated` | Gated combination of max and mean | Better than WAP |
| `auto` | Learned alpha for exponential weighting | Good |
| `sap` (Self-Attentive Pooling) | Tanh attention with learned context vector | Weaker |

**Paper's best:** `lnp` (Learned-Norm Pooling) — non-linear, adapts to statistical distribution of layer outputs.

---

## 6. Datasets & Label Mappings

| Task Code | Task Name | Dataset | # Classes | Example Labels |
|-----------|-----------|---------|-----------|----------------|
| `ks` | Keyword Spotting | Google Speech Commands v1 | 12 | yes, no, up, down, left, right, on, off, stop, go, _unknown_, _silence_ |
| `si` | Speaker Identification | VoxCeleb1 | 1251 | Speaker IDs 1–1251 |
| `er` | Emotion Recognition | IEMOCAP | 4 | neutral, happy, angry, sad (excited→happy) |
| `ic` | Intent Classification | Fluent Speech Commands | 31 | activate_lamp, decrease_volume, etc. |

---

## 7. Configuration Design

All experiment settings are in YAML files — **no code changes needed to modify experiments.**

### Upstream Config (`extract_embedding.yml`):
```yaml
root_data_path: /path/to/raw/audio
root_emb_path: /path/to/output/embeddings
upstream_checkpoint_path: /path/to/WavLM-Large.pt
upstream_model_type: wavlm_large
frame_pooling_type: mean
limit: null  # or integer to cap samples
```

### Downstream Config (`build_model.yml`):
```yaml
upstream_model_type: wavlm_large
selected_transformer_layers: "6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13"
frame_pooling_type: mean
layer_pooling_type: lnp
embedding_dim_shared1: 512
embedding_dim_shared2: 2000
dropout: [0.4, 0.6]
batch_size: 2048
epochs: 5
learning_rate: 0.0025
weight_decay: 5.0e-8
l1_lambda: 1.0e-7
l2_lambda: 1.0e-5
saved_checkpoint_count: 5
subset_percentage: 10
```

---

## 8. Data Flow (End-to-End)

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: UPSTREAM                                           │
│                                                             │
│  Raw Audio (.wav)                                           │
│      │                                                      │
│      ▼                                                      │
│  Energy-based silence trimming                              │
│      │                                                      │
│      ▼                                                      │
│  Layer Normalization                                        │
│      │                                                      │
│      ▼                                                      │
│  WavLM Large (frozen)                                       │
│      │  extract_features(output_layer=all)                  │
│      │  → 25 layer representations [(T, 1024) × 25]         │
│      ▼                                                      │
│  Layer Concatenation → [T, 25×1024]                         │
│      │                                                      │
│      ▼                                                      │
│  Frame Pooling (mean) → [25×1024]                           │
│      │                                                      │
│      ▼                                                      │
│  Save: .pt file + CSV row (wavpath, label, size)            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: DOWNSTREAM                                         │
│                                                             │
│  Load .pt embeddings from disk                              │
│      │                                                      │
│      ▼                                                      │
│  Select Transformer Layers (e.g., 16 of 25)                 │
│      │  → [B, 16, 1024]                                    │
│      ▼                                                      │
│  Projector (Linear 1024→512)                                │
│      │  → [B, 16, 512]                                     │
│      ▼                                                      │
│  Layer Pooling (LNP) → [B, 512]                             │
│      │                                                      │
│      ▼                                                      │
│  Dropout(0.4) → Linear(512→2000) → Dropout(0.6)            │
│      │  → [B, 2000] ← This is the unified embedding        │
│      ▼                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │ KS Head  │ SID Head │ ER Head  │ IC Head  │             │
│  │ (→12)    │ (→1251)  │ (→4)     │ (→31)    │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
│      │                                                      │
│      ▼                                                      │
│  Masked CE Loss (ignore_index=-1) + L1/L2 reg               │
│      │                                                      │
│      ▼                                                      │
│  Checkpoints: epoch / best (max acc_all) / opt (max avg)   │
│      │                                                      │
│      ▼                                                      │
│  Evaluation: acc, loss, prediction CSVs per task            │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate opencv

# Key dependencies:
# - Python 3.9
# - PyTorch 2.7.1 + CUDA 12
# - torchaudio 2.7.1
# - HuggingFace: transformers, datasets, huggingface-hub
# - Audio: librosa, pydub, soxr, pysoundfile
# - ML: scikit-learn, pandas, matplotlib, seaborn
# - Tools: hydra-core, hyperpyyaml, speechbrain, captum, gdown, optuna, umap-learn, xgboost
```

---

## 10. Extensibility Points (Relevant to Our FYP)

The codebase is designed to be extended. Key extension points:

1. **New upstream models:** Add to `upstream/model/load_upstream.py` (e.g., HuBERT, wav2vec 2.0)
2. **New pooling strategies:** Add to `downstream/pooling/pooling.py`
3. **New datasets/tasks:** Add to `constant_mapping.py` + new extractor classes
4. **New task combinations:** Any `_`-separated string in `--task_type`
5. **Training strategies:** Modify `downstream/trainer/trainer_model.py` (e.g., task weighting)
6. **New MTL architectures:** Modify `downstream/model/downstream_model.py` (e.g., parameter-based sharing)

For our FYP, we'll likely need to extend items 5 and 6 (task imbalance handling, parameter-based MTL).

---

## 11. Notes

- **Code duplication:** `upstream/utils/` and `downstream/utils/` contain identical copies of several utility files. A refactored version should share these.
- **Pretrained checkpoints available:** Google Drive link for `ks_si_er` task combination (best, epoch100, opt checkpoints).
- **WavLM checkpoint auto-downloads:** The upstream stage downloads WavLM-Large.pt automatically.
- **Subset mode:** `subset_percentage: 10` in config enables quick experiments with only 10% of data.
- **No git in fyp root:** Only `wavCSE/` is a git repo, the parent `fyp/` is not.
