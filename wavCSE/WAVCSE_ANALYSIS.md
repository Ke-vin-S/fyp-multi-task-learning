# wavCSE — Deep Analysis (Full)

> **Paper:** IJCNLP 2025 — Braveenan Sritharan, Uthayasanker Thayasivam (Univ. of Moratuwa)
> **Repo:** `/home/kevin/projects/fyp/wavCSE/wavCSE/`

---

## Table of Contents

1. [What Is wavCSE?](#what-is-wavcse)
2. [Directory Structure & File Map](#directory-structure--file-map)
3. [Stage 1: Upstream Embedding Extraction — Full Walkthrough](#stage-1-upstream-embedding-extraction)
4. [Stage 2: Downstream Training & Evaluation — Full Walkthrough](#stage-2-downstream-training--evaluation)
5. [Model Architectures](#model-architectures)
6. [Pooling Strategies — Source-Level Detail](#pooling-strategies)
7. [Data Flow & Label Handling](#data-flow--label-handling)
8. [Checkpoint & Evaluation System](#checkpoint--evaluation-system)
9. [Pipeline Contract (Upstream → Downstream)](#pipeline-contract)
10. [Config Schema Reference](#config-schema-reference)
11. [Dependencies & Environment](#dependencies--environment)
12. [Complete Problem Inventory](#complete-problem-inventory)
13. [How to Run (Corrected)](#how-to-run-corrected)

---

## What Is wavCSE?

wavCSE is a **modular speech representation learning framework** for systematic research in self-supervised speech representation learning (SSL) and downstream multi-task modeling. It enforces a strict **two-stage pipeline**:

```
Raw Audio  ──→  Upstream (WavLM-Large)  ──→  .pt Embeddings  ──→  Downstream Multi-Task Model  ──→ Metrics
                    │                                                    │
                    └── Frame Pooling (time dim)        Layer Pooling (layer dim) ←──┘
```

**Design philosophy:**
- **Separation of concerns** — embedding extraction is decoupled from downstream training
- **Reproducibility** — all settings in YAML configs, no code changes needed
- **Extensibility** — designed to support new upstream models, pooling strategies, and downstream tasks
- **Multi-task** — single model with shared backbone + per-task classification heads

---

## Directory Structure & File Map

```
wavCSE/                                        # Git repo root
├── environment.yml                            # Conda env: opencv, Python 3.9
├── readme.md                                  # Project docs, usage instructions
├── .gitignore
│
├── upstream/                                  # STAGE 1: Embedding extraction
│   ├── main.py                                # Orchestrator: parse args → load config → load model → extract → save
│   ├── configs/
│   │   └── extract_embedding.yml              # Upstream config (paths, device, pooling, dataset limit)
│   ├── dataset/
│   │   ├── build_embedding.py                 # BuildEmbedding: dispatches dataset_name → _build_*() method
│   │   └── extract_embedding.py               # Per-dataset wrappers: SPEECHCOMMANDSEmbedding, VoxCeleb1Embedding, IEMOCAPEmbedding, FluentSpeechCommandEmbedding
│   ├── model/
│   │   ├── load_upstream.py                   # Factory: dispatches model_type string → load_wavlm()
│   │   └── wavlm/
│   │       ├── WavLM.py                       # Full WavLM model definition (WavLM, WavLMConfig, TransformerEncoder, ConvFeatureExtractionModel)
│   │       ├── wavlm_loader.py                # Checkpoint loading: torch.load → WavLMConfig → WavLM → load_state_dict → eval
│   │       └── modules.py                     # Submodules: MultiheadAttention (with gated rel pos bias), Fp32LayerNorm, SamePad, Swish, GLU_Linear
│   ├── pooling/
│   │   └── pooling.py                         # 6 frame-pooling strategies (nn.Module)
│   └── utils/
│       ├── constant_mapping.py                # LabelKeywordMapping, TaskKeywordMapping, TaskDatasetMapping
│       ├── embedding_io.py                    # process_and_write_data(): save .pt + write CSV
│       ├── get_size.py                        # Human-readable file size (B/KB/MB)
│       ├── load_config.py                     # yaml.safe_load wrapper
│       ├── pooling_id.py                      # make_pooling_id(), make_pooling_name()
│       ├── setup_device.py                    # set_device(): GPU/CPU with fallback
│       └── setup_logging.py                   # File + console logging with timestamped filenames
│
└── downstream/                                # STAGE 2: Training + Evaluation
    ├── main.py                                # Orchestrator: parse args → load config → load embeddings → build model → train → evaluate 3 checkpoints
    ├── configs/
    │   └── build_model.yml                    # Downstream config (full: model arch, training hparams, eval settings)
    ├── dataset/
    │   ├── custom_emb_dataloader.py            # CustomEmbDataLoader(DataLoader): custom collate_stack
    │   ├── load_embedding.py                   # LoadEmbedding: task_type → dataset dispatch → CombinedDataset → PercentageSubset
    │   └── preprocess_embedding.py             # Per-dataset embedding loaders + CombinedDataset, PercentageSubset
    ├── model/
    │   └── downstream_model.py                 # DownstreamMultiTaskModel: projector → layer pooling → hidden → per-task heads
    ├── pooling/
    │   └── pooling.py                          # 10 layer-pooling strategies (6 base + weighted, gated, auto, sap)
    ├── trainer/
    │   ├── trainer_model.py                    # MultiTasksModelTrainer: full training loop, metrics, plotting
    │   └── trainer_utils.py                    # CheckpointManager, MetricsWriter, masked_ce_loss, masked_accuracy, BatchStats, EpochStats
    ├── evaluator/
    │   ├── evaluator_model.py                  # MultiTasksModelEvaluator: load checkpoint → inference → metrics → CSV
    │   └── evaluator_utils.py                  # Eval-specific BatchStats, EvalStats, CsvPredictionWriter
    └── utils/
        ├── constant_mapping.py                 # IDENTICAL to upstream/ copy
        ├── load_config.py                      # IDENTICAL to upstream/ copy
        ├── parse_transformer_layers.py         # Parse "all", "1-12", "0,2,4" → List[int]
        ├── pooling_id.py                       # IDENTICAL to upstream/ copy
        ├── setup_device.py                     # IDENTICAL to upstream/ copy
        └── setup_logging.py                    # IDENTICAL to upstream/ copy
```

**Key observation:** 6 utility files (`constant_mapping.py`, `load_config.py`, `pooling_id.py`, `setup_device.py`, `setup_logging.py`, and `pooling/pooling.py`) are **duplicated** between `upstream/` and `downstream/`, with the pooling files having diverged (upstream has 6 strategies, downstream has 10).

---

## Stage 1: Upstream Embedding Extraction

### Entry Point: `upstream/main.py`

**Flow:**

```
CLI args (--dataset_name, --config, --device_index)
  ↓
load_config() → cfg dict
  ↓
set_device() → torch.device
  ↓
load_upstream_model() → WavLM model (eval mode, on device)
  ↓
BuildEmbedding(upstream_model, frame_pooling_type, frame_pooling_param, device)
  ↓
BuildEmbedding.build_embedding(dataset_name, root_data_path) → train_data, val_data, test_data
  ↓
process_and_write_data(train_data, append=False)
process_and_write_data(val_data, append=True)
process_and_write_data(test_data, append=True)
  → .pt files + CSV
```

### Dataset Wrappers: `upstream/dataset/extract_embedding.py`

Each dataset wrapper inherits from the corresponding `torchaudio.datasets` class and overrides `get_metadata()` / `__getitem__()`.

**Common pattern in `get_metadata()`:**

```python
def get_metadata(self, n: int):
    metadata = super().get_metadata(n)        # torchaudio's metadata
    waveform = _load_waveform(...)            # load raw audio (except FluentSC)
    waveform = maybe_trim_by_filesize(...)    # energy-based silence trimming
    waveform = waveform.to(device)
    waveform = F.layer_norm(waveform, waveform.shape)  # instance norm on raw audio!

    # WavLM forward pass
    _, layer_reps = self.model.extract_features(
        waveform,
        output_layer=self.model.cfg.encoder_layers,  # all 24 layers
        ret_layer_results=True
    )[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_reps]  # [L, T, D]
    embedding = torch.cat(layer_reps, dim=0)                   # [L*T, D]  ← NOTE: concat!
    embedding = self.frame_pooling.get_vector_after_pooling(embedding, 1)  # [L, D]
    return embedding, label_index, wavpath
```

**⚠️ Critical:** The `torch.cat(layer_reps, dim=0)` concatenates **all layers across the time dimension**, producing `[L*T, D]`. Frame pooling then pools across dimension 1 (the time dimension of this concatenated tensor). This works because for the first layer the output is `[T, D]` and concat gives `[L*T, D]`, so `dim=1` pools the T dimension correctly. However, this is fragile — the concat assumes all layers have identical time steps.

### Silence Trimming

Two functions in `upstream/dataset/extract_embedding.py`:

**`trim_to_non_silent_window()`:** Selects the `target_seconds`-long window with maximum short-time energy from the audio. Uses a moving sum of frame energies via `F.conv1d`.

**`maybe_trim_by_filesize()`:** If file size > `MAX_WAV_BYTES` (1.7 MB) OR waveform longer than `MAX_SECONDS` (60s), apply energy-based trimming. This handles VoxCeleb1's variable-length recordings.

### Audio Preprocessing Differences Across Datasets

| Dataset | Loading Method | Sample Rate | Mono Conversion |
|---------|---------------|-------------|-----------------|
| Speech Commands | `_load_waveform()` | As stored | Handled by `_load_waveform` |
| VoxCeleb1 | `_load_waveform()` | As stored | Handled by `_load_waveform` |
| IEMOCAP | `_load_waveform()` | As stored | Handled by `_load_waveform` |
| Fluent SC | `torchaudio.load()` + resample | Resampled to 16kHz | `.mean(dim=0, keepdim=True)` |

**⚠️ Inconsistency:** FluentSpeechCommands uses `torchaudio.load()` directly (not `_load_waveform`), does explicit stereo→mono conversion, and resamples. The other three use `_load_waveform`. This means FluentSC handles differently: different loading path, explicit resampling, explicit mono conversion.

### Model Loading: `upstream/model/load_upstream.py`

```python
def load_upstream_model(upstream_model_type, checkpoint_path, device):
    if model_key.startswith("wavlm"):
        model = load_wavlm(checkpoint_path=ckpt_path, device=device)
        return model
    raise ValueError(f"Unknown upstream_model_type='{upstream_model_type}'")
```

Currently **only supports WavLM**. The "Extensibility" claim in README mentions HuBERT, wav2vec2 but none are implemented.

### WavLM Loader: `upstream/model/wavlm/wavlm_loader.py`

```python
def load_wavlm(checkpoint_path, device):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg_model = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg_model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model
```

- Loads checkpoint on CPU first (safe)
- Extracts config from checkpoint's `"cfg"` key and weights from `"model"` key
- ~316M parameters for WavLM-Large (24 transformer layers, 1024-dim, 16 heads)

### Embedding I/O: `upstream/utils/embedding_io.py`

**`process_and_write_data()`** — the bridge function:

```python
def process_and_write_data(data, root_emb_path, upstream_model_type,
                           frame_pooling_type, dataset_name, limit=None,
                           append=False, frame_pooling_param=None):
    frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
    dir_emb_path = os.path.join(root_emb_path, upstream_model_type,
                                frame_pool_id, dataset_name)
    csv_path = os.path.join(dir_emb_path,
        f"{dataset_name}_{upstream_model_type}_{frame_pool_id}.csv")

    for index, (embedding, label, wavpath) in enumerate(data):
        if limit is not None and index >= limit:
            break
        embpath = os.path.join(dir_emb_path, wavpath)
        embpath = embpath.replace(".wav",
            f"_{upstream_model_type}_{frame_pool_id}.pt")
        torch.save(embedding, embpath)
        csv_file.write(f"{wavpath},{label},{get_file_size(embpath)}\n")
```

**Output structure on disk:**
```
{root_emb_path}/
  {upstream_model_type}/                  # e.g., wavlm_large
    {frame_pool_id}/                      # e.g., mix0d5
      {dataset_name}/                     # e.g., speechcommand
        train/
          sample1.pt
          ...
        val/
          ...
        test/
          ...
        {dataset_name}_{model}_{pool}.csv  # CSV with wavpath,label,file_size
```

### Utility: `pooling_id.py`

**`make_pooling_id()`** generates folder-safe strings:
```python
("mean", None)  → "mean"
("lnp", 16)     → "lnp16"
("mix", 0.5)    → "mix0d5"   # '.' replaced with 'd'!
("mix", -0.5)   → "mixm0d5"  # negative: 'm' prefix
```

**`make_pooling_name()`** generates human-readable strings:
```python
("mean", None)  → "mean"
("mix", 0.5)    → "mix 0.5"
```

---

## Stage 2: Downstream Training & Evaluation

### Entry Point: `downstream/main.py`

**Flow:**

```
CLI args (--task_type, --config, --device_index)
  ↓
load_config() → cfg dict
  ↓
parse_transformer_layers(selected_transformer_layers, upstream_model_type) → List[int]
  ↓
set_device() → torch.device
  ↓
LoadEmbedding(...) → loader.load_embedding(task_type, subset_percentage) → train_data, val_data, test_data
  ↓
DownstreamMultiTaskModel(...) → model.to(device)
  ↓
MultiTasksModelTrainer(...) → trainer.train()
  ↓
MultiTasksModelEvaluator(checkpoint_tag="opt").run()
MultiTasksModelEvaluator(checkpoint_tag="best").run()
MultiTasksModelEvaluator(checkpoint_tag="epoch").run()
```

### Embedding Loading: `downstream/dataset/load_embedding.py`

**`LoadEmbedding.__init__()`** establishes path mappings:

```python
frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
self.emb_speechcommand_path = os.path.join(
    root_emb_path, upstream_model_type, frame_pool_id, "speechcommand")
# ... same for voxceleb, iemocap, fluentspeechcommand
```

**`load_embedding(task_type, subset_percentage)`:**

```python
# task_type = "ks_si_er" → ["speechcommand", "voxceleb", "iemocap"]
dataset_id_array = self._build_dataset_array_from_task_type(task_type)

# Build one-hot index patterns for multi-task label mapping
# e.g., ["speechcommand", "voxceleb"] → {"speechcommand": "10", "voxceleb": "01"}
pattern_len = len(dataset_id_array)
for i, ds in enumerate(dataset_id_array):
    bits = ["0"] * pattern_len
    bits[i] = "1"
    self.index_pattern_map[ds] = "".join(bits)

# Load each dataset separately
for ds in dataset_id_array:
    train, val, test = load_dataset_dict[ds]()

# Combine across datasets
training_data = CombinedDataset(training_data_array)

# Optionally take percentage subset
if subset_percentage is not None:
    training_data = PercentageSubset(training_data, subset_percentage)
```

### Dataset Wrappers: `downstream/dataset/preprocess_embedding.py`

Each wrapper inherits from `torchaudio.datasets.*` and overrides `__getitem__()` to **load precomputed .pt embeddings from disk** instead of raw audio.

**Common pattern in `get_metadata()`:**

```python
def get_metadata(self, n: int):
    metadata = super().get_metadata(n)
    wav_relpath = metadata[0]

    # Construct embedding path from wav path
    emb_relpath = wav_relpath.replace(".wav",
        f"_{self.upstream_model_type}_{self.frame_pool_id}.pt")
    emb_path = os.path.join(self.root_embedding, emb_relpath)
    embedding = torch.load(emb_path, map_location=self.device)

    # Select subset of transformer layers
    if self.transformer_layer_array is not None:
        embedding = embedding[self.transformer_layer_array, :]  # [selected_L, D]

    # Map label using index_pattern for multi-task
    label_index = self.label_mapping.get(label, default)
    index_tuple = self._build_index_tuple(label_index)  # "10" → (label_index, -1)
    return embedding, torch.tensor(index_tuple, dtype=torch.long)
```

**`_build_index_tuple(label_index)` transforms labels for multi-task:**
```python
# For index_pattern = "10" (task 0 = ks, task 1 = si):
# Label for ks → (ks_index, -1)
# Label for si → (-1, si_index)
# "-1" is ignore_index → masked out in loss
def _build_index_tuple(self, label_index):
    return tuple(label_index if c == "1" else -1 for c in self.index_pattern)
```

### Dataset Split Details

| Dataset | Train | Val | Test | Split Logic |
|---------|-------|-----|------|-------------|
| Speech Commands | `subset="training"` | `subset="validation"` | `subset="testing"` | torchaudio built-in |
| VoxCeleb1 | `subset='train'` | `subset='dev'` | `subset='test'` | torchaudio + `iden_split.txt` |
| IEMOCAP | indices[4::10], [9::10] excluded | indices[4::10] | indices[9::10] | Custom 4:10 / 9:10 split |
| FluentSC | `subset="train"` | `subset="valid"` | `subset="test"` | torchaudio built-in |

### Combined Dataset & Subsetting

**`CombinedDataset`** — sequential concatenation:
```python
class CombinedDataset(Dataset):
    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
```

**`PercentageSubset`** — evenly-spaced deterministic indices:
```python
class PercentageSubset(Subset):
    def __init__(self, dataset, percentage):
        total = len(dataset)
        num_samples = max(1, total * percentage // 100)
        step = total / num_samples
        indices = [int(i * step) for i in range(num_samples)]  # evenly spaced!
```

### Custom DataLoader: `downstream/dataset/custom_emb_dataloader.py`

```python
class CustomEmbDataLoader(DataLoader):
    def defined_collate(self, batch):
        sequences, labels = zip(*batch)
        input_seq = torch.stack(sequences)   # [B, L, D]
        labels = torch.stack(labels).long()   # [B, num_tasks]
        return input_seq, labels
```

---

## Model Architectures

### Upstream: WavLM-Large

Defined in `upstream/model/wavlm/WavLM.py` (316M params).

```
Input: raw audio [B, T]
  ↓
7-layer CNN feature extractor:
  Conv1D(1→512, stride=5, kernel=10)
  Conv1D(512→512, stride=2, kernel=3) × 6
  ↓
Projection: Linear(512 → 1024) if needed
  ↓
GELU activation
  ↓
24× TransformerEncoderLayer (1024-dim, 16 heads, FFN 4096)
  ├── Self-attention with ConvolutionalRelativePositionEncoding (128 filters, 16 groups)
  ├── GatedRelativePositionBias (320 buckets, max_dist=1280)
  └── FFN: Linear(1024→4096) → GELU → Linear(4096→1024)
  ↓
Output: [T, 1024] per layer + all layer representations
```

**`extract_features()` API** used in upstream:
```python
_, layer_reps = model.extract_features(
    waveform,
    output_layer=model.cfg.encoder_layers,  # all layers
    ret_layer_results=True
)[0]
# layer_reps: list of (layer_output, attention_weights) tuples
```

### Downstream: `DownstreamMultiTaskModel`

**Architecture:**

```
Input: [B, L, D]  (L = #selected layers, D = 1024 for large, 768 for base)
  │
  ├── Linear(D → 512)  # projector_layer
  ├── [layer pooling across L dim]   ← 10 strategies
  ├── Dropout(0.4)
  │
  ├── Linear(512 → 2000)  # hidden_layer
  ├── Dropout(0.6)
  │
  ├── Linear(2000 → n_classes_0)   ──→ logits_0
  ├── Linear(2000 → n_classes_1)   ──→ logits_1
  ├── ...
  └── Linear(2000 → n_classes_N-1) ──→ logits_N-1

Output: MultiClassifierOutput(logits=(...), prediction=(...))
```

**`_input_dim_from_upstream()`:**
```python
def _input_dim_from_upstream(self, upstream_model_type):
    variation = upstream_model_type.split("_")[-1]  # "base" or "large"
    return 768 if variation == "base" else 1024
```

**`_output_dims_from_task_type()`** determines per-task class counts from label mappings:
```python
ks → 12, si → 1251, er → 4, ic → 31
```

**`forward()`:**
```python
def forward(self, input_seq):
    embedding_shared = self.projector_layer(input_seq)          # [B, L, 512]
    embedding_shared = self.pooling(embedding_shared, dim=1)    # [B, 512]
    embedding_shared = self.dropout_shared1(embedding_shared)

    embedding_shared = self.hidden_layer(embedding_shared)      # [B, 2000]
    embedding_shared = self.dropout_shared2(embedding_shared)

    logits_list = [head(embedding_shared) for head in self.classifiers]
    return MultiClassifierOutput(logits=tuple(logits_list), prediction=...)
```

**Note:** Layer pooling is applied **after** the first projection (on `[B, L, 512]`) NOT on the original `[B, L, 1024]` input. This means the pooling operates on a projected representation.

---

## Pooling Strategies

### Frame Pooling (time dim) — 6 strategies

| Strategy | Param | Formula | Learnable? |
|----------|-------|---------|-----------|
| `mean` | None | `x.mean(dim=d)` | No |
| `max` | None | `x.max(dim=d).values` | No |
| `mix` | `ratio ∈ [0,1]` | `ratio·max + (1-ratio)·mean` | No |
| `lnp` | `p (int)` | `(Σ|x|^p / n)^(1/p)` | No |
| `smp` | `λ (float)` | `Σ softmax(λ·x) · x` | No |
| `lse` | `r (float)` | `(1/r)·log(Σ exp(r·x) / n)` | No |

### Layer Pooling (layer dim) — 10 strategies

Adds 4 learnable strategies to the base 6:

#### Weighted Pooling
```python
def weighted_pooling(self, x, d):
    w = F.softmax(self.position_weights, dim=0)   # [L] learnable
    out = torch.sum(x * w.view(1, -1, 1), dim=d)   # [B, D]
```
**Learnable:** `position_weights` (shape `[L]`). Softmax-normalized per-layer weights.

#### Gated Pooling
```python
def gated_pooling(self, x, d):
    avg = x.mean(dim=d)                              # [B, D]
    mx = x.max(dim=d).values                          # [B, D]
    s = torch.sum(x * self.position_weights.view(1, -1, 1), dim=d)  # [B, D]
    g = torch.sigmoid(s)                              # [B, D] vector gate
    out = g * mx + (1.0 - g) * avg
```
**Learnable:** `position_weights` (shape `[L]`). Output is a learned interpolation between max and mean.

#### Auto Pooling
```python
def auto_pooling(self, x, d):
    exp_alpha_x = torch.exp(self.alpha * x)
    out = torch.sum(x * exp_alpha_x, dim=d) / (torch.sum(exp_alpha_x, dim=d) + 1e-12)
```
**Learnable:** `alpha` (scalar). Generalized softmax-like pooling with learned temperature.

#### Self-Attentive Pooling (SAP)
```python
def self_attentive_pooling(self, x, d):
    h = torch.tanh(self.sap_W(x))         # [B, L, D]
    scores = self.sap_v(h).squeeze(-1)     # [B, L]
    a = torch.softmax(scores, dim=d)       # [B, L]
    out = torch.sum(x * a.unsqueeze(-1), dim=d)  # [B, D]
```
**Learnable:** `sap_W` (Linear D→D), `sap_v` (Linear D→1). Standard self-attentive pooling.

### Pooling Instantiation Logic (downstream `main.py:127-131`)

```python
layer_pooling_param = (
    cfg["pooling"]["layer_pooling_param"]    # from config
    if layer_pooling_type not in ["weighted", "gated"]
    else len(transformer_layer_array)        # seq_len = number of selected layers
)
```

**Rationale:** `weighted` and `gated` need `pooling_param` to know `L` (number of layers) for initializing `position_weights`. For `sap`, `pooling_param` must be 512 (the projected embedding dim after `projector_layer`).

### Introspection via `_last_aux`

The downstream pooling stores auxiliary outputs after each forward pass:

```python
pooling = model.pooling  # inside DownstreamMultiTaskModel
pooling.get_last_aux()
# Returns for weighted: {"position_weights": ..., "weighted_weights": ...}
# Returns for gated:    {"position_weights": ..., "gate": ...}
# Returns for auto:     {"alpha": ...}
# Returns for sap:      {"sap_weights": ..., "attn_scores": ...}
```

---

## Data Flow & Label Handling

### Multi-Task Label Encoding

The `index_pattern` mechanism maps single-task labels to multi-task tuples:

```
Task type:      "ks_si"
Dataset:         ["speechcommand", "voxceleb"]
Index patterns: {"speechcommand": "10", "voxceleb": "01"}

For a Speech Commands sample with label "yes" (index 0):
  index_pattern = "10" → (0, -1)  # -1 means "ignored for this task"

For a VoxCeleb1 sample with speaker "42" (index 41):
  index_pattern = "01" → (-1, 41)
```

### Loss Computation: `masked_ce_loss()`

```python
def masked_ce_loss(logits, labels, loss_fn, ignore_index=-1):
    mask = labels != ignore_index
    labels_masked = labels[mask]
    if labels_masked.numel() == 0:
        return None, 0
    logits_masked = logits[mask, :]
    loss = loss_fn(logits_masked, labels_masked)
    return loss, int(labels_masked.size(0))
```

**Multi-task loss weighting:**

```python
loss_weight = 1.0 / float(self.num_tasks)
```

Each task contributes equally (unweighted average). L1 and L2 regularization are added globally.

### Accuracy Computation: `masked_accuracy()`

```python
def masked_accuracy(pred, labels, ignore_index=-1):
    mask = labels != ignore_index
    correct = int((pred[mask] == labels[mask]).sum().item())
    total = int(labels[mask].numel())
    return correct, total
```

"All" accuracy = `sum(correct_all_tasks) / sum(samples_all_tasks)` — can be dominated by larger tasks.

---

## Checkpoint & Evaluation System

### CheckpointManager (`trainer_utils.py`)

Three checkpoint variants:

| Tag | File Naming | Selection Criterion |
|-----|------------|-------------------|
| `best` | `{task_type}_best.pth` | Highest `accuracy_all` = `sum(correct) / sum(samples)` across all tasks |
| `opt` | `{task_type}_opt.pth` | Highest **average** of per-task accuracies: `mean(accuracy_task[t])` |
| `epochN` | `{task_type}_epochN.pth` | Saved every epoch, rolling window of `saved_checkpoint_count` |

**"best" vs "opt" distinction:**
```python
# BEST: pooled accuracy (favors larger tasks)
if test_accuracy_all > self.best_accuracy_all_threshold:
    save_best()

# OPT: average of per-task accuracies (more balanced)
avg_acc = mean(test_accuracy_task[t] for t in range(num_tasks))
if avg_acc > self.opt_accuracy_avg_threshold:
    save_opt()
```

**Rolling epoch checkpoints:**
```python
def save_epoch_checkpoint(self, epoch):
    path = f"{base}_epoch{epoch}.pth"
    torch.save(model.state_dict(), path)
    if epoch > saved_checkpoint_count:
        old_path = f"{base}_epoch{epoch - saved_checkpoint_count}.pth"
        os.remove(old_path)  # only keep most recent N
```

### Evaluator (`evaluator_model.py`)

**`_latest_run_id()`** auto-discovers the most recent run directory (by mtime):
```python
def _latest_run_id(self, root_dir):
    run_dirs = sorted(os.listdir(root_dir),
        key=lambda d: os.path.getmtime(os.path.join(root_dir, d)),
        reverse=True)
    return run_dirs[0]  # latest
```

**`_resolve_checkpoint_path()`** finds checkpoint by tag substring matching:
```python
def _resolve_checkpoint_path(self, checkpoints_root, checkpoint_tag):
    ckpt_files = [f for f in os.listdir(checkpoints_root) if f.endswith(".pth")]
    matches = [f for f in ckpt_files if tag in f.lower()]  # substring match!
    return os.path.join(checkpoints_root, matches[0])
```

**⚠️ Fragile:** Substring matching means tag="opt" matches both "opt" and "optimizer" (if such a file exists). Tag="epoch" matches all epoch files (picks the first alphabetically/mod-time sorted).

### Evaluation Outputs

```
results/run_{timestamp}/
├── train_{task_type}.txt          # Training metrics (all tasks)
├── train_{task_type}_Accuracy.png  # Accuracy plot
├── train_{task_type}_Loss.png     # Loss plot
├── train_{task_type}_lr.png       # Learning rate plot
├── train_{task}.txt               # Per-task metrics
├── train_{task}_Accuracy.png      # Per-task accuracy plot
├── train_{task}_Loss.png          # Per-task loss plot
├── eval_metrics_{tag}.txt         # Evaluation metrics (acc/loss)
├── eval_predictions_{tag}_{task}.csv  # true_label, pred_label
└── data_count.txt                 # Debug: per-batch valid counts
```

---

## Pipeline Contract (Upstream → Downstream)

The contract between stages is the **file naming convention**:

```
Embedding path on disk:
  {root_emb_path}/{model_type}/{frame_pool_id}/{dataset_name}/{wav_relpath}_{model_type}_{frame_pool_id}.pt

CSV path on disk:
  {root_emb_path}/{model_type}/{frame_pool_id}/{dataset_name}/{dataset_name}_{model_type}_{frame_pool_id}.csv
```

**Downstream reconstructs these paths:**
```python
emb_relpath = wav_relpath.replace(".wav",
    f"_{self.upstream_model_type}_{self.frame_pool_id}.pt")
emb_path = os.path.join(self.root_embedding, emb_relpath)
```

**Critical:** The `root_emb_path` and `frame_pool_id` must match exactly between stages. The default configs have `~/embedding2` (upstream) vs `~/embedding` (downstream) — **this will break the pipeline**.

---

## Config Schema Reference

### Upstream Config (`extract_embedding.yml`)

```yaml
log_level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
device:
  type: cuda                       # cuda, cpu
  index: 0                         # GPU index
paths:
  root_data_path: ~/voice_dataset  # Raw audio datasets
  root_emb_path: ~/embedding2      # Output embeddings (MISMATCH with downstream!)
  upstream_checkpoint_path: ~/pretrained_checkpoints/SSL/WavLM-Large.pt
upstream:
  model_type: wavlm_large          # Only wavlm_large implemented
pooling:
  frame_pooling_type: mix          # mean, max, mix, lnp, smp, lse
  frame_pooling_param: 0.5         # strategy-specific parameter
dataset:
  limit: 1                         # ⚠️ DANGER: defaults to 1 sample!
```

### Downstream Config (`build_model.yml`)

```yaml
log_level: INFO
device:
  type: cuda
  index: 1                         # ⚠️ Different from upstream (0 vs 1)
paths:
  root_data_path: ~/voice_dataset
  root_emb_path: ~/embedding       # ⚠️ MISMATCH with upstream (~/embedding2)
  results_root: results
  checkpoints_root: checkpoints
upstream:
  model_type: wavlm_large
  selected_transformer_layers: 6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13
dataset:
  subset_percentage: 10            # Uses 10% of data by default
  ignore_index: -1
pooling:
  frame_pooling_type: mean
  frame_pooling_param: null
  layer_pooling_type: mix
  layer_pooling_param: 0.5
model:
  embedding_dim_shared1: 512
  embedding_dim_shared2: 2000
  dropout_prob_shared1: 0.4
  dropout_prob_shared2: 0.6
training:
  num_epochs: 5                    # ⚠️ Very low (real: 100+)
  batch_size: 2048
  learning_rate: 0.0025
  weight_decay: 0.00000005
  saved_checkpoint_count: 1
  shuffle_train: true
  shuffle_val: true
  pin_memory: false
  drop_last_train: true
  drop_last_val: false
  num_workers: 0
  patience: 1
  factor: 0.5
  l1_lambda: 0.0000001
  l2_lambda: 0.00001
evaluation:
  batch_size: 1                    # ⚠️ Extremely slow eval
  shuffle: false
  pin_memory: false
  drop_last: false
  num_workers: 0
```

---

## Dependencies & Environment

**Conda env:** `opencv` (Python 3.9)

**Core ML stack:**
- `torch==2.7.1`, `torchaudio==2.7.1`, `torchvision==0.22.1`
- CUDA 12.x (`nvidia-*` pip packages)

**Audio:**
- `librosa==0.10.2`, `soundfile`, `soxr`
- `speechbrain==1.0.3` (VoxCeleb/IEMOCAP loading helpers)
- `transformers==4.43.3`, `huggingface-hub==0.29.2`

**Data/ML:**
- `scikit-learn`, `scipy`, `pandas`, `numpy`
- `datasets==3.3.2` (HuggingFace)
- `imbalanced-learn`

**Config/hparams:**
- `hydra-core`, `omegaconf`, `pyyaml`
- `optuna`

**Plots:**
- `matplotlib`, `seaborn`

**Misc:**
- `tqdm`

---

## Complete Problem Inventory

### 🔴 Critical (Pipeline Will Fail)

| # | Problem | Where | Detail |
|---|---------|-------|--------|
| 1 | **`dataset.limit: 1` default** | `extract_embedding.yml:86` | Processes only 1 sample per train/val/test split. Must be `null` for real runs. |
| 2 | **Upstream vs downstream `root_emb_path` mismatch** | Upstream: `~/embedding2`, Downstream: `~/embedding` | Upstream writes to `~/embedding2`, downstream reads from `~/embedding`. Downstream will not find any embeddings. |
| 3 | **No validation that upstream has been run** | `downstream/main.py` | Downstream crashes with opaque `FileNotFoundError` if embeddings don't exist. No early check. |
| 4 | **`device.index: 1` in downstream** | `build_model.yml:27` | Single-GPU machines (index 0) will silently fall back to CPU or crash depending on `set_device()` behavior. |

### 🟠 Configuration Issues

| # | Problem | Detail |
|---|---------|--------|
| 5 | `training.num_epochs: 5` | Way too low for convergence. Real runs need 100+. |
| 6 | `evaluation.batch_size: 1` | Extremely slow eval. No clear reason for batch_size=1. |
| 7 | `dataset.subset_percentage: 10` | Uses only 10% of data by default. Surprising for new users. |
| 8 | `dataset.limit: 1` in upstream | Already critical, but also silently produces a 1-row CSV. |
| 9 | `upstream_checkpoint_path` is hardcoded | Points to `~/pretrained_checkpoints/SSL/WavLM-Large.pt`. README claims auto-download — this contradicts. |

### 🟡 Code Quality / Duplication

| # | Problem | Where | Detail |
|---|---------|-------|--------|
| 10 | **6 utility files duplicated** | `upstream/utils/` vs `downstream/utils/` | `constant_mapping.py`, `load_config.py`, `pooling_id.py`, `setup_device.py`, `setup_logging.py` are byte-identical copies. Any fix must be applied twice. |
| 11 | **Pooling code duplicated with drift** | `upstream/pooling/pooling.py` vs `downstream/pooling/pooling.py` | Upstream has 6 strategies, downstream has 10 (adds weighted/gated/auto/sap). Upstream's has no `_last_aux`. |
| 12 | **`save_debug_counts` writes every batch** | `trainer_model.py:236-239` | Writes to `data_count.txt` at every batch step. For 100 epochs × 1000s of batches, produces massive files. No disable flag. |
| 13 | **`setup_logging` accumulates logs** | `setup_logging.py` | Creates `logs/run_{timestamp}.log` every run. Grows unbounded. |
| 14 | **No unit tests** | Entire project | Zero tests. No CI pipeline. |

### 🔵 Potential Bugs / Fragile Code

| # | Problem | Where | Detail |
|---|---------|-------|--------|
| 15 | **`frame_pooling_param` type mismatch** | `BuildEmbedding.__init__:29` | Declared as `Optional[int]` but used as `Optional[Union[int, float]]` in `Pooling.__init__`. Float params silently truncated. |
| 16 | **IEMOCAP "exc" → "hap" hardcoded** | `extract_embedding.py:313`, `preprocess_embedding.py:257` | Excited mapped to happy without documentation or configuration. |
| 17 | **IEMOCAP path parsing fragile** | `extract_embedding.py:275` | `split("/")[3].split("_")[0]` assumes specific directory layout. |
| 18 | **VoxCeleb1 label range hardcoded** | `constant_mapping.py:37` | `{str(i): i-1 for i in range(1, 1252)}` assumes exactly 1251 speakers. |
| 19 | **FluentSC uses different loading path** | `extract_embedding.py:366` | Uses `torchaudio.load()` + `.mean()` + resample vs `_load_waveform` for others. |
| 20 | **`"all"` transformer layers hardcoded to 12/24** | `parse_transformer_layers.py:31` | Adding a new model with different layer count requires code change. |
| 21 | **`make_pooling_id` replaces `.` with `d`** | `pooling_id.py:37` | `0.5` → `0d5`. Can cause naming conflicts if param contains `d`. |
| 22 | **`PercentageSubset` uses evenly-spaced indices** | `preprocess_embedding.py:380-381` | Not random — could introduce bias. No `random_seed` param. |
| 23 | **Evaluator checkpoint substring matching** | `evaluator_model.py:177` | `tag="opt"` could match "optimizer_save.pth" if it exists. |
| 24 | **Evaluator auto-discovers latest run** | `evaluator_model.py:151-167` | Uses `mtime` — fragile across filesystems (e.g., git clone resets timestamps). |
| 25 | **`CombinedDataset` concatenates sequentially** | `preprocess_embedding.py:358` | Batches from DataLoader may be single-dataset (no interleaving). |
| 26 | **`maybe_trim_by_filesize` can hit non-wav files** | `extract_embedding.py` | Metadata may contain non-wav paths; `.replace(".wav", ...)` will not match. |

### 🟣 Architectural Concerns

| # | Problem | Detail |
|---|---------|--------|
| 27 | **Multi-task loss is unweighted** | `1/num_tasks` per task. Imbalanced tasks have equal influence. |
| 28 | **"best" checkpoint metric favors large tasks** | Uses pooled accuracy. VoxCeleb1 (1251 classes, many samples) dominates IEMOCAP (4 classes, few samples). |
| 29 | **No cross-validation or multiple seeds** | Single train/val/test split. No reproducibility with different seeds. |
| 30 | **Extensibility is aspirational** | README claims support for HuBERT, wav2vec2, but `load_upstream.py` dispatches solely on `"wavlm"` prefix. |
| 31 | **No data augmentation** | Upstream has silence trimming only. Downstream uses pre-computed embeddings — no augmentation possible after extraction. |
| 32 | **FluentSC resampling may mismatch upstream** | Upstream resamples to 16kHz but metadata doesn't record original sample rate. |
| 33 | **Layer pooling operates on projected 512-dim, not original 1024** | `downstream_model.py:105-106` — projector_layer reduces dim before pooling. This means learnable pooling (sap, weighted) works on 512-dim projections, not the original WavLM representations. |
| 34 | **No gradient tracking in upstream** | `torch.no_grad()` in all dataset wrappers. Correct for inference, but means no fine-tuning. |

---

## How to Run (Corrected)

### Prerequisites

```bash
conda env create -f environment.yml
conda activate opencv
```

### Stage 1: Upstream

**First, fix config:**
```yaml
# upstream/configs/extract_embedding.yml
dataset:
  limit: null           # ⬅️ CRITICAL: process all samples, not just 1
paths:
  root_data_path: /path/to/your/datasets
  root_emb_path: /path/to/save/embeddings   # e.g., ~/embedding
  upstream_checkpoint_path: /path/to/WavLM-Large.pt
```

**Run:**
```bash
cd /home/kevin/projects/fyp/wavCSE/wavCSE/upstream
python main.py --dataset_name speechcommand --config configs/extract_embedding.yml --device_index 0
```

### Stage 2: Downstream

**First, fix config:**
```yaml
# downstream/configs/build_model.yml
device:
  index: 0              # ⬅️ Match your GPU (not 1!)
paths:
  root_emb_path: /path/to/save/embeddings   # ⬅️ MUST match upstream's root_emb_path
dataset:
  subset_percentage: 100   # ⬅️ Use all data
training:
  num_epochs: 100          # ⬅️ Real training config
evaluation:
  batch_size: 2048         # ⬅️ Match training batch size for speed
```

**Run:**
```bash
cd /home/kevin/projects/fyp/wavCSE/wavCSE/downstream
python main.py --task_type ks_si_er_ic --config configs/build_model.yml --device_index 0
```

### Quick Test (Limit Mode)

To verify the pipeline works end-to-end quickly:
1. Keep `dataset.limit: 5` in upstream
2. Set `dataset.subset_percentage: 100` and `training.num_epochs: 2` in downstream
3. Use a single task: `--task_type ks`
4. Confirm `.pt` files are created, CSV is written, training runs, evaluation produces metrics
