# Where Each Upgrade Connects to wavCSE

> Four architecture diagrams — one per paper — showing exactly where each method integrates into wavCSE's pipeline.

---

## wavCSE Baseline (Reference)

```
                          wavCSE — CURRENT ARCHITECTURE
                          ═══════════════════════════════

  Raw Audio
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │         Frozen WavLM-Large (24 layers)       │  ← 316M params, frozen
  │         Output: [T, 1024] per layer          │
  └─────────────────────────────────────────────┘
      │  selected layers (hand-picked subset)
      ▼
  ┌─────────────────────────────────────────────┐
  │         Frame Pooling (time dim)             │  ← Not learnable
  │         [T, 1024] → [1, 1024] per layer     │
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ①     Projector: Linear(1024 → 512)        │  ← Trainable
  │         [B, L, 1024] → [B, L, 512]          │
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │         Layer Pooling (layer dim)            │  ← 10 strategies
  │         [B, L, 512] → [B, 512]              │
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ②     Hidden: Linear(512 → 2000)           │  ← Trainable
  │         Dropout(0.4) → ReLU → Dropout(0.6)  │
  │         [B, 512] → [B, 2000]                │
  └─────────────────────────────────────────────┘
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼
  ┌────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Head   │      │  Head    │      │  Head    │      │  Head    │
  │  KS    │      │   SI     │      │   ER     │      │   IC     │
  │ 12 cls │      │ 1251 cls │      │  4 cls   │      │  31 cls  │
  └────────┘      └──────────┘      └──────────┘      └──────────┘
   ③ Content        ③ Speaker       ③ Emotion        ③ Intent

  LOSS = (L_KS + L_SI + L_ER + L_IC) / num_tasks    ← Equal weight
```

**Integration points**: ① Projector | ② Hidden layer | ③ Classifier heads

---

## Upgrade 1: Grad-TAG — Gradient-Based Task Affinity

### Integration Point: Between ② and ③ (reads gradients, no architecture change)

```
                          wavCSE + GRAD-TAG
                          ═════════════════

  ... (WavLM, Frame Pool, Projector, Layer Pool — all unchanged) ...

      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ②     Hidden: Linear(512 → 2000)           │
  │         [B, 512] → [B, 2000]                │
  └─────────────────────────────────────────────┘
      │
      │    ┌──────────────────────────────────────────┐
      │    │  ★ GRAD-TAG HOOK (passive — no model change)  │
      │    │                                              │
      │    │  After loss computation, before backward:    │
      │    │  ∇_KS = autograd.grad(L_KS, hidden_output)   │
      │    │  ∇_SI = autograd.grad(L_SI, hidden_output)   │
      │    │  ∇_ER = autograd.grad(L_ER, hidden_output)   │
      │    │                                              │
      │    │  For each pair (i,j):                         │
      │    │  A_ij ≈ −α · ∇_iᵀ ∇_j                        │
      │    │  (Taylor expansion — no extra training)       │
      │    │                                              │
      │    │  After epoch → output m×m affinity heatmap:  │
      │    │  ┌──────────┬────────┬──────────┐           │
      │    │  │          │Content │ Speaker  │ Emotion   │
      │    │  │ Content  │  1.0   │  +0.6    │  −0.2     │
      │    │  │ Speaker  │ +0.6   │   1.0    │  −0.1     │
      │    │  │ Emotion  │ −0.2   │  −0.1    │   1.0     │
      │    │  └──────────┴────────┴──────────┘           │
      │    └──────────────────────────────────────────────┘
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼
  ┌────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Head   │      │  Head    │      │  Head    │      │  Head    │
  │  KS    │      │   SI     │      │   ER     │      │   IC     │
  └────────┘      └──────────┘      └──────────┘      └──────────┘

  Training loop: UNCHANGED. Forward pass: UNCHANGED. Backward: UNCHANGED.
  Grad-TAG is a passive observer — reads gradients, doesn't modify them.
```

**What changes**: Nothing in the architecture. A gradient hook is added in the training loop after `loss.backward()`. The model never knows Grad-TAG exists.

**What it produces**: An m×m affinity matrix every epoch. Positive = tasks reinforce. Negative = tasks compete. Used to inform all subsequent architecture and training decisions.

**Key variable path**: `trainer_model.py` → after `loss_i.backward(retain_graph=True)` → `torch.autograd.grad(loss_i, hidden_output)` → compute A_ij → log to metrics.

---

## Upgrade 2: MTI-Net — Cross-Task Attention

### Integration Point: Between ② and ③ (new module inserted)

```
                          wavCSE + MTI-NET
                          ═════════════════

  ... (WavLM, Frame Pool, Projector, Layer Pool — all unchanged) ...

      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ②     Hidden: Linear(512 → 2000)           │  ← UNCHANGED
  │         [B, 512] → [B, 2000]                │
  └─────────────────────────────────────────────┘
      │
      │    h = [B, 2000]  (same hidden output for all tasks — cloned)
      │
      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ★ CROSS-TASK ATTENTION (NEW MODULE)                        │
  │                                                             │
  │  For each task i:                                           │
  │    Q_i = Linear(h_i)    ──→ Query (2000 → d_k)             │
  │                                                             │
  │  For each task j:                                           │
  │    K_j = Linear(h_j)    ──→ Key   (2000 → d_k)             │
  │    V_j = Linear(h_j)    ──→ Value (2000 → d_k)             │
  │                                                             │
  │  Cross-attention (each task attends to ALL tasks):          │
  │    α_ij = softmax(Q_iᵀ · K_j / √d_k)                       │
  │    h'_i = Σ_j α_ij · V_j                                   │
  │                                                             │
  │                          ┌───┐                              │
  │  BEFORE:  h ──────────→ │KS │  (same input, cloned 3 ways) │
  │            ──────────→ │SI │                                │
  │            ──────────→ │ER │                                │
  │                          └───┘                              │
  │                                                             │
  │                          ┌───┐                              │
  │  AFTER:   h → Attn → h'─→│KS │  (each head sees weighted    │
  │                          │SI │   combination of all tasks)  │
  │                          │ER │                              │
  │                          └───┘                              │
  └─────────────────────────────────────────────────────────────┘
      │
      │    h'_KS, h'_SI, h'_ER  (task-aware representations)
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼
  ┌────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Head   │      │  Head    │      │  Head    │      │  Head    │
  │  KS    │      │   SI     │      │   ER     │      │   IC     │
  └────────┘      └──────────┘      └──────────┘      └──────────┘

  Attention weights α_ij are learned, per-input, and visualize as a task
  interaction graph. Thick edges = strong attention between those tasks.
```

**What changes**: One new module (`CrossTaskAttention`) inserted between `self.hidden` and `self.classifiers`. Each task gets Q, K, V projection layers. The hidden layer itself is unchanged.

**What it produces**: Per-input task interaction. An interpretable attention graph showing which tasks the model learned to listen to.

**Key file change**: `downstream/model/downstream_model.py` — insert `CrossTaskAttention` in `forward()` between hidden output and classifier calls.

---

## Upgrade 3: Recon — Layer-Wise Gradient Conflict

### Integration Point: Analyzes ① and ② (conditionally duplicates ②)

```
                          wavCSE + RECON
                          ══════════════

  ... (WavLM — frozen, excluded from analysis) ...

      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ①     Projector: Linear(1024 → 512)        │  ← ★ Analyzed
  │                                              │     S-Conflict score
  └─────────────────────────────────────────────┘     computed here
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │         Layer Pooling                        │  ← ★ Analyzed
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │   ②     Hidden: Linear(512 → 2000)           │  ← ★ Analyzed
  │                                              │     HIGH conflict → SPLIT
  └─────────────────────────────────────────────┘
      │
      │    ┌──────────────────────────────────────────┐
      │    │  ★ RECON ANALYSIS (after warmup)          │
      │    │                                           │
      │    │  Per layer l, for each task pair (i,j):   │
      │    │  S_ijˡ = 1 − |cos(g_iˡ, g_jˡ)|            │
      │    │                                           │
      │    │  ┌──────────┬─────────┬────────┬───────┐ │
      │    │  │Projector │LayerPool│ Hidden │ Heads │ │
      │    │  │  GREEN   │  GREEN  │  RED   │ GREEN │ │
      │    │  │  share   │  share  │ SPLIT  │  n/a  │ │
      │    │  └──────────┴─────────┴────────┴───────┘ │
      │    └──────────────────────────────────────────┘
      │
      │    IF hidden flagged RED → duplicate:
      │
      │    ┌─────────────────┐
      │    │ Hidden_KS (2000) │ → Head_KS
      │    ├─────────────────┤
      │    │ Hidden_SI (2000) │ → Head_SI
      │    ├─────────────────┤
      │    │ Hidden_ER (2000) │ → Head_ER
      │    └─────────────────┘
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼
  ┌────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Head   │      │  Head    │      │  Head    │      │  Head    │
  │  KS    │      │   SI     │      │   ER     │      │   IC     │
  └────────┘      └──────────┘      └──────────┘      └──────────┘

  GREEN layers stay shared. RED layers get per-task copies.
  WavLM excluded — frozen parameters, no gradient conflict to analyze.
```

**What changes**: Gradient hooks added to every trainable `nn.Linear`. After warmup (e.g., 5 epochs), S-Conflict computed per layer. High-conflict layers duplicated into per-task copies mid-training. Low-conflict layers unchanged.

**What it produces**: A layer conflict heatmap showing exactly where tasks fight. Surgical architecture adaptation — duplicate only where needed.

**Key file changes**:
- `downstream/model/downstream_model.py` — gradient hooks on trainable layers, conditional duplication logic
- `downstream/trainer/trainer_model.py` — warmup phase, S-Conflict computation, mid-training architecture switch

---

## Upgrade 4: Kanakis — MTRL Reparameterization

### Integration Point: At ③ (modifies classifier head weights)

```
                          wavCSE + KANAKIS (MTRL)
                          ═══════════════════════

  ... (WavLM, Frame Pool, Projector, Layer Pool, Hidden) ...
  ... (Optionally: Cross-Task Attention from MTI-Net) ...

      │
      │    h = [B, 2000]  (or h' from cross-task attention)
      │
      ├──────────────────┬──────────────────┬──────────────────┐
      ▼                  ▼                  ▼                  ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  ★ MTRL ON CLASSIFIER HEADS                                  │
  │                                                              │
  │  BEFORE (current wavCSE):                                    │
  │    logits_i = W_i @ h + b_i                                  │
  │    Each W_i is independent. No coupling between heads.       │
  │                                                              │
  │  AFTER (Kanakis reparameterization):                         │
  │                                                              │
  │    Ω = m×m learnable correlation matrix                      │
  │        ┌──────────────────┐                                  │
  │        │ Ω = [1.0  0.6  0.1]│  ← Ω_KS,SI = 0.6              │
  │        │     [0.6  1.0  0.3]│    (KS and SI share strength)  │
  │        │     [0.1  0.3  1.0]│  ← Ω_KS,ER = 0.1              │
  │        └──────────────────┘     (ER is independent)          │
  │                                                              │
  │    W_eff_i = Σ_j Ω_ij · W_base_j                             │
  │    logits_i = W_eff_i @ h + b_i                              │
  │                                                              │
  │    Standard backprop over B and Ω — no alternating opt.      │
  └──────────────────────────────────────────────────────────────┘
      │                  │                  │
      ▼                  ▼                  ▼
  ┌────────┐      ┌──────────┐      ┌──────────┐
  │ Head   │      │  Head    │      │  Head    │       ...
  │  KS    │      │   SI     │      │   ER     │
  │ W_eff  │      │  W_eff   │      │  W_eff   │
  └────────┘      └──────────┘      └──────────┘

  Ω is initialized as identity (no coupling) and learned during training.
  After training, Ω IS the interpretable task correlation structure.
```

**What changes**: One new parameter added — a learnable m×m matrix Ω in the classifier head block. The forward pass is modified to compute effective weights as correlation-weighted combinations. No change to the hidden layer, attention, or any other component.

**What it produces**: A directly interpretable task correlation matrix. Ω_ij = learned correlation between tasks i and j.

**Key file change**: `downstream/model/downstream_model.py` — add `self.task_corr` parameter, modify classifier `forward()` to use `W_effective`.

---

## Combined: All Four Upgrades on wavCSE

```
                    wavCSE + ALL FOUR TASK RELATION UPGRADES
                    ══════════════════════════════════════════

  Raw Audio → [Frozen WavLM] → [Frame Pool] → [Projector ①]
                                                     │
                                                     │  ← Recon analyzes ①
                                                     ▼
                                              [Layer Pooling]
                                                     │
                                                     │  ← Recon analyzes here
                                                     ▼
                                    ┌─────────────────────────────────┐
                                    │ ② Hidden layer                  │
                                    │                                 │
                                    │ IF Recon flags RED → split:     │
                                    │   Hidden_KS │ Hidden_SI │ Hidden_ER │
                                    └─────────────────────────────────┘
                                                     │
                                                     │  ← Grad-TAG reads gradients here
                                                     │
                                    ┌─────────────────────────────────┐
                                    │ ★ Cross-Task Attention (MTI-Net)│
                                    │   h'_i = Σ_j α_ij · V_j        │
                                    └─────────────────────────────────┘
                                                     │
                              ┌──────────────────────┼──────────────────────┐
                              ▼                      ▼                      ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │ ★ MTRL on Heads (Kanakis)                               │
                    │   W_eff_i = Σ_j Ω_ij · W_base_j                        │
                    │   ┌──────────┐   ┌──────────┐   ┌──────────┐           │
                    │   │ Head_KS  │   │ Head_SI  │   │ Head_ER  │   ...     │
                    │   └──────────┘   └──────────┘   └──────────┘           │
                    └─────────────────────────────────────────────────────────┘

  Integration summary:
  ┌────────────┬──────────────────────────┬─────────────────────┬────────────┐
  │  Upgrade   │  Integration Point       │  What Changes        │  Type      │
  ├────────────┼──────────────────────────┼─────────────────────┼────────────┤
  │ Grad-TAG   │  Gradient flow at ②      │  Nothing (passive)   │ Diagnostic │
  │ MTI-Net    │  Between ② and ③         │  +1 attention module │ New module │
  │ Recon      │  All trainable (①②③)     │  ±duplicate layers   │ Conditional│
  │ Kanakis    │  Classifier heads (③)    │  +1 parameter matrix │ New param  │
  └────────────┴──────────────────────────┴─────────────────────┴────────────┘
```
