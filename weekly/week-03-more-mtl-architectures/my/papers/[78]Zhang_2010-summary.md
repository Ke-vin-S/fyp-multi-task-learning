# [78] Zhang et al. (2010) — Bayesian Online Learning for Multi-label and Multi-variate Performance Measures

**Venue:** AISTATS 2010 (JMLR W&CP Volume 9)
**Authors:** Xinhua Zhang (U. Alberta), Thore Graepel & Ralf Herbrich (Microsoft Research Cambridge)

---

## Problem Setting

Multi-label classification where each example can belong to multiple classes simultaneously. Three practical challenges:

1. **Scale** — large n (data points), D (features), C (classes). Must stay O(nDC).
2. **Multi-variate performance measures** — macro-average Fβ-score and AUC couple labels across data points/classes in a non-decomposable way. Models trained to minimize training error often perform poorly under these measures.
3. **Label correlation** — labels can be highly correlated; many applications use tree-structured ontologies.

## Key Idea: BOMC (Bayesian Online Multi-label Classification)

A **probabilistic linear classifier** learned via Bayesian online inference, coupled with **principled test-time labeling** that maximizes the expected F1-score under the posterior.

### Model (Section 2)

- Each class c has a linear discriminant weight vector w_c ~ N(μ_c, diag(σ²_{c,1},...,σ²_{c,D})).
- For **multi-class**: likelihood modeled via a factor graph similar to **TrueSkill**. The noisy discriminant f_c of the correct class must exceed all others by margin ε.
- For **multi-label**: instead of O(C²) pairwise comparisons, they assume a **total ordering** with a **global bias b** — relevant classes score above b, irrelevant below. Loosely couples classes through this shared bias.

### Online Learning via Gaussian Density Filtering (Section 3)

- Uses **assumed density filtering (GDF)**: at each step, absorb likelihood of one training example, compute posterior, then approximate it as a diagonal Gaussian (minimizing KL divergence).
- Inference performed via **Expectation Propagation (EP)** on the factor graph.
- Key efficiency: EP only needs to iterate over the bias b and difference nodes {α_c, d_c, β_c}, not all weights — reducing per-iteration cost from O(DC) to O(C).

### Test-Time Labeling (Section 4)

Unlike frequentist methods that threshold independently, BOMC labels test data **jointly** by maximizing the **expected F1-score** under the posterior:

```
l* = argmax_{l ∈ {0,1}^n} E_{y~p(y|X_test)} [F1-score(y, l)]
```

- p(y=1 | x) is computed in closed form (Eq. 2) using the Gaussian posterior.
- Since exact optimization is intractable, they reduce the search space from {0,1}^n to a single threshold θ: label class relevant iff p(y=1) > θ.
- Expected F1 is estimated via **Monte Carlo samples** from the posterior (S ≈ 5–10).
- An efficient single-pass algorithm uses "sufficient statistics" α, β, γ to compute ^ExpFs for all candidate thresholds without storing all weight samples.

### Soundness Check (Section 4.2)

They validate that optimizing expected F1 against the model posterior is a good proxy for optimizing true F1 against ground truth — by showing the two curves (expected vs. true F1 as functions of θ) follow similar trends.

## Variants Tested

| Variant | Description |
|---------|-------------|
| **BOMC CMN** | Class mass normalization: simple non-Bayesian threshold tuning that matches class ratios between train and test. |
| **BOMC Sample** | Uses 5 posterior samples + threshold optimization via expected F1. |
| **BOMC IND CMN / IND Sample** | Trains each class **independently** (no shared global bias), with closed-form posterior. Tests both CMN and sample-based thresholding. |

## Experiments (Reuters1-v2, Section 5)

- **Reuters1-v2**: 804k documents, 3 category groups (topics: 103, industries: 354, regions: 366 classes), tf-idf features (~35k features, ~77 non-zero per example).
- **Baselines**: Batch SVM (liblinear), LaSVM, Passive-Aggressive (PA) with online and batch CV.

### Key Results

1. **BOMC CMN achieves highest macro-average F1 among all online learners** most of the time, and is often competitive with or better than batch SVM — despite using only a single pass.
2. BOMC Sample is slightly worse than BOMC CMN but still competitive.
3. **Coupled > Decoupled**: BOMC with shared global bias significantly outperforms independent per-class training on industries and regions groups — the global bias is learned with higher precision because it is updated more often.
4. **Training time linear** in number of training examples. BOMC runs ~2-3x slower than PA/SVM (partly attributed to F# vs. C implementation). LaSVM achieves competitive F1 but is far slower.
5. EP converges quickly — relative change of mean/variance falls below 10⁻³ after ~3 iterations.

## Relevance to FYP

- **Bayesian MTL perspective**: The shared global bias acts as a simple form of **parameter-based multi-task coupling** — all tasks (classes) share a common bias parameter, demonstrating that even loose coupling improves over fully independent training.
- **Multi-variate performance measures**: Directly addresses the problem of optimizing non-decomposable evaluation metrics (macro F1), which is relevant to multi-task evaluation where tasks may have competing metrics.
- **Online learning + Bayesian inference**: The Gaussian density filtering approach is an instance of scalable Bayesian learning that could inform the design of the MTL library's training strategies.
- **Task imbalance**: The class-mass normalization heuristic and cost-sensitive factor adjustment are simple mitigation strategies for label skew — analogous to task-imbalance handling in MTL.

---

## Mapping Zhang 2010 (BOMC) onto the wavCSE Architecture

### Side-by-Side Component Mapping

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    ZHANG 2010 (BOMC)                         │          wavCSE (current)              │
├──────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
│                                                              │                                        │
│  Input: document xᵢ ∈ Rᴰ (tf-idf features)                  │  Input: raw audio waveform             │
│      │                                                       │      │                                 │
│      ▼                                                       │      ▼                                 │
│  ┌──────────────────────┐                                    │  ┌──────────────────────────┐          │
│  │  Feature extraction   │  ← hand-crafted features          │  │  WavLM-Large (frozen)     │  ← learned representations  │
│  │  (tf-idf, static)     │                                    │  │  7-layer CNN + 24 Trans.  │                              │
│  └──────────────────────┘                                    │  │  → 25 × [T, 1024]         │                              │
│      │                                                       │  └──────────────────────────┘          │
│      ▼                                                       │      │                                 │
│  ┌──────────────────────┐                                    │  ┌──────────────────────────┐          │
│  │  Linear discriminant  │                                    │  │  Frame + Layer Pooling    │  ← collapses time & layers  │
│  │  a_c = ⟨w_c, x⟩       │  ← one weight vector per class   │  │  → [512]                  │                              │
│  │  w_c ∈ Rᴰ            │                                    │  └──────────────────────────┘          │
│  └──────────────────────┘                                    │      │                                 │
│      │                                                       │      ▼                                 │
│      ▼                                                       │  ┌──────────────────────────┐          │
│  ┌──────────────────────┐                                    │  │  FC1: 1024 → 512          │  ← shared feature projector  │
│  │  Noisy discriminant   │                                    │  │  (per-layer, shared)      │                              │
│  │  f_c = a_c + 𝒩(0,β²)  │  ← Gaussian noise channel        │  └──────────────────────────┘          │
│  └──────────────────────┘                                    │      │                                 │
│      │                                                       │      ▼                                 │
│      ▼                                                       │  ┌──────────────────────────┐          │
│  ┌──────────────────────┐                                    │  │  FC2: 512 → 2000          │  ← UNIFIED EMBEDDING          │
│  │  Global bias b        │  ← ONE shared scalar              │  │  + Dropout(0.6)           │     (implicit coupling hub)   │
│  │  f_c > b  → relevant  │     couples all classes           │  └──────────────────────────┘          │
│  │  f_c < b  → irrelevant│     updated by EVERY example      │      │                                 │
│  └──────────────────────┘                                    │      ├──────────┬──────────┬──────────┤│
│      │                                                       │      ▼          ▼          ▼          ││
│      ▼                                                       │  ┌──────┐  ┌──────┐  ┌──────┐        ││
│  ┌──────────────────────┐                                    │  │  KS  │  │ SID  │  │  ER  │        ││
│  │  Per-class head       │  ← w_c has mean μ_c, variance σ²_c│  │ 2000 │  │ 2000 │  │ 2000 │        ││
│  │  (Bayesian: Gaussian  │     = full distribution, not just │  │ →12  │  │→1251 │  │ →4   │        ││
│  │   posterior over w_c) │     a point estimate              │  └──────┘  └──────┘  └──────┘        ││
│  └──────────────────────┘                                    │     ↑          ↑          ↑            ││
│                                                              │  Independent task heads                ││
│                                                              │  (point estimates, no uncertainty)     ││
├──────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
│                                                              │                                        │
│  TRAINING: Gaussian Density Filtering (online, one-pass)     │  TRAINING: Mini-batch SGD (multi-epoch)│
│  Expectation Propagation on factor graph                     │  Standard backprop through frozen WavLM │
│  Posterior approximated as diagonal Gaussian after each ex.  │  Point estimates, no posterior          │
│                                                              │                                        │
│  LOSS: implicit in likelihood p(y|w,x)                       │  LOSS: (L_KS + L_SID + L_ER) / 3       │
│  (not a sum of per-task losses)                              │  equal-weight averaging                 │
│                                                              │                                        │
├──────────────────────────────────────────────────────────────┼────────────────────────────────────────┤
│                                                              │                                        │
│  TEST-TIME: maximize expected macro-F1 jointly               │  TEST-TIME: argmax per head            │
│  l* = argmax E_{y~p(w)}[F1(y, l)]                            │  independently                          │
│  Uses Monte Carlo samples from posterior                     │  No joint optimization                  │
│                                                              │                                        │
│  THRESHOLD TUNING:                                           │  THRESHOLD TUNING:                      │
│  - BOMC Sample: find θ* maximizing expected F1               │  - None (argmax directly)               │
│  - BOMC CMN: match class ratios train→test                   │  - No class-imbalance handling          │
│  - Cost-sensitive factor: I(d > ln(e + 1/r))                 │                                          │
│                                                              │                                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### What wavCSE Already Has (Implicit MTL Coupling)

The **FC2 layer** (Linear 512→2000) is wavCSE's implicit equivalent of Zhang 2010's global bias:

```
                    ┌──────────┐
                    │   FC2    │  ← All 3 tasks backprop through here
                    │ 512→2000 │     Gets gradient signal from KS + SID + ER
                    └────┬─────┘     "Updated more often" than any single head
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       KS Head        SID Head       ER Head
```

This matches the Zhang 2010 finding: *"the global bias serves as a hub and is updated more often"* — FC2 is exactly that hub in wavCSE. But the **task heads are still independent** (no shared parameters between them, no uncertainty modeling).

### What wavCSE Is Missing (Zhang 2010 Gaps)

```
                              RAW AUDIO
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │            WavLM-Large (frozen)                │
          └───────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │         Pooling → FC1 → FC2 (shared hub)       │
          └───────────────────────────────────────────────┘
                                  │
                                  │ [2000] unified embedding
                                  │
                    GAP 1: NO DISTRIBUTION OVER WEIGHTS
                    ─────────────────────────────────────
                    wavCSE has point estimates.
                    Zhang 2010 maintains μ and σ² per weight.
                    ↓
          ┌───────────────────────────────────────────────┐
          │  Bayesian FC2: instead of W ∈ R⁵¹²ˣ²⁰⁰⁰       │
          │  maintain W ~ 𝒩(μ, diag(σ²))                   │
          │  → samples from posterior for test-time        │
          └───────────────────────────────────────────────┘
                                  │
                    GAP 2: NO EXPLICIT SHARED BIAS IN HEADS
                    ────────────────────────────────────────
                    wavCSE heads are completely independent.
                    Zhang 2010 couples them via shared bias b.
                    ↓
          ┌───────────────────────────────────────────────┐
          │  Shared Task Bias b ∈ R (or b ∈ R³)            │
          │  KS logit += b_KS    (b_KS = b + δ_KS)         │
          │  SID logit += b_SID  (b_SID = b + δ_SID)       │
          │  ER logit += b_ER    (b_ER = b + δ_ER)         │
          │  b gets gradient from ALL tasks, δ_c per task  │
          └───────────────────────────────────────────────┘
                                  │
                    GAP 3: EQUAL-WEIGHT LOSS AVERAGING
                    ──────────────────────────────────
                    L = (L_KS + L_SID + L_ER) / 3
                    No per-task uncertainty or adaptive weighting.
                    Zhang 2010: cost-sensitive factor based on class ratio.
                    ↓
          ┌───────────────────────────────────────────────┐
          │  Task-imbalance aware loss:                     │
          │  L = Σ_t w_t · L_t                              │
          │  where w_t ∝ 1/√(n_t)  (inverse data size)    │
          │  or learned via task uncertainty (Kendall 2018) │
          └───────────────────────────────────────────────┘
                                  │
                    GAP 4: INDEPENDENT TEST-TIME ARGMAX
                    ──────────────────────────────────
                    Each head: argmax(logits)
                    No joint optimization across tasks.
                    Zhang 2010: maximizes expected F1 jointly.
                    ↓
          ┌───────────────────────────────────────────────┐
          │  Joint test-time optimization:                  │
          │  For tasks that share structure (e.g., speaker  │
          │  + emotion are correlated), label jointly by    │
          │  maximizing expected combined metric under      │
          │  posterior samples                              │
          └───────────────────────────────────────────────┘
```

### Proposed: Zhang-2010-Inspired wavCSE Architecture

```
                              RAW AUDIO
                                  │
                                  ▼
          ┌──────────────────────────────────────────────────────────────┐
          │                   WavLM-Large (frozen)                        │
          │   CNN → 24 × Transformer → 25 × [T, 1024]                    │
          └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────────────────────────┐
          │    Pooling → Layer Select → FC1 (shared, trainable)           │
          └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
          ┌──────────────────────────────────────────────────────────────┐
          │         FC2: Linear(512 → 2000) + Dropout                     │
          │                                                               │
          │  ┌─────────────────────────────────────────────────────────┐ │
          │  │  BAYESIAN VARIANT (Zhang 2010-style):                    │ │
          │  │  W ~ 𝒩(μ_w, diag(σ²_w))    ← distribution, not point    │ │
          │  │  At test time: sample W⁽ˢ⁾ ~ posterior                  │ │
          │  │  → S different predictions → uncertainty estimates      │ │
          │  └─────────────────────────────────────────────────────────┘ │
          └──────────────────────────────────────────────────────────────┘
                                  │
                                  │ [2000] unified embedding
                                  │
                    ┌─────────────┴─────────────┐
                    │  SHARED TASK COUPLING      │  ← ZHANG 2010 INSPIRED
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │ Global bias b ∈ R   │  │  One shared scalar
                    │  │ b ~ 𝒩(μ_b, σ²_b)    │  │  Updated by ALL tasks
                    │  │ (Bayesian if desired)│  │  "Hub" like Zhang 2010
                    │  └────────┬────────────┘  │
                    │           │               │
                    │  ┌────────┼────────────┐  │
                    │  │  Per-task deltas     │  │  Task-specific offsets
                    │  │  δ_KS, δ_SID, δ_ER  │  │  Small perturbations
                    │  │  b_c = b + δ_c       │  │  from shared baseline
                    │  └────────┼────────────┘  │
                    └───────────┼───────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
    ┌───────────┐        ┌───────────┐        ┌───────────┐
    │  KS Head  │        │ SID Head  │        │  ER Head  │
    │ 2000 → 12 │        │2000 → 1251│        │ 2000 → 4  │
    │           │        │           │        │           │
    │ logits += │        │ logits += │        │ logits += │
    │   b_KS    │        │  b_SID    │        │   b_ER    │
    └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
          │                     │                     │
          ▼                     ▼                     ▼
       "yes"              "speaker_42"           "happy"

    ═══════════════════════════════════════════════════════
                        LOSS FUNCTION
    ═══════════════════════════════════════════════════════

    Current (equal-weight):
        L = (L_KS + L_SID + L_ER) / 3

    Zhang 2010-inspired (task-imbalance aware):
        L = w_KS·L_KS + w_SID·L_SID + w_ER·L_ER

    where w_t options:
      (a) Static: w_t ∝ 1/√(n_t)     ← Class Mass Normalization analogue
      (b) Learned: w_t = 1/(2·σ_t²)  ← Kendall 2018, homoscedastic uncertainty
      (c) Cost-sensitive: modify margin ε per task based on data ratio

    ═══════════════════════════════════════════════════════
                    TEST-TIME LABELING
    ═══════════════════════════════════════════════════════

    Current: argmax(logits) independently per head

    Zhang 2010-inspired (if Bayesian):
        For each test utterance:
            1. Sample S weight configurations from posterior
            2. Get S predictions per task
            3. For tasks with shared structure:
               Find labeling maximizing expected combined metric
            4. Return: prediction + uncertainty (variance across samples)
```

### Key Architectural Differences Summarized

| Aspect | Zhang 2010 (BOMC) | wavCSE (current) | wavCSE + Zhang 2010 |
|--------|-------------------|------------------|---------------------|
| **Feature extractor** | Static tf-idf | Frozen WavLM (316M params) | Same (frozen WavLM) |
| **Shared coupling** | Global bias `b` (scalar) | FC2 layer (512→2000) | FC2 + explicit shared bias `b` |
| **Task-specific params** | Weight vectors `w_c` per class | Independent linear heads | Heads + per-task deltas `δ_t` |
| **Parameter type** | Gaussian distributions (μ, σ²) | Point estimates | Point estimates (or Bayesian with MC samples) |
| **Training** | Online GDF + EP (one-pass) | Mini-batch SGD (multi-epoch) | Same SGD (EP expensive for deep nets) |
| **Loss** | Implicit in likelihood | Equal-weight sum | Weighted sum (CMN or uncertainty) |
| **Test-time** | Joint expected-F1 maximization with samples | Independent argmax | Independent argmax (joint optimization complex for heterogeneous tasks) |
| **Uncertainty** | Yes (posterior variance) | No | Yes if Bayesian variant |