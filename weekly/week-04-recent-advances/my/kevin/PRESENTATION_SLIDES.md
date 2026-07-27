# Part 3: Modern Task Relation Learning Advancements

> **Format**: One slide per paper following: Reference → Main Idea → Architecture → Formulation → Adaptation to WavCSE.
> **Rule**: 2-3 bullet points per section. Details in transcript, not on slides.
> **Slides**: 1 intro + 4 papers + 1 mentions = 6 slides total.

---

## Slide 1: Intro — Task Relation Learning for WavCSE

### Part 3 — Modern Task Relation Learning Advancements

**Context**: Zhang & Yang Survey Section 2.4 — learn pairwise task relationships from data. Methods: MTRL (2010), MTGP (2007), sparse extensions, asymmetric relations. All pre-2017. All parameter-level, input-independent.

**The Gap**:
- No gradient-based task affinity — MTRL gives parameter covariance, not gradient conflict
- No feature-level task interaction — survey methods regulate parameters, not feature maps
- No architecture-level diagnostics — tells you THAT tasks relate, not WHERE they conflict
- MTRL requires alternating optimization — doesn't fit standard deep learning loops

**Our Addition**:

| Paper | Key Innovation | In LibMTL? |
|-------|----------------|------------|
| Grad-TAG (KDD 2024) | Gradient-based task affinity discovery | NO |
| MTI-Net (ECCV 2020) | Cross-task attention at feature level | NO |
| Recon (ICLR 2023) | Per-layer gradient conflict → auto architecture | NO |
| Kanakis (ECCV 2020) | MTRL as reparameterization → standard backprop | NO |

---

## Slide 2: Grad-TAG — Gradient-Based Task Affinity

### Grad-TAG — Gradient-Based Task Affinity Matrix

*Reference: Li, Sharma & Zhang, "Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity" (KDD 2024)*

### Main Idea

- Task affinity previously required training all O(n²) task pairs — infeasible beyond a few tasks
- Gradient linearization: train ONE shared model, extract per-task gradients, use Taylor expansion to estimate pairwise affinity
- Result: an m×m affinity matrix from gradient signals — NOT the same as MTRL's parameter covariance V

### Architecture

- Passive diagnostic — hooks into training loop, does not modify the model
- After each epoch: extract per-task gradients at shared representation
- Output: m×m affinity heatmap — green = positive affinity, red = negative affinity

### Formulation

**A_ij = L_j(θ − α·g_i) − L_j(θ)  ≈  −α · g_iᵀ g_j**

**Where**: θ = shared parameters | g_i = gradient of task i's loss (∇L_i) | α = step size

Positive dot product (g_iᵀ g_j > 0) = tasks reinforce. Negative = tasks compete.

Captures optimization dynamics — distinct from MTRL's parameter covariance.

```
Expected heatmap:
          Content   Speaker   Emotion
Content   [ 1.0       0.6      -0.2  ]
Speaker   [ 0.6       1.0      -0.1  ]
Emotion   [-0.2      -0.1       1.0  ]
```

### Adaptation to WavCSE

- Hook into `trainer_model.py` at hidden layer output (2000-dim)
- Extract per-task gradients after loss computation
- Compute pairwise affinity matrix per epoch as training metric
- Zero architecture change. Passive diagnostic only. LibMTL: NOT available.

---

## Slide 3: MTI-Net — Cross-Task Feature Attention

### MTI-Net — Cross-Task Attention for Dynamic Task Interaction

*Reference: Vandenhende, Georgoulis & Van Gool, "MTI-Net: Multi-Scale Multi-Task Interaction Network" (ECCV 2020)*

### Main Idea

- Survey methods learn a fixed task covariance matrix — identical for every input
- MTI-Net moves task interaction from parameter level to feature level
- Cross-task attention: tasks attend to each other dynamically, per input, producing interpretable weights

### Architecture

- Inserted between shared hidden representation and per-task classifier heads
- Each task projects features → Query, Key, Value; attends to every other task's K,V
- Multi-scale: computed at multiple resolutions and fused
- Attention weights are per-input and interpretable as a task interaction graph

### Formulation

**Q_i = W_Qᵢ h_i,   K_j = W_Kⱼ h_j,   V_j = W_Vⱼ h_j**

**α_ij = softmax(Q_iᵀ K_j / √d_k)**

**h'_i = Σ_j α_ij · V_j**

**Where**: h_i = task i's hidden representation | Q,K,V = Query, Key, Value projections | α_ij = attention weight (how much task i listens to task j) | h'_i = updated representation

α_ij ARE the learned pairwise task interaction strengths. Per-input, interpretable as a graph. Survey methods have no analog — they produce fixed matrices identical for all inputs.

### Adaptation to WavCSE

- Insert `CrossTaskAttention` between `self.hidden` and `self.classifiers`
- Each task gets QKV projections with a bottleneck dimension
- Lightweight parameter overhead, minimal code change
- LibMTL: MTAN has spatial attention, not cross-task

---

## Slide 4: Recon — Layer-Wise Gradient Conflict

### Recon — Data-Driven Layer Sharing via Gradient Conflict

*Reference: "Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning" (ICLR 2023)*

### Main Idea

- MTL architectures designed by intuition — shared bottom, task-specific top, but which layers?
- Per-layer gradient conflict scores replace guesswork with data
- Layers where gradients align → share. Layers where they conflict → duplicate per task

### Architecture

- Gradient hooks on every trainable layer during warmup
- S-Conflict = 1 − |cos(∇_i, ∇_j)| computed per layer for all task pairs
- High-conflict layers duplicated mid-training; low-conflict stay shared
- Early layers: low conflict. Later layers: high conflict.

### Formulation

**S_ijˡ = 1 − |cos(g_iˡ, g_jˡ)| = 1 − |g_iˡᵀ g_jˡ| / (||g_iˡ|| · ||g_jˡ||)**

**Where**: l = layer index | g_iˡ = gradient of task i's loss w.r.t. layer l's output | S_ijˡ = conflict score between tasks i,j at layer l

S=0 → gradients perfectly aligned (tasks agree — share). S=1 → gradients orthogonal (tasks fight — split).

Different from survey decomposition (W = L + S): Recon splits layers architecturally based on gradient dynamics, not parameter matrices algebraically.

```
Expected:  [Projector]  [Layer Pool]  [Hidden]  [Heads]
             GREEN        GREEN         RED       GREEN
             share        share         SPLIT     n/a
```

### Adaptation to WavCSE

- Gradient hooks on projector, hidden, classifier heads (WavLM excluded — frozen)
- Expected: hidden layer flagged red → duplicate into per-task copies
- Modest parameter increase. Benefit: zero gradient interference at bottleneck
- LibMTL: no per-layer analysis in any existing method

---

## Slide 5: Kanakis — MTRL Reparameterization

### Kanakis — Lightweight Task Correlation on Classifier Heads

*Reference: Kanakis et al., "Reparameterizing Multi-Task Networks with a Task Relationship Matrix" (ECCV 2020)*

### Main Idea

- MTRL (Zhang & Yeung 2010): learn task covariance V via alternating convex optimization
- Kanakis: same concept as reparameterization W = B · Ω → standard backprop
- Same output (m×m correlation matrix), simpler mechanism, minimal code

### Architecture

- Applied to per-task classifier heads — the final Linear layers
- Base weight B shared; Ω (m×m for m tasks) captures pairwise correlations
- W_i = Σ_j Ω_ij · B_j. Ω initialized as identity, learned via standard SGD/Adam
- No alternating optimization, no extra loss terms, no custom training loop

### Formulation

**BEFORE (MTRL):**  min tr(W V⁻¹ Wᵀ) — alternating optimization over W and V, convex but non-standard

**Where**: W = task parameter matrix (d features × m tasks) | V = m×m task covariance matrix

**AFTER (Kanakis):**  W_eff_i = Σ_j Ω_ij · W_base_j — joint optimization via standard backprop

**Where**: Ω = m×m learnable correlation matrix | W_base = shared base weights | W_eff_i = effective weights for task i

Ω after training IS the learned task structure. Ω_ij large → tasks share strength. Ω_ij ≈ 0 → independent.

### Adaptation to WavCSE

- `self.task_corr = nn.Parameter(torch.eye(num_tasks))` in model init
- `W_effective = self.task_corr @ self.classifiers_weight` in forward
- m×m parameters (one per task pair). Negligible overhead, minimal code
- LibMTL: CrossStitch at feature level, not classifier weight level

---

## Slide 6: Honourable Mentions

### Honourable Mentions — Other Papers Worth Noting

**Available in LibMTL — Use as Training Baselines:**

| Method | Venue | Essence | Role |
|--------|-------|---------|------|
| PCGrad | NeurIPS 2020 | Projects conflicting gradients to orthogonal subspaces | Gradient baseline |
| CAGrad | NeurIPS 2021 | Min-max optimization within trust region | Stronger baseline |
| Nash-MTL | ICML 2022 | Game-theoretic Nash bargaining for fair weighting | Fairness baseline |
| RLW | TMLR 2022 | Random loss weighting from Dirichlet distribution | Sanity check |
| FairGrad | ICML 2024 | Degradation constraint — no task falls below threshold | Worst-task protection |
| ExcessMTL | ICML 2024 | Minimizes excess risk; links gradient conflict to theory | Theoretical framework |

**Not in LibMTL — Future Directions:**

| Method | Venue | Essence | Why Not Selected |
|--------|-------|---------|-----------------|
| MetaLink | ICLR 2022 | Knowledge graph connecting data and task nodes | Major architecture overhaul |
| MHR | NeurIPS 2023 | Multi-head adapter routing for gradient alignment | Adapter-focused; wavCSE fine-tunes |
| Learning to Branch | ICML 2020 | Tree-structured branching for task-specific layers | Overkill for small task count |
| Rotograd | NeurIPS 2022 | Gradient space rotation for shared/task-specific split | Overlaps with Recon |

**Complete Picture**: 4 core additions + 6 LibMTL baselines + 4 future directions.