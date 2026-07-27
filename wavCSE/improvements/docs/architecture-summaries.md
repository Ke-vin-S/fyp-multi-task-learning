# wavCSE Architectural Improvements — Summaries

---

## 1. Global Bias Coupling (GBC)

**Paper:** Zhang et al. (2010) — BOMC

### Integration into wavCSE
- Forward path unchanged except the bias term
- Global coupling vector added to each task's logits via per-task projection layers

### Approach
- Shared bias hub couples all task heads — receives gradients from every example, converges with lower variance
- `logit_t = W_t · h + b_t + proj_t(b_global + gb_proj(h))`

### Why Use It
- Pools statistical strength across imbalanced tasks. Lightest change. No training loop changes.

### Expected Outcome
- Better accuracy on data-poor tasks. Stable convergence. Projection weights show cross-task dependence.

---

## 2. Task Structure Matrix (TSM)

**Paper:** Ciliberto et al. (2015) — SKMTL

### Integration into wavCSE
- Backbone and hidden layer unchanged. Independent heads replaced by K latent classifiers + structure matrix A

### Approach
- Tasks decompose as sparse convex combinations of shared latent classifiers
- `logit_t = Σ_k A_soft[t,k] · g_k(h) + b_t`  with `λ ||A||_1`

### Why Use It
- Discovers task groupings from data. L1 sparsity blocks negative transfer. Convergent alternating optimization.

### Expected Outcome
- Interpretable task clusters in A. Unrelated tasks isolated. Parameter-efficient when K < T.

---

## 3. Precision Matrix Regularization (PMR)

**Paper:** Gonçalves et al. (2016) — MSSL

### Integration into wavCSE
- Architecture and forward pass unchanged. Precision matrix Ω added to training loss only

### Approach
- Task parameters modeled as Gaussian with sparse precision Ω — encodes conditional independence
- `L += γ(-log|Ω| + Tr(WΩW^T)) + λ|off_diag(Ω)|_1`  with `Ω = L·L^T`

### Why Use It
- Formal probabilistic model. Inference identical to original. Warmup prevents noise-fitting.

### Expected Outcome
- Visualizable dependency graph from Ω. Controlled negative transfer via zero entries. Reproducible across runs.
