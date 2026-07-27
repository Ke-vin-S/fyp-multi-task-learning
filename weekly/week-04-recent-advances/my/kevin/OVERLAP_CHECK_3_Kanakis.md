# Overlap Check #3: Kanakis et al. vs MTRL (Zhang & Yeung)

**Paper**: Kanakis et al., "Reparameterizing Multi-Task Networks with a Task Relationship Matrix," ECCV 2020.

**Question**: Is the learned m×m task relationship matrix Ω genuinely different from MTRL's learned m×m task covariance matrix V (Zhang & Yeung 2010 [79], 2014 [80]), and is it already covered by the survey's Section 4.4 on deep MTRL (Long et al. 2017 [95])?

---

## 1. What Each Formulation Does

### MTRL (Zhang & Yeung 2010/2014) — Survey Section 4.2

Learns a parameter matrix **W** (d × m) and a task covariance matrix **V** (m × m) by minimizing:

$$min_{W,b,V \succ 0} \sum_{i} \sum_{j} \ell(y_{ij}, w_i^T x_{ij} + b_i) + \frac{\lambda_1}{2} ||W||_F^2 + \frac{\lambda_2}{2} \text{tr}(W V^{-1} W^T)$$

- **V** is a **regularizer** — it penalizes W via the trace term
- V is positive definite, constrained (e.g., tr(V) = 1)
- The prior: W ~ MN(0, I, V)
- V^{-1} appears in the loss; large V_{ij} → task i and j's parameters are pulled together
- Jointly **convex** in (W, b, V)
- Originally formulated for **linear models** (shallow)

### Deep MTRL (Long et al. 2017 [95]) — Survey Section 4.4

Extends MTRL to deep networks by placing a **tensor-variate normal prior** on the parameter tensor of fully connected layers:

$$\mathcal{W} \sim \mathcal{TN}(0, \Sigma_1, \Sigma_2, \ldots, \Sigma_K)$$

- Each Σ_k captures relationships along a mode of the tensor (tasks, features, layers)
- Still a **regularization** approach (Bayesian prior on weights)
- The first deep MTRL method according to the survey
- Applied to **all fully connected layers**, not just the final classifier

### Kanakis et al. (ECCV 2020)

Adds a learned **task relationship matrix Ω** (m × m) that **reparameterizes** the final classifier layer:

$$W_{cls} = B \cdot \Omega$$

- **B** is a shared weight basis (d × m or d × h, where h is the shared dimension)
- **Ω** is the task relationship matrix (m × m or h × m)
- Each task i's classifier weights are: $w_i = B \cdot \omega_i$ (linear combination of basis columns)
- **Ω** is a **reparameterization** — it is the model architecture, not a regularizer
- Learned via standard backpropagation (no alternating optimization)
- Applied **only to the final classifier layer** of a deep network

---

## 2. Side-by-Side Comparison

| Aspect | MTRL (Zhang & Yeung) | Long et al. 2017 [95] | Kanakis 2020 |
|--------|---------------------|----------------------|--------------|
| **Mechanism** | Regularization (penalty in loss) | Regularization (Bayesian prior) | **Reparameterization** (architectural) |
| **m×m matrix** | V (task covariance) | Σ_k (per-mode covariance) | Ω (task relationship) |
| **Where applied** | All of W (full model) | All FC layers (tensor) | **Final classifier layer only** |
| **Learning** | Alternating convex opt | SGD + prior | **Standard backprop** |
| **Effect on W** | Soft constraint via penalty | Soft constraint via prior | **Hard constraint via factorization** |
| **Convex?** | Yes (shallow) | No | No |
| **Covered by survey?** | Yes (Section 4.2) | Yes (Section 4.4) | **No** |

---

## 3. The Crucial Distinction: Reparameterization vs Regularization

The difference is not cosmetic — it changes HOW the task relationship matrix interacts with the weights:

**MTRL's V (regularization)**:
- W is learned freely; V tells the optimizer "try to make W consistent with this covariance"
- If the data strongly contradicts V, the model can override it (soft constraint)
- Requires **alternating optimization** or separate update rules for W and V
- V can be any positive definite matrix (up to constraints)

**Kanakis's Ω (reparameterization)**:
- W is **explicitly factorized** as BΩ — the columns of W live in the column space of B, modulated by Ω
- Ω directly determines the linear combination of shared basis used by each task
- Learned via standard backprop (differentiable, no special optimization needed)
- Ω may have different constraints depending on implementation (often unconstrained or with mild normalization)

**Practical consequence for implementation**:
- MTRL: ~50-100 lines, separate V update step, convergence monitoring
- Kanakis: ~30 lines, add `nn.Parameter(torch.eye(m))` to your classifier, that's it

---

## 4. Is Kanakis the Same as MTRL's V?

**No.** They share the same *concept* (learn an m×m matrix that captures pairwise task correlations) but differ in *mechanism*:

- **Same**: Both produce an m×m matrix you can visualize as a task correlation heatmap
- **Not same**: MTRL's V is a regularizer in the loss; Kanakis's Ω is a reparameterization in the architecture

Think of it this way:
- MTRL says: "Learn W freely, but penalize W if it doesn't match V"
- Kanakis says: "W is BΩ — the correlation structure is built into the parameterization"

These are different mathematical operations on W, even though the output (an m×m matrix) looks the same.

---

## 5. Is Kanakis Already Covered by the Survey's Deep MTRL (Long et al. 2017)?

**No.** The survey covers Long et al. 2017 [95] in Section 4.4 ("Deep & Tensor Extensions"), which extends MTRL to deep networks via a tensor-variate normal prior. But:

- Long et al. is still a **regularization** approach (prior on weights)
- Kanakis is a **reparameterization** approach (factorized weights)
- Long et al. applies to **all FC layers**; Kanakis applies **only to the final classifier**
- The survey does not discuss reparameterization as a mechanism for task relation learning

Kanakis is **adjacent** to what the survey covers but not included in it.

---

## 6. A Complication: Kanakis Is Also Related to the LOW-RANK Approach

An important observation: Kanakis's formulation $W = B \Omega$ is structurally similar to **Alternating Structure Optimization (ASO)** from the low-rank approach (Ando & Zhang 2005):

- **ASO**: $w_i = u + \Theta^T v_i$ — shared subspace + task-specific vectors
- **Kanakis**: $w_i = B \omega_i$ — shared basis + task-specific combination weights

Kanakis effectively **bridges** the low-rank and task relation approaches:
- The parameterization is low-rank-style (shared basis + per-task coefficients)
- The interpretation is task-relation-style (Ω is analyzed as a correlation matrix)
- It's applied at the classifier layer of a deep network

This blending of categories means Kanakis doesn't fit neatly into any single survey category — it's a hybrid.

---

## 7. Verdict

| Question | Answer |
|----------|--------|
| Same as MTRL's V? | **No** — different mechanism (reparameterization vs regularization) |
| Same as Long et al. 2017? | **No** — Long et al. uses tensor-variate prior (regularization); Kanakis uses reparameterization |
| Covered by survey Section 4.4? | **No** — the survey's deep MTRL coverage is specifically Long et al. 2017 |
| Genuinely new concept? | **No** — the concept (learn m×m task correlation matrix) is the same as MTRL |
| Genuinely new mechanism? | **Yes** — reparameterization is genuinely different from regularization |

**Bottom line**: Kanakis is **not the same** as MTRL's V, and it's **not covered** by the survey. But it's also not a radical departure — it applies the **same core concept** (learned task correlations) through a **different mechanism** (reparameterization instead of regularization). The novelty is in the mechanism and the simplicity of integration into deep networks, not in the idea of learning task correlations.

For the FYP, Kanakis is valuable as a **pragmatic, implementation-light way to achieve task correlation learning in a deep network** compared to MTRL's alternating optimization or Long et al.'s tensor-variate prior. It's a 30-line change that produces the same kind of interpretable output.
