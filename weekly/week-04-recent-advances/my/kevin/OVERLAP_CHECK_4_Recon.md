# Overlap Check #4: Recon vs. Zhang & Yang Survey

**Status: GENUINELY NEW** (not a variant of the Decomposition approach, not covered by the survey)

---

## The Verdict

Recon introduces a **conceptual framework entirely absent from the Zhang & Yang survey**: using per-layer gradient conflict signals (S-Conflict scores) to drive architectural interventions (layer duplication). This is not the same as the decomposition approach, nor a gradient-based variant of it.

---

## 1. Decomposition (Section 2.5) vs. Recon: Same Question, Different Mechanism

The survey's decomposition approach (`W = L + S`, Jalali et al. 2010; Chen et al. 2011; Gong et al. 2012; Zhong & Kwok 2012) and Recon both ask "what should be shared vs. task-specific?" but answer it through fundamentally different mechanisms:

| Dimension | Decomposition (Section 2.5) | Recon |
|-----------|----------------------------|-------|
| **Unit of analysis** | Individual parameters (entries in W) | Entire layers |
| **Signal** | Optimization loss + regularization (the regularizer induces S to be column-sparse) | Gradient conflict (cosine similarity of per-task gradients, i.e. S-Conflict scores) |
| **Intervention** | Parameter values are decomposed: `W = shared_component + specific_component` — the ARCHITECTURE does not change | High-conflict layers ARE DUPLICATED into task-specific copies — the ARCHITECTURE changes mid-training |
| **When sharing is determined** | During optimization, via regularization penalty | During a separate analysis phase (warmup period), before the architectural change |

The decomposition approach keeps the network architecture fixed. Each task still has one set of weights per layer; those weights are just the sum of a shared component L and a task-specific component S. Recon, by contrast, literally creates new parameters by copying layers. The two are **structurally different** — Recon gives each task its own independent copy of a high-conflict layer, whereas decomposition gives each task a shared base plus additive deviations.

---

## 2. Does the Survey Cover Gradient-Based Sharing Analysis? No.

The survey determines sharing structure through:

- **Regularized optimization** (Section 2.5): sparsity-inducing norms on S force W into shared + specific parts
- **Probabilistic priors** (Section 2.4): matrix-variate normal priors with learned task covariance V
- **Clustering** (Section 2.3): discrete groups from data proximity
- **Low-rank constraints** (Section 2.2): trace norm shrinks the parameter space

**None of these use gradient signals.** The S-Conflict cosine similarity metric — the core innovation of Recon — does not appear in any of the five approaches. The survey does not discuss:

- Measuring per-task gradient alignment at individual layers
- Using gradient conflict as a signal for sharing decisions
- A separate analysis phase before architecture modification

Sections 2.4 (Task Relation) discusses learning a task covariance matrix V from data, but V captures similarity between task **parameters**, not similarity between task **gradient directions**. These are different quantities: parameter similarity reflects learned weight configurations, while gradient similarity reflects optimization dynamics. Recon specifically targets optimization conflicts, which is an orthogonal concern.

---

## 3. The Three Things That Make Recon Genuinely New

### 3a. Layer-level analysis unit

The survey's approaches all operate at the **parameter matrix** level (W as a d x m matrix of task parameters). Recon operates at the **layer** level — it treats entire layers as atomic units of analysis and intervention. This is motivated by deep learning: in a shared trunk with task-specific heads, different layers serve different purposes (early: general features; middle: task-agnostic; late: task-specific). The parameter-level decomposition cannot capture this layered structure.

The closest analogue in the survey is Long et al. (2017) [95] (Section 2.4.4), which extends MTRL to tensor parameters for deep networks. But that is still about learning a **parameter covariance structure** across layers, not about **gradient conflict** or **layer duplication**.

### 3b. Gradient conflict as a sharing signal

The S-Conflict score — average cosine similarity of all pairs of task gradients at a layer — is a new concept. It answers "does this layer receive conflicting update directions from different tasks?" rather than "which tasks are related?" or "should these parameters be shared?" The survey has no equivalent.

### 3c. Architecture modification during training

Recon modifies the model architecture mid-training (shared warmup -> conflict analysis -> layer duplication -> continued training). The survey's approaches either fix the architecture upfront (hard parameter sharing, cross-stitch networks) or learn a parameter decomposition during training. Starting with full sharing, measuring conflict, and THEN branching is a training protocol not described in the survey.

---

## 4. What Could Be Misleading

The superficial similarity to decomposition comes from shared motivation: both recognize that uniform sharing hurts some tasks. But saying "Recon is decomposition with gradients" is like saying "a fork is a spoon with tines" — the mechanism is fundamentally different:

- **Decomposition**: regularizer induces L (shared) + S (specific) in parameter VALUES
- **Recon**: gradient analysis determines which LAYERS become task-specific COPY parameters

One is an algebraic split of a fixed matrix; the other is an architectural split guided by optimization dynamics. They are **complementary**, not overlapping.

---

## 5. Recommendation

Recon qualifies as Upgrade #4 with no overlap concerns. It should be properly situated relative to the survey as a **new conceptual paradigm**: neither feature-based (shared trunk), parameter-based (decomposition of W), nor optimization-based (loss weighting), but **architecture-discovery-based** — using gradient conflict to determine which layers should be shared.

For the paper, the positioning should be: "While the survey covers five approaches to determine sharing structure, none use gradient conflict signals to drive architectural decisions at the layer level. Recon introduces this as a sixth paradigm."
