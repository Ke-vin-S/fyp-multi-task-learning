# A Survey on Multi-Task Learning — Brief

**Authors:** Yu Zhang & Qiang Yang  
**Venue:** IEEE Transactions on Knowledge and Data Engineering (TKDE), Vol. 34, No. 12, December 2022  
**Published:** 2021 (online), 2022 (print)  
**Citations:** ~1,500+

---

## 1. What is MTL?

**Multi-Task Learning (MTL)** learns multiple related tasks jointly so that knowledge from each task helps the others, improving generalization for all tasks. It differs from:

| Paradigm | Difference |
|----------|------------|
| **Transfer learning** | MTL treats all tasks equally; transfer learning prioritizes a target task |
| **Multi-label learning** | In MTL, different tasks have *different data* (not shared instances with multiple labels) |
| **Multi-view learning** | Multiple views → same task; MTL → multiple tasks |
| **Continual learning** | Tasks arrive sequentially; MTL learns them together |

### Core Questions

1. **When to share?** — Single-task or multi-task? Usually decided by human experts or model selection.
2. **What to share?** — Feature, instance, or parameter. Most work is on feature-based and parameter-based MTL.
3. **How to share?** — The algorithm/mechanism for sharing.

---

## 2. The Five Approaches

### 2.1 Feature Learning Approach
Learn a common feature representation for all tasks.

| Sub-category | What it does | Key examples |
|-------------|-------------|--------------|
| **Feature Transformation** | Learns a linear/nonlinear transform of original features | MTFL (Argyriou et al.), Multi-task sparse coding, Deep MTL (shared layers), Cross-stitch networks, Adversarial MTL |
| **Feature Selection** | Selects a subset of original features | `ℓ₂,₁` norm regularization (Obozinski et al.), Capped-`ℓ_{p,1}`, Multi-level Lasso, Probabilistic models with horseshoe priors |

**Key insight for our FYP:** The feature transformation approach is the basis for wavCSE-style MTL — shared encoder extracts features, task-specific heads on top. Deep MTL (shared layers) dominates in speech/NLP.

### 2.2 Low-Rank Approach
Assumes the parameter matrix **W** (tasks × features) is low-rank — related tasks share a low-dimensional subspace.  
**Key methods:** Trace norm regularization (`‖W‖_{S(1)}`), capped trace norm, spectral k-support norm, tensor trace norms for deep models.  
**Connection:** Equivalent to assuming task parameters are linear combinations of a few latent basis tasks.

### 2.3 Task Clustering Approach
Groups similar tasks into clusters; tasks within a cluster share more than across clusters.  
**Key methods:** Multi-task Bayesian neural networks (mixture-of-Gaussians prior), Dirichlet process clustering, regularization with between/within-cluster variance terms.

### 2.4 Task Relation Learning Approach
Learns quantitative task relationships (pairwise similarity, covariance, or graph structure) directly from data.  
**Key methods:** Task covariance learning, graph regularization, Bayesian task relation priors.

### 2.5 Decomposition Approach
Decomposes the parameter matrix into components (e.g., shared + task-specific, or latent factors).  
**Key idea:** `w_i = u_i + Θ^T v_i` — shared subspace component + task-specific component. Enables handling outlier tasks naturally.

---

## 3. Quick Reference: Which Approach Fits What?

| Approach | When to Use | Strength | Weakness |
|----------|-------------|----------|----------|
| Feature Learning | High-dim data, related tasks | Powerful representations | Vulnerable to outlier tasks |
| Low-Rank | Many tasks, clear shared structure | Theoretical guarantees | Linear assumptions |
| Task Clustering | Tasks naturally group | Interpretable groups | Choosing #clusters |
| Task Relation | Pairwise task similarities matter | Flexible, data-driven | Computationally heavier |
| Decomposition | Mix of shared + unique structure | Handles outliers | More hyperparameters |

---

## 4. MTL + Other Paradigms

- **Semi-supervised MTL** — unlabeled data helps learn better shared representations
- **Active MTL** — query most informative task-data pairs for labeling
- **Unsupervised MTL** — learn shared representations without labels (e.g., multi-task autoencoders)
- **Reinforcement MTL** — share policies or value functions across related RL tasks
- **Multi-view MTL** — each task has multiple feature views

---

## 5. Speech & NLP Applications (Relevant to FYP)

In speech and NLP:
- Data has **sequential structure** → RNN-based deep MTL models dominate
- **Feature transformation approach** is most widely used
- Example: adversarial multi-task learning for robust speech recognition (Shinohara, 2016)

**Popular approaches by application area:**
- **Computer vision:** Deep feature transformation
- **Speech/NLP:** RNN-based feature transformation, task relation learning
- **Web:** Simple linear models + boosting (large-scale data)

---

## 6. Key Theoretical Results

- **Generalization bounds** improve with more tasks: best rate is **O(1/√(mn₀))** where *m* = #tasks, *n₀* = avg samples/task
- **Local Rademacher complexity** gives tightest bounds
- MTL helps most when tasks are genuinely related — **outlier tasks degrade all tasks**

---

## 7. Open Problems & Future Directions

1. **Outlier tasks** — No principled way to detect/handle them. Critical for safe MTL deployment.
2. **Robust deep MTL** — Shared-layer deep MTL is vulnerable to noisy/outlier tasks. Need flexible, robust architectures.
3. **Beyond supervised MTL** — Most work is on supervised tasks. Need more models for unsupervised, semi-supervised, RL, etc.
4. **When to share** — Still decided by humans. Need automatic model selection between single-task and multi-task.
5. **Broader AI applications** — Logic, planning, reasoning.

---

## 8. Relevance to Our FYP

| Paper concept | FYP connection |
|---------------|---------------|
| Feature transformation approach | wavCSE baseline — shared encoder + task-specific heads |
| Parameter-based MTL (low-rank, task relation) | LibMTL weighting strategies — how to balance task losses |
| Task clustering | Could group content/speaker/emotion tasks by similarity |
| Outlier tasks | **Core FYP focus** — task imbalance and mitigation |
| Speech applications | Directly applicable — our domain |
| Deep MTL + shared layers | Speech LLM backbone for our experiments |

---

## 9. Quick Taxonomy (from the paper)

```
MTL
├── Feature-based
│   └── Feature Learning Approach
│       ├── Feature Transformation (MTFL, deep shared layers, cross-stitch, adversarial)
│       └── Feature Selection (ℓ₂,₁ norm, capped-ℓ_{p,1}, probabilistic)
│
└── Parameter-based
    ├── Low-Rank Approach (trace norm, spectral k-support)
    ├── Task Clustering Approach (Dirichlet process, mixture priors)
    ├── Task Relation Learning Approach (covariance, graph regularization)
    └── Decomposition Approach (shared + task-specific components)
```
