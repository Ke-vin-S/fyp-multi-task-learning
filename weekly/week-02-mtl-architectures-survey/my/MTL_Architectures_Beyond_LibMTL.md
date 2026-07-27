# MTL Architectures Beyond LibMTL

LibMTL covers **Feature Learning** (8 architectures) + **Optimization** (24 weighting strategies). Here are the architectures from the other four categories in the Zhang & Yang survey.

---

## 1. Low-Rank Approach

**Core idea:** The parameter matrix **W** (d features × m tasks) is low-rank — related tasks lie in a shared low-dimensional subspace. Fewer degrees of freedom → less overfitting.

### Key Methods

| Method | Paper | How it works |
|--------|-------|--------------|
| **ASO (Alternating Structure Optimization)** | Ando & Zhang (2005) | `w_i = u_i + Θ^T v_i`. Θ is a shared h×d subspace (h ≪ d). Two separate sets of params. |
| **Trace Norm Regularization** | Pong et al. (2010) | Penalizes `‖W‖_{S(1)}` (sum of singular values). Convex relaxation of rank. |
| **Capped Trace Norm** | Han & Zhang (2015) | Penalizes only *small* singular values. Better rank control than plain trace norm. |
| **Spectral k-Support Norm** | McDonald et al. (2015) | Tighter convex relaxation than trace norm. |
| **Deep Tensor Trace Norm** | Yang & Hospedales (2017) | Reshapes deep network weights into tensors. Applies tensor trace norms across tasks. |

### Intuition
```
Trace norm shrinks singular values:
  W = U Σ V^T
  Regularizer = σ₁ + σ₂ + ... + σ_min(m,d)
  
→ Small singular values → 0
→ Rank drops → tasks share a compact subspace
```

### When to use
- Many tasks, expect them to share a low-dim structure
- Linear models or shallow networks (trace norm is hard to apply in deep nets)

---

## 2. Task Clustering Approach

**Core idea:** Tasks form natural clusters. Tasks in the same cluster share more with each other than with tasks in other clusters.

### Key Methods

| Method | Paper | How it works |
|--------|-------|--------------|
| **Task Clustering k-NN** | Thrun & Sullivan (1998) | First-ever task clustering. Uses pairwise transfer accuracy matrix to group tasks, then pool data within clusters. |
| **Bayesian Task Clustering** | Bakker & Heskes (2003) | Mixture-of-Gaussians prior on task weights: `w_i ~ Σ_j p_j N(μ_j, Σ_j)`. Each Gaussian = one cluster. |
| **Dirichlet Process Clustering** | Xue et al. (2007) | `w_i ~ G`, `G ~ DP(α, G₀)`. Number of clusters learned automatically (no need to pre-specify). |
| **Nested Dirichlet Process** | Rodriguez et al. (2008) | Two-level DP: cluster tasks AND cluster states within sequential data per task. |
| **Regularized Task Clustering** | Jacob et al. (2009) | Three-way regularizer: global penalty + between-cluster variance + within-cluster variance. Convex formulation. |
| **Multi-Cluster Feature Learning** | Kang et al. (2011) | Extends MTFL to K clusters. Each cluster gets its own shared feature transformation. |
| **Flexible Task Grouping** | Kumar & Daumé (2012) | `min_{W,b,Z} L(W,b) + λ₁‖W‖² + λ₂ Σzij‖w_i - w_j‖² + λ₃‖Z‖_{2,1}`. Learns which tasks are similar via Z matrix. |

### Intuition
```
Task space:
  
  [Content, Speaker-en, Speaker-zh] → Cluster 1 (linguistic)
  [Emotion, Sentiment]              → Cluster 2 (affective)
  
→ Tasks within a cluster share feature transforms or priors
→ Tasks across clusters are more independent
```

### When to use
- Tasks naturally group by domain or type
- You want interpretable task groupings
- Don't know #clusters → use Dirichlet process

---

## 3. Task Relation Learning Approach

**Core idea:** Instead of grouping tasks into hard clusters, learn a **quantitative pairwise relationship** between every pair of tasks — a similarity matrix, covariance, or graph.

### Key Methods

| Method | Paper | How it works |
|--------|-------|--------------|
| **Multi-Task Gaussian Processes** | Bonilla et al. (2008) | Learns a **task covariance matrix** Σ (m×m). Task i and j share more if Σ_{ij} is large. |
| **Task Covariance Learning** | Zhang & Yeung (2010) | `min L(W,b) + tr(W D^{-1} W^T)` where D is a learned task covariance. Related to MTFL. |
| **Multi-Task Kernel** | Evgeniou & Pontil (2004) | Same kernel for all tasks but with a task-similarity matrix mixed in. |
| **Graph Regularization** | Kato et al. (2008) | Given a task similarity graph G, penalizes `Σ a_{ij} ‖w_i - w_j‖²`. Similar tasks → similar weights. |
| **Learned Graph** | Chen et al. (2011) | Learns the graph edge weights AND the model parameters jointly. |
| **Clustered Multi-Task** | Zhou et al. (2011) | Combines clustering + relations: tasks in a cluster share mean, but pairwise relations refine the prior. |

### Intuition
```
Task covariance matrix Σ (learned):

       C   S   E
  C [ 1.0 0.7 0.3 ]
  S [ 0.7 1.0 0.5 ]
  E [ 0.3 0.5 1.0 ]
  
→ Content & Speaker are more related (0.7)
→ Content & Emotion are less related (0.3)
→ Regularization: w_i ~ N(0, Σ ⊗ I)
```

### When to use
- Want fine-grained, quantitative task relationships
- Tasks don't fall neatly into clusters
- Bayesian/probabilistic framework preferred

---

## 4. Decomposition Approach

**Core idea:** Decompose each task's parameters into **multiple components** — typically a shared part + a task-specific part. This naturally handles outlier tasks (their task-specific component dominates).

### Key Methods

| Method | Paper | How it works |
|--------|-------|--------------|
| **Dirty Model** | Jalali et al. (2010) | `W = P + Q`. P is sparse (shared features). Q is sparse (task-specific features). Uses different sparsity penalties. |
| **Robust MTL** | Chen et al. (2011) | `W = L + S`. L is low-rank (shared structure). S is column-sparse (outlier tasks). |
| **Robust Feature Learning** | Gong et al. (2012) | `W = L + S`. L captures shared features, S captures outlier task deviations. |
| **Flexible MTL** | Zhong & Kwok (2012) | General framework: `w_i = u + Θ^T v_i + b_i`. Shared mean + shared subspace + task-specific residual. |
| **Probabilistic Dirty Model** | Hernández-Lobato et al. (2015) | Bayesian version. Uses horseshoe priors to identify shared vs task-specific features. |
| **Multi-Task Multi-Output** | Rai et al. (2012) | `W = U Σ V^T` with factorized priors. Shared factors + task-specific factors. |

### Intuition
```
Decomposition for 3 tasks:

W =   L     +     S
    (shared)   (task-specific)

    [0.5 0.3]   [0.1 0.0 0.0]   ← outlier task
    [0.5 0.3]   [0.0 0.1 0.0]
    [0.5 0.3]   [0.0 0.0 0.5]   ← outlier task

→ L has repeated rows → shared structure
→ S captures what makes each task unique
→ Outlier tasks have large entries in S
```

### When to use
- Suspect outlier tasks that shouldn't share
- Want robustness — the model can "turn off" sharing for bad tasks
- Need to identify WHICH tasks are outliers

---

## 5. Quick Decision Guide

| Situation | Best Approach |
|-----------|---------------|
| Tasks clearly related, want deep model | Feature Learning (LibMTL architectures) |
| Many linear tasks, shared low-dim structure | Low-Rank |
| Tasks naturally group by type/domain | Task Clustering |
| Want fine-grained pairwise task similarity | Task Relation Learning |
| Suspect some tasks are outliers | Decomposition |
| Deep model + need task balancing | LibMTL weighting (gradient-level) |

---

## 6. What's Missing for Our FYP

For our speech MTL framework, we should consider adding:

1. **Low-rank regularization** — Could help when we have many speech tasks with shared structure
2. **Task clustering** — Natural fit: content tasks cluster together, paralinguistic tasks cluster together
3. **Decomposition** — Directly addresses our **task imbalance** problem: outlier/heavy tasks get their own component
4. **Task relation learning** — Could reveal interesting relationships between content/speaker/emotion

The LibMTL weighting strategies handle **training-time balancing**, but these architecture-level approaches handle **model-level sharing** — they're complementary and could be combined.
