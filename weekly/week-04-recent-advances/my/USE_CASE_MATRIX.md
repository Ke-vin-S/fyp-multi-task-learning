# Use Case Matrix — Which Paper Applies to Which wavCSE Component?

> Maps each Tier 1 & Tier 2 paper to specific wavCSE components they could extend.
> See RANKINGS.md for paper details and justification.

## wavCSE Component Architecture (for reference)

```
Raw Audio → [Frozen WavLM-Large] → Frame Pooling → [Projector 1024→512] → Layer Pooling → [Hidden 512→2000] → [Task Head 1: Content]
                                                                                                          → [Task Head 2: Speaker]
                                                                                                          → [Task Head 3: Emotion]
                                                                                                          → [Task Head 4: Intent]
                                    ↑                    ↑                    ↑              ↑                    ↑
                              (1) Upstream         (2) Projection      (3) Pooling    (4) Shared Hidden   (5) Per-Task Heads

(6) Loss Aggregation: L = (L_content + L_speaker + L_emotion) / 3
(7) Training Loop: standard SGD/Adam, equal-weight averaging
```

## Component → Paper Mapping

### (1) Upstream (Frozen WavLM-Large Backbone)

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **Recon** | T1 | Which layers are shared vs task-specific | Compute S-Conflict scores per WavLM layer → convert high-conflict layers to task-specific |
| **AdaShare** | T1 | Learnable binary masks over layers | Gumbel-sigmoid masks → each task selects a subset of the 24 WavLM layers |
| **Rotograd** | T2 | Gradient space rotation for shared/task-specific decomposition | Rotate WavLM gradients to align shared acoustic features, preserve task-specific |

### (2) Projection (1024→512 Linear Layer)

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **Task-Specific Feature Learning** (Chang et al., AAAI 2021) | T2 | Orthogonal task subspaces | Replace single projector with task-specific orthogonal projections |
| **AdaShare** | T1 | Binary masks | Could extend masks to projection layer |

### (3) Layer Pooling (10 strategies: mean, max, mix, weighted, gated, auto, sap, etc.)

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **Grad-TAG** | T1 | Inform which layers to pool per task | Task affinity scores could weight layer contributions per task |
| **TAG** | T1 | Which layers benefit which tasks | Gradient-based affinity between layers and tasks |
| **Recon** | T1 | Layer-wise conflict analysis | Identify which layers cause gradient conflict between speech tasks |

### (4) Shared Hidden Layer (2000-dim)

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **MTI-Net** | T1 | Cross-task attention modulation | Each task sees weighted combination of all task representations |
| **TaskExpert** | T2 | MoE instead of single hidden | Multiple experts with task-specific gating replace the single hidden layer |
| **MIRTL** | T2 | Decompose into shared + task-specific | Split 2000-dim into shared subspace + per-task subspaces |
| **Task-Specific Feature Learning** | T2 | Orthogonal subspaces | Ensure the 2000-dim contains orthogonal content/speaker/emotion subspaces |

### (5) Per-Task Classification Heads

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **Task Relationship Matrix** (Kanakis et al.) | T2 | Learned correlation in classifier weights | Add m×m correlation matrix between head parameters |
| **Output-Level Task Relatedness** (Xiao et al.) | T2 | Feedback between task outputs | Emotion prediction feeds into speaker head via Gumbel gate |
| **Self-Auxiliaries** | T2 | Task clones for asymmetric transfer | Clone emotion head as auxiliary for content task |
| **FairBranch** | T2 | Branch by parameter similarity | Group similar task heads (content+speaker) vs outlier (emotion) |

### (6) Loss Aggregation

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **CAGrad** | T1 | Replace equal weighting with min-max optimization | Per-step gradient optimization over task losses |
| **Nash-MTL** | T1 | Replace equal weighting with Nash bargaining | Fair gradient aggregation via quadratic programming |
| **PCGrad** | T1 | Replace equal weighting with gradient projection | Project conflicting gradients to orthogonal subspaces |
| **RLW** | T1 | Replace fixed weights with random sampling | Sample weights from Dirichlet each batch |
| **FairGrad** | T1 | Add fairness constraint to aggregation | Ensure no task degrades > threshold from single-task baseline |
| **ExcessMTL** | T1 | Replace loss with excess risk | Minimize excess risk rather than raw loss |
| **IMTL** | T2 | Balance gradient magnitudes | Normalize per-task gradients before aggregation |
| **GradVac** | T2 | Target similarity instead of binary conflict | Smooth gradient similarity maintenance |
| **FAMO** | T2 | Dual descent weight learning | Learn task weights adaptively during training |
| **STCH** | T2 | Convex hull optimization | Find optimal gradient combination within convex hull |
| **MultiBalance** | T2 | Representation-level balancing | Balance gradients w.r.t. shared representation only |
| **Aligned-MTL** | T2 | Active gradient alignment | Rotate gradients to agree, not just remove conflicts |

### (7) Training Loop / Meta-Design

| Paper | Tier | What It Would Change | How |
|-------|------|---------------------|-----|
| **Grad-TAG** | T1 | Pre-training analysis | Run once to discover task affinities → inform architecture |
| **TAG** | T1 | During-training monitoring | Log inter-task affinities each epoch → visualize evolution |
| **Which Tasks Together** | T1 | Experimental design | Systematic pairwise training to determine optimal task groupings |
| **ExcessMTL** | T1 | Theoretical framework | Use excess risk analysis to explain results in paper |
| **MetaLink** | T2 | Architecture overhaul | KG-based reformulation of the entire MTL problem |

## Task-Specific Analysis

### For Content↔Speaker Relationship (expected: high positive correlation)
| Paper | What It Reveals |
|-------|----------------|
| Grad-TAG | Quantitative affinity score between KS and SI |
| TAG | How SI gradient updates affect KS loss (and vice versa) |
| Recon | Which WavLM layers have low conflict between KS and SI |

### For Content↔Emotion Relationship (expected: weaker/conflicting)
| Paper | What It Reveals |
|-------|----------------|
| CAGrad | Whether gradient conflict is severe enough to need min-max optimization |
| ExcessMTL | Whether excess risk is higher for content+emotion than content+speaker |
| Recon | Which layers show high S-Conflict scores (likely to be emotion-relevant upper layers) |

### For Task Imbalance (Speaker: 1251 classes vs Emotion: 4 classes)
| Paper | What It Addresses |
|-------|------------------|
| IMTL | Gradient magnitude normalization — prevents speaker task from dominating |
| FairGrad | Explicit degradation threshold — ensures emotion doesn't get worse |
| Nash-MTL | Fair bargaining — theoretically guarantees no task is sacrificed |
| Self-Auxiliaries | Asymmetric transfer — speaker auxiliary helps emotion without hurting speaker |

## Recommended Experimental Grid

For the ACL 2027 paper, run this matrix:

```
                    │ RLW (baseline) │ PCGrad │ CAGrad │ Nash-MTL │ FairGrad
────────────────────┼────────────────┼────────┼────────┼──────────┼──────────
KS + SI             │      ✓         │   ✓    │   ✓    │    ✓     │    ✓
KS + ER             │      ✓         │   ✓    │   ✓    │    ✓     │    ✓
SI + ER             │      ✓         │   ✓    │   ✓    │    ✓     │    ✓
KS + SI + ER        │      ✓         │   ✓    │   ✓    │    ✓     │    ✓
────────────────────┼────────────────┼────────┼────────┼──────────┼──────────
                    │   All combos   │                │
  + Grad-TAG        │   analyzed     │                │
  affinity scores   │   per cell     │                │
```

This produces:
- 5 methods × 4 task combinations = 20 experiments
- Plus single-task baselines (3 experiments)
- Plus Grad-TAG affinity analysis per task combination
- Total: ~25-30 experimental runs

Each run on wavCSE downstream takes ~few hours on single GPU → manageable for FYP timeline.
