# Overlap Check #2: MTI-Net vs. Zhang & Yang Survey

**File:** `/home/kevin/projects/fyp/approaches/recent/summaries/mti-net.md`
**Survey Section:** Section 2.4 (Task Relation Learning), with references to Sections 2.1 (Feature Learning) and 2.5 (Decomposition)

**Verdict: MODERN VARIANT — same conceptual category, genuinely new mechanism.**

MTI-Net belongs conceptually to Section 2.4 of the survey (Task Relation Learning), but the specific mechanism it introduces — cross-task attention on feature maps — is not present anywhere in the survey's coverage. This is not a fundamentally new category of approach, but it is a meaningfully novel implementation within that category.

---

## 1. What is the same: the high-level concept

The survey's Section 2.4 is defined by a single overarching idea:

> "Explicitly quantifies how tasks relate to each other — then uses those relationships to guide the learning process." (Section 2.4, Overview)

MTI-Net does exactly this:
- It computes pairwise attention maps between tasks (quantifying where task A finds useful information in task B).
- Those attention maps modulate each task's feature representations (guiding the learning process).

In this abstract sense, MTI-Net is textbook Section 2.4. The survey's own taxonomy would place it under "Generation 2: Learning Task Relations from Data," alongside MTRL [79, 80] and MTGP [76].

---

## 2. What is different: the mechanism

Here is where the survey's coverage ends and MTI-Net diverges. Section 2.4 describes three categories of mechanisms for representing and using task relations:

### Mechanism A: Covariance-based (MTRL family) [79-95]
- Task relations = a positive-definite covariance matrix V (m x m).
- V is used as a **parameter-level regularizer**: tr(W V^{-1} W^T) pulls w_i and w_j toward each other when V_ij is large.
- The interaction is global (one scalar per task pair), symmetric, and operates on **parameters**, not features.
- **Deep MTRL [95]** (Long et al., 2017) extends this to neural nets via a tensor-variate normal prior on FC layer parameters — still at the **parameter regularization** level. Not attention, not feature-level.

### Mechanism B: Functional (MTGP family) [76-78]
- Task relations = a task covariance matrix learned in function space via Gaussian processes.
- Models correlations between task outputs, not feature representations.
- No attention mechanism.

### Mechanism C: Asymmetric [97]
- Task relations = matrix A where A_ij is how much task j contributes to task i.
- Still at the **parameter level**: W ~= WA.
- No attention, no feature-level modulation.

### What MTI-Net introduces (not in any of the above):

1. **Feature-level attention, not parameter-level regularization.** Task A's **feature map** attends to task B's feature map to produce modulated features for task B. This is entirely different from regularizing task parameters toward a shared mean/covariance. The survey has no method that operates at the feature-representation level.

2. **Spatially-aware interactions.** The cross-task attention weights are **per-location** (per pixel in vision, per timestep in speech), not one global scalar per pair. Task A might attend to different regions of task B's features for different inputs.

3. **Multi-scale interactions.** MTI-Net computes cross-task attention at 4 resolution scales independently and fuses them. The survey has no concept of multi-scale task relations.

4. **No explicit covariance/similarity matrix.** MTI-Net does not produce an interpretable "task relation matrix" in the traditional sense. The relations are implicit in the learned attention weights, which are input-dependent (different inputs produce different attention patterns).

5. **Bidirectional but asymmetric in content.** Task A's features modulate Task B, and vice versa — but the attention maps are independently learned, so the influence need not be symmetric. This goes beyond the survey's [97] which models asymmetry only at the parameter level.

---

## 3. Could it fit in another survey section?

### Section 2.1 (Feature Learning / Shared Representations)
- MTI-Net does use a shared encoder backbone, but that is standard practice, not the innovation.
- Section 2.1 covers methods where a shared feature representation is learned and then split into task-specific branches (often called "feature-based MTL"). The sharing is **static** — determined by architecture, not by learned task interactions.
- MTI-Net goes beyond this by making the task branches **dynamically modulate each other** via attention. The survey's Section 2.1 has no mechanism for task-to-task feature modulation.

### Section 2.5 (Decomposition)
- Not applicable. MTI-Net does not decompose the parameter matrix W into shared + task-specific components.

### No other section fits better than 2.4.

---

## 4. Key quotes from the survey documents

From Section 2.4:
> The core MTRL formulation is: min tr(W V^{-1} W^T) where V is the task covariance matrix. (Section 2.4, Mathematical Deep Dive)

> Deep MTRL [95] extends MTRL to neural networks: places a tensor-variate normal distribution as a prior on the parameter tensor in fully connected layers. (Section 2.4, 4.4)

These are the survey's deepest foray into neural task relation learning (Long et al., 2017), and they are clearly distinct from MTI-Net (2020). One is parameter-level regularization; the other is feature-level attention.

The survey also says:
> [95] bridges task relation learning with modern deep learning — the first deep MTRL method. (Section 2.4, 4.4)

This statement was arguably overtaken by MTI-Net just a few years later. The "bridge" was incomplete — tensor-variate priors are not the same as learned cross-task attention.

---

## 5. Why this matters for your FYP

| Aspect | Survey Section 2.4 | MTI-Net | Gap for FYP |
|--------|-------------------|---------|-------------|
| Level of interaction | Parameter / function | Feature map | Feature-level is unexplored in survey |
| Spatial awareness | None (global scalar per pair) | Per-location attention | Relevant for speech (temporal attention) |
| Multi-scale | None | 4 resolution scales | Relevant for WavLM's hierarchical features |
| Implementation complexity | Low (regularizer only) | High (architectural change) | Trade-off: greater expressiveness vs. harder training |

For your FYP, this distinction matters because:
- If you claim MTI-Net as a "new approach category," you are overstating it — it is clearly a Task Relation Learning method.
- But it is also clearly **not represented** by any of the specific mechanisms the survey describes. The survey's Sections 2.4.1-2.4.6 do not contain attention-based cross-task feature modulation.
- A fair characterization: "MTI-Net is a Task Relation Learning approach (per the Zhang & Yang taxonomy), but it introduces a novel mechanism — cross-task attention on feature maps — that is not covered in the survey's review, which focuses on covariance-based and parameter-level relation learning."

---

## 6. Bottom line

| | Assessment |
|---|-----------|
| **Same conceptual category as survey?** | Yes — Section 2.4 (Task Relation Learning) |
| **Same mechanism as anything in survey?** | No — cross-task attention on feature maps is absent from the survey |
| **New category of approach?** | No — it fits cleanly into the existing Section 2.4 framework |
| **New mechanism within that category?** | Yes — attention-based feature modulation is distinct from covariance-based regularization (MTRL), functional relations (MTGP), and tensor-variate priors (Deep MTRL) |
| **Overall verdict** | **MODERN VARIANT.** The survey's taxonomy correctly classifies it, but the survey's specific methods do not cover it. It is a meaningful architectural innovation within an established category. |
