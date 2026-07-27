# FYP Project Context: Extending wavCSE for Multi-Task Learning Analysis

## Our FYP: The Next Step

Our project directly extends the wavCSE paper. Here's how:

---

## Relationship Between wavCSE and Our FYP

```
wavCSE (Sritharan & Thayasivam, 2025)          Our FYP (2026)
─────────────────────────────────────          ──────────────
"What if we train multiple speech         →    "How do different ways of sharing
 tasks together to get a unified               knowledge between tasks affect
 embedding?"                                   individual task performance?"
                                             
Focus: Producing a good embedding          Focus: Understanding MTL strategies
                                             
One MTL strategy: feature-based           →    Compare: feature-based vs.
with equal weighting                           parameter-based MTL
                                             
Fixed loss balancing                      →    Study: task imbalance + mitigation
                                             
3 tasks (KS, SID, ER)                     →    Content + Speaker + Emotion tasks
                                             
Single SSL backbone                       →    Same speech LLM backbone
```

---

## Our Project Description (Verbatim)

> Multi-task learning (MTL) enables models to jointly learn multiple related tasks by sharing information across them, often leading to improved generalization. However, different MTL strategies vary significantly in how knowledge is shared between tasks, and their impact on individual task performance remains insufficiently understood.
>
> This project aims to **systematically analyze how different MTL formulations influence performance across speech-related tasks**. The study will use a speech LLM as the backbone to extract representations and support multi-task training, allowing consistent comparison across settings.
>
> To achieve this, a **modular experimental framework** will be developed to evaluate different MTL strategies on speech tasks. The study will implement and compare multiple approaches, including **feature-based MTL** and **parameter-based MTL**, using the same speech LLM setup.
>
> Using this framework, performance will be analyzed across **content, speaker, and emotion tasks** to understand how different sharing strategies influence each task. In addition, the **effect of task imbalance** will be examined, along with simple strategies to reduce its impact.
>
> Through this analysis, the project aims to provide a clearer understanding of how MTL design choices affect task performance and interaction.

---

## Key Differences: What We're Adding

### 1. Systematic MTL Strategy Comparison

| Aspect | wavCSE | Our FYP |
|--------|--------|---------|
| **Feature-based MTL** | ✅ Used (shared backbone + task heads) | ✅ Will include |
| **Parameter-based MTL** | ❌ Not explored | ✅ Will implement and compare |
| **Comparison framework** | ❌ Single strategy | ✅ Modular framework for A/B testing |

**Feature-based MTL:** Tasks share the same input representations but have separate model parameters (what wavCSE does).

**Parameter-based MTL:** Tasks share model parameters through mechanisms like:
- Hard parameter sharing (shared bottom layers)
- Soft parameter sharing (separate networks with regularization to encourage similarity)
- Cross-stitch networks
- Mixture-of-experts

### 2. Task Imbalance Analysis

| Aspect | wavCSE | Our FYP |
|--------|--------|---------|
| **Loss balancing** | Equal weighting only | Multiple strategies |
| **Dataset imbalance** | Not studied | Will examine effects |
| **Mitigation strategies** | None | Will implement and evaluate |

We'll explore strategies like:
- Uncertainty weighting (Kendall et al.)
- Gradient normalization (GradNorm)
- Dynamic task prioritization
- Dynamic Weight Average (DWA)

### 3. Systematic Performance Analysis

Instead of just reporting "does it work?", we'll analyze:
- **Task interactions:** Does adding task B help or hurt task A?
- **Sharing depth:** How much sharing is optimal?
- **Task affinity:** Which tasks benefit from co-training?
- **Imbalance effects:** What happens when one task has 100x more data?

---

## Reference Papers

Three key references for our work:

1. **Sritharan & Thayasivam (2025)** — wavCSE paper (IJCNLP-AACL 2025)
   - Our baseline; we extend this work
   - Provides the speech LLM backbone and MTL setup

2. **Zhang & Yang (2021)** — "A Survey on Multi-Task Learning" (IEEE TKDE)
   - Theoretical taxonomy of MTL strategies
   - Covers feature-based vs. parameter-based, loss balancing, task relationship learning
   - Foundation for our systematic comparison

3. **Lin & Zhang (2023)** — "LibMTL: A Python Library for Deep Multi-Task Learning" (JMLR)
   - Reference architecture for our modular MTL framework
   - Provides standardized implementations of many MTL methods
   - We'll build a similar library specialized for speech tasks

---

## Deliverables

1. **Python Library for Speech MTL** — Modular framework supporting:
   - Multiple MTL strategies (feature-based, parameter-based)
   - Multiple loss balancing strategies
   - Easy configuration and comparison
   - Integration with speech LLM backbones

2. **Comprehensive Experimental Analysis** — Understanding of:
   - How sharing strategies affect content vs. speaker vs. emotion tasks
   - Task imbalance effects and mitigation
   - Design recommendations for speech MTL

3. **Target Publication:** ACL 2027

---

## Implementation Plan (High-Level)

### Phase 1: Build the Framework
- Fork/extend wavCSE codebase
- Add parameter-based MTL architectures
- Add multiple loss balancing strategies
- Create modular configuration system (like LibMTL)
- Add experiment tracking (metrics, logging, visualization)

### Phase 2: Run Experiments
- Baseline: feature-based MTL (wavCSE replication)
- Parameter-based MTL variants
- Task imbalance scenarios (controlled experiments)
- Ablation studies on sharing depth

### Phase 3: Analysis & Write-up
- Task interaction analysis
- Performance comparison across strategies
- Recommendations for MTL design in speech
- Paper writing for ACL 2027

---

## Key Decisions for Our FYP

| Decision | Options | Notes |
|----------|---------|-------|
| **Speech backbone** | WavLM Large (same as wavCSE) vs. newer models | Consistency with wavCSE is valuable |
| **Tasks** | KS+SID+ER (3 tasks) vs. adding IC (4 tasks) | More tasks = more complex interactions |
| **MTL strategies** | Which parameter-based methods to implement? | Start with hard sharing + soft sharing |
| **Imbalance scenarios** | Which types? (data size, task difficulty, label noise) | Prioritize data size imbalance |
| **Framework design** | Extend wavCSE directly vs. build new | Extending is faster but may be constrained |
