# Task Relationship Learning — Paper Rankings for wavCSE Extension

> Research date: 2026-07-19 | For: FYP Multi-Task Learning for Speech

## Ranking Criteria (Weighted)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Applicability to wavCSE | 40% | Can it be integrated into frozen-WavLM + per-task heads architecture? |
| Task Relationship Modeling | 25% | How explicitly does it model task relatedness? Interpretability? |
| Empirical Strength | 20% | Scale of experiments, benchmark results, statistical rigor |
| Implementation Feasibility | 15% | Code availability, complexity, compute for FYP timeline |

---

## Tier 1: Must-Read — Directly Applicable to wavCSE

These papers can be integrated into wavCSE's architecture with moderate effort and directly advance the project's north star.

### 1. Grad-TAG: Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity
- **Authors**: Dongyue Li, Aneesh Sharma, Hongyang R. Zhang
- **Venue**: KDD 2024
- **arXiv**: 2409.06091
- **Category**: Task Relation + Gradient
- **Ranking justification**: The most computationally practical method for discovering speech task relationships. Uses gradient-based linearization to estimate pairwise and higher-order task affinities without retraining all task combinations — 45× fewer FLOPs than full training. For the 3-task speech scenario, this is trivially cheap. Also includes SDP-based task clustering. Directly applicable: run one training pass on wavCSE, extract gradient-based task affinities between content/speaker/emotion, and visualize the learned task structure.
- **wavCSE integration**: Run downstream training once → extract Grad-TAG affinities → inform architecture design (which tasks share more layers) + publish the task affinity matrix as an interpretability result.
- **Implementation complexity**: Low-Medium

### 2. TAG: Efficiently Identifying Task Groupings for Multi-Task Learning
- **Authors**: Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, Chelsea Finn
- **Venue**: NeurIPS 2021
- **arXiv**: 2211.00288
- **Category**: Task Relation
- **Ranking justification**: The foundational gradient-based task affinity method. Computes affinities from a single jointly-trained model by looking at how each task's gradient update affects other tasks' losses. Scales as O(n) rather than O(n²). Proven on 39 tasks. For wavCSE's 3 tasks, this is instantaneous — run one multi-task training, extract inter-task affinity scores, and you have a quantitative answer to "which speech tasks help each other?"
- **wavCSE integration**: Add affinity computation hook during training → log content↔speaker, speaker↔emotion, content↔emotion affinities per epoch → visualize evolution of task relationships during training.
- **Implementation complexity**: Low

### 3. Which Tasks Should Be Learned Together in Multi-Task Learning?
- **Authors**: Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
- **Venue**: ICML 2020
- **arXiv**: 1905.07575
- **Category**: Task Relation
- **Ranking justification**: The systematic framework for answering "should these 3 speech tasks share a backbone?" Provides a rigorous methodology: train all task pairs, measure performance vs single-task baseline, cluster tasks. For wavCSE's 3 tasks, this gives the definitive answer on optimal task grouping — train content+speaker, content+emotion, speaker+emotion pairs and measure affinity. Directly supports the FYP objective of "analyzing how MTL design choices affect individual task performance."
- **wavCSE integration**: Run pairwise task combinations through the downstream pipeline → produce an affinity matrix → determine optimal task groupings.
- **Implementation complexity**: Low (just run multiple configs)

### 4. CAGrad: Conflict-Averse Gradient Descent for Multi-Task Learning
- **Authors**: Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu
- **Venue**: NeurIPS 2021
- **arXiv**: 2110.14048
- **Category**: Gradient
- **Ranking justification**: Theoretically principled gradient conflict resolution. Formulates MTL optimization as finding a gradient direction that minimizes average loss while staying within a ball around the average. Balances between the mean gradient and individual task gradients via convex optimization. For speech tasks with very different loss scales (KS ~12 classes vs Speaker ~1251 classes), this prevents any single task from dominating the shared backbone update.
- **wavCSE integration**: Replace the simple `loss_weight = 1/num_tasks` in `trainer_model.py` with CAGrad's min-max optimization over task gradients.
- **Implementation complexity**: Medium

### 5. Nash-MTL: Multi-Task Learning as a Bargaining Game
- **Authors**: Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, Ethan Fetaya
- **Venue**: ICML 2022
- **arXiv**: 2202.08617
- **Category**: Gradient
- **Ranking justification**: Game-theoretic formulation that guarantees fair gradient updates. Computes scale-invariant weighted gradient combinations via Nash bargaining — solves a small quadratic program per step. Particularly valuable for the FYP's task imbalance problem: emotion recognition (4 classes, small dataset) vs speaker identification (1251 classes, large dataset). Nash-MTL ensures the smaller task isn't sacrificed.
- **wavCSE integration**: Replace equal loss weighting with Nash bargaining solution over per-task gradients. Already implemented in LibMTL — can use as reference.
- **Implementation complexity**: Medium

### 6. Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning
- **Authors**: Anonymous (published at ICLR 2023)
- **Venue**: ICLR 2023
- **Category**: Gradient + Architecture
- **Ranking justification**: Introduces layer-wise S-Conflict scores to identify which shared layers cause the most gradient conflict. High-conflict layers are converted to task-specific parameters. This is directly relevant to wavCSE's architecture question: which WavLM layers should be shared vs task-specific? Unlike wavCSE's current recursive layer selection (manual/hand-tuned), Recon learns this automatically from gradient signals.
- **wavCSE integration**: Compute S-Conflict scores per WavLM layer → automatically determine which layers should be shared (low conflict) vs task-specific (high conflict) → replace manual `selected_transformer_layers` config.
- **Implementation complexity**: Medium-High

### 7. AdaShare: Learning What to Share for Multi-Task Learning
- **Authors**: Ximeng Sun, Rameswar Panda, Rogerio Feris
- **Venue**: CVPR 2020
- **arXiv**: 1911.12190
- **Category**: Task Relation + Architecture
- **Ranking justification**: Learns task-specific binary masks over shared network blocks using Gumbel-sigmoid relaxation. For wavCSE, this could replace the manual layer selection with learned masks over WavLM layers — each speech task gets its own subset of backbone features. Provides interpretability: which layers does emotion use vs content?
- **wavCSE integration**: Add learned binary masks over WavLM layer outputs → each task selects a subset of layers → visualize which layers each speech task prefers.
- **Implementation complexity**: Medium

### 8. ExcessMTL: Excess Risk Optimization for Multi-Task Learning
- **Authors**: Feiyang Ye, Baijiong Lin, Yu Zhang
- **Venue**: ICML 2024
- **arXiv**: 2405.12003
- **Category**: Gradient + Theory
- **Ranking justification**: The most theoretically rigorous recent MTL paper. Minimizes excess risk across tasks rather than raw loss — this is important for speech MTL where tasks have inherently different difficulty levels and noise characteristics. Links gradient conflicts to excess risk, providing theoretical understanding of WHY certain speech task combinations work better than others.
- **wavCSE integration**: Use excess risk framework to analyze WHY content+speaker works better than content+emotion (or vice versa) — publish as theoretical contribution.
- **Implementation complexity**: Medium

### 9. PCGrad: Gradient Surgery for Multi-Task Learning
- **Authors**: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
- **Venue**: NeurIPS 2020
- **arXiv**: 2001.06782
- **Category**: Gradient
- **Ranking justification**: The canonical gradient surgery baseline. Simple, effective, widely adopted. Detects conflicting gradients via negative cosine similarity and projects each onto the other's orthogonal subspace. Essential baseline for any gradient-based MTL experiment. Already in LibMTL.
- **wavCSE integration**: Drop-in replacement for current equal-weight loss averaging. Minimum viable gradient-based extension.
- **Implementation complexity**: Low

### 10. RLW: Random Loss Weighting for Multi-Task Learning
- **Authors**: Baijiong Lin, Feiyang Ye, Yu Zhang, Ivor W. Tsang
- **Venue**: TMLR 2022
- **arXiv**: 2111.10603
- **Category**: Gradient
- **Ranking justification**: A deceptively simple finding: randomly sampling task weights each iteration often matches or beats sophisticated methods. This is the essential baseline — if your fancy task relationship method doesn't beat random weighting, it's not useful. Must be included in any rigorous wavCSE experiment.
- **wavCSE integration**: 10-line change: sample weights from Dirichlet distribution each batch. Critical baseline for all experiments.
- **Implementation complexity**: Trivial

### 11. MTI-Net: Multi-Scale Multi-Task Interaction Network
- **Authors**: Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool
- **Venue**: ECCV 2020
- **arXiv**: 2003.12908
- **Category**: Task Relation + Architecture
- **Ranking justification**: Cross-task attention mechanism at multiple scales. Each task's features are modulated by other tasks' features. For speech: emotion features could attend to speaker features, content could attend to both. The attention weights are learnable and interpretable — can visualize how much each task "listens to" other tasks.
- **wavCSE integration**: Add cross-task attention between per-task classifier inputs → each task head sees a weighted combination of all task representations → learnable task interaction.
- **Implementation complexity**: Medium

### 12. FairGrad: Fairness-Aware Multi-Task Learning
- **Authors**: Seung-Han Baek, Jaemin Yoo, Sanghyuk Lee
- **Venue**: ICML 2024
- **arXiv**: 2402.15698
- **Category**: Gradient
- **Ranking justification**: Explicitly prevents any task from degrading below a threshold relative to its single-task performance. Critical for the FYP's task imbalance concern: adding content/speaker tasks shouldn't destroy emotion recognition. Provides both fairness guarantees and practical implementation.
- **wavCSE integration**: Add fairness constraint to gradient aggregation → ensure no task degrades > X% from its single-task baseline.
- **Implementation complexity**: Medium

---

## Tier 2: Valuable Ideas — Needs Adaptation for Speech

These papers have strong methods but require more adaptation effort or are less directly pluggable into wavCSE.

### 13. MetaLink: Relational Multi-Task Learning
- **Authors**: Kaidi Cao, Jiaxuan You, Jure Leskovec
- **Venue**: ICLR 2022 (Spotlight)
- **arXiv**: 2303.07666
- **Category**: Task Relation
- **Key idea**: Builds a knowledge graph connecting data nodes and task nodes, reformulating MTL as link prediction. Up to 27% improvement on biochemical/vision benchmarks.
- **Relevance**: Ambitious — could model speech tasks as nodes in a graph with learned edges. But significant architectural change from wavCSE's simple pipeline.
- **Implementation complexity**: High

### 14. MIRTL: Informative Relationship Multi-Task Learning
- **Authors**: Chang et al.
- **Venue**: Knowledge-Based Systems, 2024
- **Category**: Task Relation
- **Key idea**: Decomposition-based framework that quantifies pairwise knowledge contribution using exclusive Lasso for task-specific characteristics. Avoids compulsory similarity constraints.
- **Relevance**: The decomposition approach could separate shared acoustic features from task-specific speech features. Complements the decomposition approach from Zhang & Yang Section 2.5.
- **Implementation complexity**: Medium-High

### 15. Modeling Output-Level Task Relatedness with Feedback Mechanism
- **Authors**: Xiao et al.
- **Venue**: arXiv 2024 (2404.00885)
- **Category**: Task Relation
- **Key idea**: One task's output serves as hidden feature for another via Gumbel gating. Models task relatedness at the OUTPUT level (most work models parameter/feature level).
- **Relevance**: Novel angle for speech: emotion prediction could serve as a feature for speaker identification (or vice versa). The Gumbel gating learns which tasks benefit from others' outputs.
- **Implementation complexity**: Medium

### 16. Reparameterizing MTL with Task Relationship Matrix
- **Authors**: Menelaos Kanakis, Songyou Peng, David Bruggemann, Fisher Yu, Luc Van Gool
- **Venue**: ECCV 2020
- **arXiv**: 2007.07579
- **Category**: Task Relation
- **Key idea**: Task relationship matrix reparameterizes the final classifier layers. Captures task correlations in the classification heads.
- **Relevance**: Direct fit for wavCSE's multi-head classifier — add a learned relationship matrix between the 3 per-task classification heads. Lightweight change.
- **Implementation complexity**: Low-Medium

### 17. TaskExpert: Dynamically Assembling Multi-Task Representations
- **Authors**: Hanrong Ye, Dan Xu
- **Venue**: ICCV 2023
- **arXiv**: 2211.13999
- **Category**: Architecture + Task Relation
- **Key idea**: Mixture-of-experts with task-specific gating and multi-task feature memory. Decomposes shared features into task-generic subspaces.
- **Relevance**: Could replace wavCSE's simple shared hidden layer with expert-based routing. But adds significant complexity.
- **Implementation complexity**: High

### 18. Aligned-MTL: Aligned Gradient for Multi-Task Learning
- **Authors**: Zixiang Chen, Yi Xu, Qilong Wang, Yinghuan Shi, He-Xin, Yang Gao
- **Venue**: CVPR 2023
- **arXiv**: 2303.11483
- **Category**: Gradient
- **Key idea**: Extends PCGrad with active gradient alignment — not just removing conflicts but actively rotating gradients to agree.
- **Relevance**: Strong gradient method, but CAGrad and Nash-MTL are more theoretically grounded. Useful as a comparison point.
- **Implementation complexity**: Medium

### 19. IMTL: Towards Impartial Multi-Task Learning
- **Authors**: Liyang Liu, Yi Li, Zhanghui Kuang, et al.
- **Venue**: ICLR 2021
- **arXiv**: 2010.01096
- **Category**: Gradient
- **Key idea**: Balances gradient magnitudes across tasks to prevent large-gradient tasks from dominating. Simple normalization-based approach.
- **Relevance**: Directly addresses the KS (many samples, strong gradients) vs Emotion (few samples, weak gradients) imbalance.
- **Implementation complexity**: Low

### 20. GradVac: Gradient Vaccine for Multi-Task Optimization
- **Authors**: Zirui Wang, Yulia Tsvetkov, Orhan Firat, Yuan Cao
- **Venue**: NeurIPS 2021
- **arXiv**: 2106.00108
- **Category**: Gradient
- **Key idea**: Maintains a target gradient similarity rather than binary conflict detection. Smoothly pushes task gradients toward a target cosine similarity.
- **Relevance**: More nuanced than PCGrad. For speech tasks where some gradient disagreement is healthy, GradVac keeps beneficial diversity.
- **Implementation complexity**: Medium

### 21. FAMO: Fast Adaptive Multi-Objective Optimization
- **Authors**: Bo Liu, Xingchao Liu, Peter Stone, Qiang Liu
- **Venue**: NeurIPS 2023
- **arXiv**: 2310.03467
- **Category**: Gradient
- **Key idea**: Fast alternating optimization with learned task weights via dual descent. Lower overhead than full Pareto methods.
- **Relevance**: Efficient enough for the WavLM-scale backbone. Competitive with Nash-MTL at lower compute cost.
- **Implementation complexity**: Medium

### 22. Rotograd: Gradient Rotation for Multi-Task Learning
- **Authors**: Adrian Javaloy, Isabel Valera
- **Venue**: NeurIPS 2022
- **arXiv**: 2110.11132
- **Category**: Gradient
- **Key idea**: Decomposes task gradients into shared and task-specific components by rotating the gradient space.
- **Relevance**: Could disentangle shared acoustic information from task-specific speech information in gradient space. Theoretically elegant for the content/speaker/emotion decomposition.
- **Implementation complexity**: Medium-High

### 23. FairBranch: Mitigating Bias Transfer in Fair Multi-Task Learning
- **Authors**: Roy et al.
- **Venue**: IJCNN 2024
- **arXiv**: 2310.13746
- **Category**: Architecture + Task Imbalance
- **Key idea**: Branches model by parameter similarity → groups related tasks → resolves fairness loss gradient conflicts between groups.
- **Relevance**: The branching mechanism could group speech tasks (e.g., content+speaker in one branch, emotion in another).
- **Implementation complexity**: Medium-High

### 24. Self-Auxiliaries for Asymmetric Knowledge Transfer
- **Authors**: Graffeuille et al.
- **Venue**: arXiv 2024 (2410.15875)
- **Category**: Task Imbalance
- **Key idea**: Creates "self-auxiliary" clones of tasks to enable asymmetric transfer — helps some tasks without hurting others.
- **Relevance**: Directly addresses the asymmetric case: content task might help emotion more than vice versa. Lee et al. (2016) [97] initiated this line — this is the modern deep learning successor.
- **Implementation complexity**: Medium

### 25. MultiBalance: Multi-Objective Gradient Balancing
- **Authors**: He et al.
- **Venue**: arXiv 2024 (2411.11871)
- **Category**: Gradient + Loss
- **Key idea**: Balances per-task gradients w.r.t. shared representations (not all parameters), avoiding 70-80% computational overhead of full gradient balancing.
- **Relevance**: Computationally practical for large WavLM backbone. Balances at representation level which is where task conflicts manifest.
- **Implementation complexity**: Medium

### 26. STCH: Stochastic Convex Hull for Multi-Task Learning
- **Authors**: Xingyu Zhou, Xi Lin, Weiwen Fu, Qingfu Zhang
- **Venue**: ICML 2024
- **arXiv**: 2406.05285
- **Category**: Gradient
- **Key idea**: Optimizes within the convex hull of task gradients. Stochastic solver with convergence guarantees.
- **Relevance**: Theoretical guarantees for finding a Pareto-stationary point. Strong but complex.
- **Implementation complexity**: Medium-High

### 27. Adaptive Multi-Task Learning with Task-Specific Feature Learning
- **Authors**: Jianlong Chang et al.
- **Venue**: AAAI 2021
- **Category**: Task Relation + Architecture
- **Key idea**: Learns task-specific feature subspaces via orthogonal projections from shared representation.
- **Relevance**: Could ensure wavCSE's 2000-dim hidden representation contains orthogonal subspaces for content/speaker/emotion. Improves interpretability.
- **Implementation complexity**: Medium

### 28. MHR: Multi-Head Adapter Routing
- **Authors**: Caccia et al.
- **Venue**: NeurIPS 2023
- **Category**: Architecture
- **Key idea**: Partitions adapter parameters into heads with separate routing. Found routing's main benefit is better multi-task optimization (higher gradient alignment).
- **Relevance**: For wavCSE: the routing insights apply even without adapters. The finding that routing improves gradient alignment is directly applicable.
- **Implementation complexity**: High

---

## Tier 3: Interesting but Lower Priority

These papers are valuable for depth/breadth but not immediate priorities for wavCSE extension.

| # | Paper | Venue | Category | Why Lower Priority |
|---|-------|-------|----------|-------------------|
| 29 | **Learning to Branch for MTL** (Mao et al., ICML 2020) | ICML 2020 | Task Relation | Tree-structured architecture not natural for 3-task flat setup |
| 30 | **DSelect-k** (Hazimeh et al., NeurIPS 2021) | NeurIPS 2021 | Architecture | Designed for 100s of tasks; overkill for 3-task speech |
| 31 | **GradDrop** (Chen et al., NeurIPS 2020) | NeurIPS 2020 | Gradient | Stochastic dropping less principled than CAGrad/Nash-MTL |
| 32 | **MoCo: Momentum Conflict Resolution** (Chen et al., ICLR 2023) | ICLR 2023 | Gradient | Smoothing conflicts via momentum; CAGrad addresses root cause |
| 33 | **SDMGrad** (Hu et al., NeurIPS 2023) | NeurIPS 2023 | Gradient | Heavy theory; marginal gain over simpler methods for 3 tasks |
| 34 | **HoME** (Kuaishou, 2024) | Industry | Architecture | Industrial-scale system; overengineered for research prototype |
| 35 | **RI-PCGrad** (Meng et al., 2024) | Applied Intelligence 2024 | Gradient | Incremental PCGrad improvement; minor gain |
| 36 | **SLGrad** (Grégoire et al., 2023) | Applied Intelligence 2024 | Gradient | Sample-level weighting; orthogonal to task-level relationship focus |
| 37 | **DLBW** (2025) | Cluster Computing 2025 | Loss | Cross-entropy + AUC specific; not general MTL |
| 38 | **PIVRG** (CVPR 2025) | CVPR 2025 | Loss | Very recent; code likely unavailable |
| 39 | **EMA Loss Weighting** (Lakkapragada et al., AAAI 2024) | AAAI 2024 (Student Abstract) | Loss | Too simple; student abstract level |
| 40 | **DTRN** (Liu et al., CIKM 2023) | CIKM 2023 | Architecture | Recommender system specific |
| 41 | **Parameter Efficient Transfer Learning for Speech** (2022) | arXiv 2022 | Speech | Independent adapters not joint MTL |
| 42 | **ML-SUPERB 2.0** (Shi et al., Interspeech 2024) | Interspeech 2024 | Speech | Benchmark paper; no new MTL method |
| 43 | **IFTMTL** (2025) | SciDirect 2025 | Task Relation | Linear model focus; not deep learning |
| 44 | **MT-HAL** (Malenica et al., 2023) | arXiv 2023 | Task Relation | Nonparametric approach; doesn't integrate with deep nets |
| 45 | **MRN: Multilinear Relationship Networks** (Long et al., 2017) | NeurIPS 2017 | Task Relation | Pre-2020; foundational but already in Zhang & Yang survey |

---

## Key Surveys & Background (Read but Not Ranked)

| Paper | Venue | Why Read |
|-------|-------|----------|
| **Unleashing the Power of MTL** (Yu et al., 2024) | arXiv 2024 | Comprehensive MTL taxonomy across 3 eras; traces regularization→relationship→pre-training evolution |
| **MTL in NLP: An Overview** (Chen et al., 2024) | ACM Computing Surveys 2024 | NLP-specific architectures + benchmarks |
| **Sharing to Learn, Learning to Share** (Upadhyay et al., 2024) | IEEE Access 2024 | Meta-review connecting MTL, meta-learning, transfer learning |
| **Zhang & Yang MTL Survey** (2021) | IEEE TKDE | Already in your library; the foundational taxonomy |

---

## Summary: Recommended First Wave

For the ACL 2027 submission, focus on these 6 papers as the core contribution:

1. **Grad-TAG** (KDD 2024) — compute task affinities for speech tasks → quantitative evidence of task relationships
2. **CAGrad** (NeurIPS 2021) — principled gradient conflict resolution as the optimization backbone
3. **Nash-MTL** (ICML 2022) — fairness baseline ensuring emotion task isn't sacrificed  
4. **RLW** (TMLR 2022) — essential baseline proving your method beats random
5. **ExcessMTL** (ICML 2024) — theoretical framework for WHY certain speech task combos work
6. **Recon** (ICLR 2023) — layer-wise conflict analysis → data-driven architecture design

Together these cover: task relationship DISCOVERY (Grad-TAG), task relationship EXPLOITATION (CAGrad/Nash-MTL), theoretical UNDERSTANDING (ExcessMTL), architecture DESIGN (Recon), and rigorous VALIDATION (RLW baseline).
