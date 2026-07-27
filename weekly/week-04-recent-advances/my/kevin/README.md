# Recent Task Relationship Learning Research for wavCSE Extension

> Consolidated research on post-2020 MTL advances applicable to speech multi-task learning.
> Research date: 2026-07-19 | Target: ACL 2027

## Purpose

This directory contains the results of an extensive literature search for recent (post-2020) papers on task relationship learning and related MTL techniques that could extend the wavCSE framework. wavCSE currently uses a frozen WavLM backbone with shared hidden layers and per-task classification heads for content, speaker, and emotion tasks — with simple equal-weight loss averaging.

**North star**: Add SOTA task relationship learning extensions to wavCSE.

## Quick-Lookup Table

| # | Paper | Venue | Category | Tier | Complexity | Key wavCSE Application |
|---|-------|-------|----------|------|------------|----------------------|
| 1 | Grad-TAG (Li et al.) | KDD 2024 | Task Relation + Gradient | T1 | Low-Med | Compute task affinities between content/speaker/emotion |
| 2 | TAG (Fifty et al.) | NeurIPS 2021 | Task Relation | T1 | Low | Gradient-based affinity from single training run |
| 3 | Which Tasks Together (Standley et al.) | ICML 2020 | Task Relation | T1 | Low | Systematic pairwise task grouping methodology |
| 4 | CAGrad (Liu et al.) | NeurIPS 2021 | Gradient | T1 | Med | Min-max gradient optimization for balanced updates |
| 5 | Nash-MTL (Navon et al.) | ICML 2022 | Gradient | T1 | Med | Game-theoretic fair gradient bargaining |
| 6 | Recon (ICLR 2023) | ICLR 2023 | Gradient + Arch | T1 | Med-High | Layer-wise conflict scores → auto architecture design |
| 7 | AdaShare (Sun et al.) | CVPR 2020 | Task Relation + Arch | T1 | Med | Learned binary masks over WavLM layers per task |
| 8 | ExcessMTL (Ye et al.) | ICML 2024 | Gradient + Theory | T1 | Med | Excess risk framework for understanding task combos |
| 9 | PCGrad (Yu et al.) | NeurIPS 2020 | Gradient | T1 | Low | Baseline gradient surgery (in LibMTL) |
| 10 | RLW (Lin et al.) | TMLR 2022 | Gradient | T1 | Trivial | Random loss weighting baseline |
| 11 | MTI-Net (Vandenhende et al.) | ECCV 2020 | Task Relation + Arch | T1 | Med | Cross-task attention between speech task heads |
| 12 | FairGrad (Baek et al.) | ICML 2024 | Gradient | T1 | Med | Prevents any task degrading below threshold |
| 13 | MetaLink (Cao et al.) | ICLR 2022 | Task Relation | T2 | High | KG-based task relationship modeling |
| 14 | MIRTL (Chang et al.) | KBS 2024 | Task Relation | T2 | Med-High | Decomposition + pairwise knowledge contribution |
| 15 | Output-Level Task Relatedness (Xiao et al.) | arXiv 2024 | Task Relation | T2 | Med | Feedback mechanism between task outputs |
| 16 | Task Relationship Matrix (Kanakis et al.) | ECCV 2020 | Task Relation | T2 | Low-Med | Learned correlation matrix in classifier heads |
| 17 | TaskExpert (Ye & Xu) | ICCV 2023 | Architecture | T2 | High | Dynamic MoE with task-specific gating |
| 18 | Aligned-MTL (Chen et al.) | CVPR 2023 | Gradient | T2 | Med | Active gradient alignment beyond PCGrad |
| 19 | IMTL (Liu et al.) | ICLR 2021 | Gradient | T2 | Low | Gradient magnitude normalization for imbalance |
| 20 | GradVac (Wang et al.) | NeurIPS 2021 | Gradient | T2 | Med | Target gradient similarity maintenance |
| 21 | FAMO (Liu et al.) | NeurIPS 2023 | Gradient | T2 | Med | Fast dual descent weight adaptation |
| 22 | Rotograd (Javaloy & Valera) | NeurIPS 2022 | Gradient | T2 | Med-High | Gradient space rotation for task decomposition |
| 23 | FairBranch (Roy et al.) | IJCNN 2024 | Architecture | T2 | Med-High | Parameter similarity-based task branching |
| 24 | Self-Auxiliaries (Graffeuille et al.) | arXiv 2024 | Task Imbalance | T2 | Med | Asymmetric knowledge transfer via task clones |
| 25 | MultiBalance (He et al.) | arXiv 2024 | Gradient + Loss | T2 | Med | Representation-level gradient balancing |
| 26 | STCH (Zhou et al.) | ICML 2024 | Gradient | T2 | Med-High | Stochastic convex hull optimization |
| 27 | Task-Specific Feature Learning (Chang et al.) | AAAI 2021 | Architecture | T2 | Med | Orthogonal task subspaces |
| 28 | MHR (Caccia et al.) | NeurIPS 2023 | Architecture | T2 | High | Multi-head adapter routing for gradient alignment |
| 29-45 | [See RANKINGS.md Tier 3] | Various | Various | T3 | — | Lower priority for immediate wavCSE extension |

## Files in This Directory

```
recent/
├── README.md              # This file — overview and quick-lookup
├── RANKINGS.md            # Tiered ranking with full justification for every paper
├── USE_CASE_MATRIX.md     # Cross-reference: which paper → which wavCSE component
├── summaries/             # Per-paper detailed summaries (Tier 1 papers)
│   ├── 01_Grad-TAG_KDD2024.md
│   ├── 02_TAG_NeurIPS2021.md
│   ├── ...
└── papers/                # Downloaded PDFs (or download list for manual)
    └── DOWNLOAD_LIST.md   # Papers needing manual download
```

## Search Methodology

### Vectors Searched
1. **Task Relationship Learning** — MTRL successors, task affinity, task covariance, task structure
2. **Gradient-Based Methods** — PCGrad, CAGrad, Nash-MTL, gradient surgery, conflict resolution
3. **Architecture-Based** — Soft sharing, task routing, cross-task attention, MoE gating
4. **Loss Weighting** — Dynamic balancing, uncertainty weighting, Pareto methods
5. **Speech-Specific MTL** — SUPERB, WavLM, HuBERT, speech SSL multi-task
6. **Task Imbalance/Negative Transfer** — Fairness, asymmetric transfer, outlier tasks

### Sources
- arxiv.org, paperswithcode.com, Semantic Scholar
- Top venues: ICML, NeurIPS, ICLR, AAAI, CVPR, ICCV, ECCV, ACL, EMNLP, Interspeech, ICASSP, JMLR, IEEE TKDE, TPAMI, ACM Computing Surveys

### Time Range
- Primary: 2020–2026
- Secondary: Pre-2020 seminal works not covered in Zhang & Yang (2021)

## Key Research Gaps Identified

1. **No existing work applies task relationship learning to speech MTL with content/speaker/emotion tasks.** wavCSE represents an open opportunity.

2. **Gradient surgery methods (PCGrad, CAGrad, Nash-MTL) have not been systematically evaluated on speech tasks.** Direct experimental contribution.

3. **Task affinity discovery via gradient signals is underexplored for speech** — Grad-TAG + TAG applied to content/speaker/emotion could be a novel contribution.

4. **The speech MTL community lacks interpretable task relationship analysis** — learned task covariance/affinity matrices would be a distinctive contribution for ACL 2027.

## Recommended Implementation Order

1. **RLW** (1 day) — Trivial baseline, essential for rigorous experiments
2. **Grad-TAG + TAG** (1 week) — Compute and visualize speech task affinities
3. **PCGrad** (2 days) — Simple gradient baseline (already in LibMTL)
4. **CAGrad** (1 week) — Principled gradient optimization  
5. **Nash-MTL** (1 week) — Fairness-comparison method
6. **Recon** (2 weeks) — Layer-wise conflict analysis and architecture redesign
7. **ExcessMTL** (1 week) — Theoretical analysis framework for paper
