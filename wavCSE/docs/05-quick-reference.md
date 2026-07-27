# Quick Reference Card — wavCSE

## Acronyms

| Acronym | Full Form |
|---------|-----------|
| wavCSE | waveform Cross-domain Speech Embedding |
| MTL | Multi-Task Learning |
| SSL | Self-Supervised Learning |
| KS | Keyword Spotting |
| SID / SI | Speaker Identification |
| ER | Emotion Recognition |
| IC | Intent Classification |
| SLI | Spoken Language Identification |
| SV | Speaker Verification |
| SGR | Speaker Gender Recognition |
| SAR | Speaker Age Recognition |
| SDR | Speaker Dialect Recognition |
| VR/AR/DR | Valence/Activation/Dominance Recognition |
| LNP | Learned-Norm Pooling |
| WAP | Weighted Average Pooling |
| RFE | Recursive Feature Elimination |
| SOTA | State Of The Art |
| FC | Fully Connected (Linear) layer |
| SUPERB | Speech processing Universal PERformance Benchmark |
| PLDA | Probabilistic Linear Discriminant Analysis |
| EER | Equal Error Rate |
| CCC | Concordance Correlation Coefficient |
| MAE | Mean Absolute Error |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| WavLM Large parameters | 315M (full), 240M (after layer selection) |
| WavLM Large memory | 1.175 GB → 0.894 GB (24% reduction) |
| Transformer layers (WavLM Large) | 24 + CNN input = 25 hidden outputs |
| Layers selected by wavCSE | 16 (layers 0–14 + 17) |
| Selected layer indices | 6,1,0,3,2,5,4,7,8,9,10,11,12,17,14,13 |
| Embedding dimension (after FC1) | 512 |
| Unified embedding dimension (after FC2) | 2000 |
| Training tasks (Phase 1) | 3 (KS, SID, ER) |
| Evaluation tasks (Phase 2) | 12 |
| Languages covered | 21 |
| Pooling strategies evaluated | 10 |
| Best pooling strategy | Learned-Norm (LNP) |

---

## Phase 1: Final Results

| Task | wavCSE | Previous SOTA | SSL Baselines (best) |
|------|--------|---------------|---------------------|
| KS | **98.81** | 98.55 (Vygon 2021) | 97.86 (WavLM) |
| SID | **97.59** | 95.65 (Hu 2023) | 95.49 (WavLM) |
| ER | **79.39** | 79.10 (Peng 2021) | 70.62 (WavLM) |

---

## Phase 2: Performance on 12 Tasks

| Task | Type | wavCSE | Best Baseline | Win? |
|------|------|--------|---------------|------|
| KS (fa) | Classification | 96.46 | 95.88 (ResNet) | ✅ |
| SLI (6 langs) | Classification | 99.23 | 94.30 (2D ConvNet) | ✅ |
| SID (12 langs) | Classification | 97.33 | 97.96 (Sritharan) | ❌ (marginal) |
| SV (zh) | Verification | 16.87 EER | 11.99 (x-vector) | ❌ |
| SGR (en) | Classification | 99.84 | 99.00 (LSTM) | ✅ |
| SAR (en) | Regression | 3.79 MAE | 5.97 (LSTM) | ✅ |
| SDR (en) | Classification | 51.27 | 16.00 (LSTM) | ✅ (3.2×!) |
| ER (el) | Classification | 89.26 | 84.85 (emotion2vec) | ✅ |
| VR (en) | Regression | 0.67 CCC | 0.35 (CNN1D) | ✅ |
| AR (en) | Regression | 0.68 CCC | 0.67 (LSTM) | ✅ |
| DR (en) | Regression | 0.59 CCC | 0.53 (LSTM) | ✅ |
| IC (en) | Classification | 71.00 | 99.31 (WavLM) | ❌ (large gap) |

**Summary: 9 wins, 3 not-best (SID marginal, SV, IC)**

---

## Pooling Strategies Ranked

| Rank | Strategy | Avg Accuracy | Type |
|------|----------|-------------|------|
| 1 | **Learned-Norm (LNP)** | **91.93** | Non-linear |
| 2 | Log-Sum-Exp (LSE) | 91.21 | Non-linear |
| 3 | Auto | 91.33 | Learned parametric |
| 4 | Mixed | 91.16 | Hybrid |
| 5 | Gated | 91.01 | Learned gating |
| 6 | Weighted Average (WAP) | 90.99 | Linear (baseline) |
| 7 | Smooth-Maximum (SMP) | 91.16 | Non-linear |
| 8 | Max | 90.62 | Non-parametric |
| 9 | Self-Attention (SAP) | 90.56 | Attention-based |
| 10 | Mean | 90.43 | Non-parametric |

---

## Datasets Quick Reference

| Dataset | Task(s) | # Classes | Language | Phase |
|---------|---------|-----------|----------|-------|
| Google Speech Commands v1 | KS | 12 | en | 1 (train) |
| VoxCeleb1 | SID | 1,251 | en | 1 (train) |
| IEMOCAP | ER | 4 | en | 1 (train), 2 (VR/AR/DR) |
| Football Keyword | KS | — | fa | 2 (eval) |
| VoxForge | SLI | 6 | de,en,es,fr,it,ru | 2 (eval) |
| Kathbath | SID | — | 12 Indian langs | 2 (eval) |
| CNCeleb v1 | SV | — | zh | 2 (eval) |
| TIMIT | SGR, SAR, SDR | — | en | 2 (eval) |
| AESDD | ER | — | el | 2 (eval) |
| Fluent Speech Commands | IC | 31 | en | 2 (eval) |

---

## Paper's Three Key Contributions

```
1. RECURSIVE LAYER SELECTION
   └── Reduces WavLM from 25→16 layers (24% memory)
   └── Method: iteratively remove least important layer
   └── Applicable to any SSL pipeline

2. LEARNED-NORM POOLING (LNP)
   └── Non-linear layer aggregation
   └── Outperforms weighted average pooling
   └── Best among 10 strategies tested

3. UNIFIED SPEECH EMBEDDING
   └── Single fixed-size vector (2000-dim)
   └── Works for linguistic + paralinguistic tasks
   └── SOTA on 9/12 downstream tasks
```
