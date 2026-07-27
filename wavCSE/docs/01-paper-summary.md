# wavCSE Paper Summary

> **Full Title:** wavCSE: Learning Fixed-size Unified Speech Embeddings via Feature-based Multi-Task Learning
>
> **Authors:** Braveenan Sritharan, Uthayasanker Thayasivam (University of Moratuwa, Sri Lanka)
>
> **Venue:** IJCNLP-AACL 2025 (Pages 879–887)
>
> **PDF:** [ACL Anthology](https://aclanthology.org/2025.ijcnlp-long.48.pdf)
>
> **Code:** [github.com/aaivu/wavCSE](https://github.com/aaivu/wavCSE)

---

## 1. Motivation

### 1.1 The Problem

Modern speech applications (virtual assistants, voice interfaces) require models that can handle **multiple tasks simultaneously**: keyword spotting ("wake word"), speaker identification (who is speaking), emotion recognition (how they're feeling), and intent classification (what they want). However:

- **Existing speech embeddings are task-specific:** Speaker embeddings (i-vector, d-vector, x-vector) work for speaker verification but not for content or emotion tasks.
- **Variable-length representations are inefficient:** SSL models produce sequences of vectors that scale with audio duration — hard to store on edge devices or transmit in real-time.
- **No unified embedding exists:** A single fixed-size vector that works for both linguistic (content, intent) and paralinguistic (speaker, emotion) tasks is needed.

### 1.2 The Proposed Solution

**wavCSE** (waveform Cross-domain Speech Embedding) is a **feature-based multi-task learning (MTL)** model that produces a **fixed-size unified speech embedding** suitable for both linguistic and paralinguistic tasks.

**Key idea:** Jointly train on keyword spotting (linguistic) + speaker identification (paralinguistic) + emotion recognition (paralinguistic) so the shared representation captures all three types of information.

---

## 2. Methodology

### 2.1 High-Level Architecture (Two-Phase Pipeline)

The framework follows the **SUPERB benchmark** structure with three key modifications:

```
Raw Audio → [Upstream: Pre-trained WavLM Large] → Frame-level Features
    → [Recursive Layer Selection: keep 16/25 layers]
    → [Temporal Pooling (mean)] → [FC1: 1024→512]
    → [Layer-wise Pooling (learned-norm)] → [FC2: 512→2000]
    → Unified Embedding (fixed-size vector)
    → [Task-specific heads: KS | SID | ER]
```

### 2.2 Three Key Architectural Modifications

#### Modification 1: Recursive Layer Selection

- **Problem:** WavLM Large produces 25 hidden layer outputs. Using all 25 → high dimensionality and computational cost.
- **Approach:** Inspired by Recursive Feature Elimination (RFE):
  1. Start with all 25 layers, apply Weighted Average Pooling (WAP)
  2. Use learned weights to identify the least important layer
  3. Remove it, retrain the model
  4. Repeat until only 1 layer remains
  5. From all intermediate subsets, select the one with highest average accuracy
- **Result:** Selected **16 layers** (layers 0–14 + layer 17), reducing WavLM parameters from **315M → 240M** and memory from **1.175 GB → 0.894 GB** (24% reduction).
- **Key insight:** Lower layers carry more weight (general acoustic info), upper layers are more specialized and contribute less.

#### Modification 2: Refined Layer-wise Pooling

- **Problem:** After reducing from 25 to 16 layers, WAP may not be optimal for aggregating the remaining layers.
- **Approach:** Evaluated **10 pooling strategies** on KS, SID, ER:
  - Weighted Average, Max, Mean, Mixed, Gated, **Learned-Norm**, Log-Sum-Exp, Smooth-Maximum, Auto, Self-Attention
- **Result:** **Learned-Norm Pooling (LNP)** achieved highest average accuracy (KS: 98.81, SID: 97.59, ER: 79.39)
- **Why:** LNP applies a non-linear transformation that adapts to the statistical distribution of layer outputs, better than linear WAP.

#### Modification 3: Feature-based Multi-Task Learning

- **Shared layers:** Projector (Linear 1024→512), Layer Pooling, Hidden Layer (Linear 512→2000), Dropouts
- **Task-specific heads:** One linear classifier per task (KS: 12 classes, SID: 1251 classes, ER: 4 classes)
- **Loss:** Equal-weighting — `L_total = (L_KS + L_SID + L_ER) / 3`
- **Composite dataset:** Merged Google Speech Commands (KS) + VoxCeleb1 (SID) + IEMOCAP (ER) since no single dataset has joint annotations
- **Index pattern mechanism:** Each sample has labels for all 3 tasks, but only the relevant position has the actual label (others = -1, ignored in loss)

### 2.3 Pooling at Two Levels

| Level | Purpose | Method |
|-------|---------|--------|
| **Frame Pooling** (temporal dim) | Collapse variable-length frames → fixed vector | Mean pooling |
| **Layer Pooling** (layer dim) | Aggregate selected transformer layers | Learned-Norm Pooling |

---

## 3. Experimental Setup

### 3.1 Phase 1: Model Development & Training

- **Training tasks:** KS (Speech Commands v1), SID (VoxCeleb1), ER (IEMOCAP)
- **Upstream model:** WavLM Large (pre-trained, frozen during downstream training? Actually the paper describes extracting embeddings from WavLM, then training the downstream MTL model)
- **Optimization:** Grid search for batch size (2048) and learning rate (0.0025); Bayesian optimization for layer dimensions (FC1: 512, FC2: 2000) and regularization (L1: 1e-7, L2: 1e-5)
- **GPU:** NVIDIA Quadro RTX 6000 (30 GB)
- **Optimizer:** AdamW with ReduceLROnPlateau scheduler

### 3.2 Phase 2: Embedding Evaluation

After Phase 1 training, the wavCSE model is **frozen**. The unified embedding (output of FC2, before task heads) is extracted and used to train lightweight task-specific models on **12 downstream tasks** across **21 languages**:

| # | Task | Type | Dataset | Language(s) | Metric |
|---|------|------|---------|-------------|--------|
| 1 | Keyword Spotting (KS) | Classification | Football Keyword | fa | Accuracy |
| 2 | Language Identification (SLI) | Classification | VoxForge | de,en,es,fr,it,ru | Accuracy |
| 3 | Speaker Identification (SID) | Classification | Kathbath | 12 Indian langs | Accuracy |
| 4 | Speaker Verification (SV) | Verification | CNCeleb v1 | zh | EER |
| 5 | Gender Recognition (SGR) | Classification | TIMIT | en | Accuracy |
| 6 | Age Recognition (SAR) | Regression | TIMIT | en | MAE |
| 7 | Dialect Recognition (SDR) | Classification | TIMIT | en | Accuracy |
| 8 | Emotion Recognition (ER) | Classification | AESDD | el | Accuracy |
| 9 | Valence Recognition (VR) | Regression | IEMOCAP | en | CCC |
| 10 | Activation Recognition (AR) | Regression | IEMOCAP | en | CCC |
| 11 | Dominance Recognition (DR) | Regression | IEMOCAP | en | CCC |
| 12 | Intent Classification (IC) | Classification | Fluent Speech Commands | en | Accuracy |

---

## 4. Results

### 4.1 Phase 1: Model Optimization Results

**Layer Selection (Figure 3):** Highest average accuracy achieved at round 10 (9 layers eliminated), selecting layers 0–14 + 17.

**Pooling Comparison (Table 2):**

| Pooling Method | KS | SID | ER |
|---|---|---|---|
| Weighted Average | 98.32 | 97.08 | 77.58 |
| **Learned-Norm** | **98.81** | **97.59** | **79.39** |
| Log-Sum-Exp | 98.52 | 97.89 | 77.22 |
| Auto | 98.45 | 97.41 | 78.12 |

**Final wavCSE vs. Baselines (Table 3):**

| Model | KS | SID | ER |
|-------|-----|------|-----|
| Task-specific SOTA | 98.55 | 95.65 | 79.10 |
| wav2vec 2.0 Large | 96.66 | 86.14 | 65.64 |
| HuBERT Large | 95.29 | 90.33 | 67.62 |
| WavLM Large | 97.86 | 95.49 | 70.62 |
| **wavCSE** | **98.81** | **97.59** | **79.39** |

wavCSE **outperforms all baselines** including task-specific SOTA models on all three training tasks.

### 4.2 Phase 2: Cross-Domain Generalization

**Key result:** wavCSE-based embeddings **outperform SOTA baselines on 9 of 12 tasks**.

**Strong performance:**
- SDR (Dialect): 51.27 vs. 16.00 baseline — largest gain
- VR (Valence): CCC 0.67 vs. 0.35 baseline
- SLI (Language ID): 99.23 vs. 94.30 baseline
- SGR (Gender): 99.84 vs. 99.00 baseline
- ER (AESDD): 89.26 vs. 84.85 baseline
- SAR (Age): MAE 3.79 vs. 5.97 baseline

**Competitive but not best:**
- SID: 97.33 vs. 97.96 (marginal, baseline used multilingual data)
- SV (Speaker Verification): EER 16.87 vs. 11.99 x-vector
- IC (Intent): 71.00 vs. 99.31 WavLM — largest gap, needs deeper semantics

### 4.3 Key Findings

1. **Lower layers carry more transferable information** — the recursive elimination shows upper layers contribute less to generalization
2. **Non-linear pooling (LNP) outperforms linear (WAP)** for aggregating across transformer layers
3. **MTL with equal weighting** is effective for learning a unified representation
4. **Strong linguistic + paralinguistic transfer** — the embedding works across both domains
5. **Weakness in semantic tasks** — intent classification needs deeper semantic abstraction

---

## 5. Contributions Summary

1. **Fixed-size unified speech embedding** — single vector that works for both linguistic and paralinguistic tasks
2. **Recursive layer selection** — reduces upstream model size by 24% while improving performance (broadly applicable to any SSL pipeline)
3. **Learned-norm pooling** — non-linear layer aggregation that outperforms weighted average
4. **Comprehensive evaluation** — 12 tasks across 21 languages, outperforming SOTA on 9
5. **Modular framework (wavCSE)** — reproducible, extensible two-stage pipeline

---

## 6. Limitations

- **No generative tasks:** Only classification, regression, verification; no TTS or ASR
- **No data augmentation:** Intentionally avoided to learn directly from raw audio
- **English-only upstream model:** WavLM Large pre-trained on English, yet embeddings transfer to 21 languages
- **Semantic gap:** Intent classification performance (~71%) shows room for improvement
- **Speaker verification gap:** Lacks fine-grained speaker-discriminative cues for verification

---

## 7. Relevance to Our FYP

Our FYP builds directly on wavCSE as the next step:

| wavCSE (This Paper) | Our FYP |
|---------------------|---------|
| Feature-based MTL only | Compare feature-based vs. parameter-based MTL |
| Equal-weighting loss only | Explore task imbalance + mitigation strategies |
| Fixed pooling strategies | Systematic comparison of sharing strategies |
| KS + SID + ER (3 tasks) | Content + Speaker + Emotion tasks |
| Single SSL backbone (WavLM) | Same speech LLM backbone for consistency |

**Reference papers mentioned for our FYP:**
- Sritharan & Thayasivam (2025) — wavCSE (this paper, the baseline we extend)
- Zhang & Yang (2021) — MTL survey (theoretical foundation for MTL strategies)
- Lin & Zhang (2023) — LibMTL (Python MTL library, reference for our library design)
