# Presentation Plan: wavCSE Paper Review

> **Meeting:** Thursday, June 26, 2026 at 7:00 AM (with supervisor)
>
> **Group Size:** 4 members
>
> **Goal:** Present a comprehensive summary of the wavCSE paper and implementation, divided equally among all members.

---

## Presentation Structure Overview

| Section | Presenter | Duration | Slides |
|---------|-----------|----------|--------|
| 1. Introduction & Motivation | Member 1 | ~6-7 min | 5-7 |
| 2. Methodology & Architecture | Member 2 | ~7-8 min | 7-9 |
| 3. Experimental Setup & Phase 1 Results | Member 3 | ~6-7 min | 5-7 |
| 4. Phase 2 Results, Conclusions & FYP Relevance | Member 4 | ~6-7 min | 5-7 |
| **Total** | | **~25-30 min** | **22-30** |

---

## Member 1: Introduction & Motivation

### Topics to Cover

#### Slide 1: Title Slide
- Paper title, authors (Braveenan Sritharan & Uthayasanker Thayasivam)
- Venue: IJCNLP-AACL 2025
- University of Moratuwa, Sri Lanka

#### Slide 2: Background — Speech Representation Learning
- Speech conveys multiple layers: linguistic content, speaker identity, emotion, intent
- Two approaches: feature engineering (MFCCs) vs. representation learning (deep learning)
- Evolution: GMMs/HMMs → Supervised DNNs → Self-Supervised Learning (SSL)
- Key SSL models: wav2vec 2.0, HuBERT, WavLM, Whisper

#### Slide 3: The Problem
- Existing SSL models produce **variable-length** representations (scale with audio duration)
- Most speech embeddings are **task-specific** (i-vector for speaker, not content)
- Virtual assistants need **multiple tasks simultaneously**: wake word, speaker ID, emotion
- **Gap:** No fixed-size unified embedding for both linguistic AND paralinguistic tasks

#### Slide 4: What is a Speech Embedding?
- Variable-length frame representations vs. Fixed-size utterance embeddings
- Frame: captures local acoustic features for a ~20ms window
- Embedding: single vector summarizing the entire utterance
- Trade-off: compactness vs. information preservation

#### Slide 5: The Proposed Solution — wavCSE
- **wavCSE** = waveform Cross-domain Speech Embedding
- Feature-based Multi-Task Learning model
- Jointly trained on 3 tasks: Keyword Spotting + Speaker ID + Emotion Recognition
- Produces one fixed-size vector that works for both linguistic and paralinguistic tasks
- Built on WavLM Large (SSL backbone)

#### Slide 6: Key Contributions (Preview)
1. Unified speech embedding for linguistic + paralinguistic tasks
2. Recursive layer selection (24% memory reduction)
3. Learned-norm pooling (outperforms weighted average)
4. SOTA on 9 of 12 downstream tasks across 21 languages

### Key Points to Emphasize
- The **practical motivation**: virtual assistants need multi-capability embeddings
- The **research gap**: no existing unified fixed-size embedding
- The **approach**: MTL forces the model to learn generalizable features

---

## Member 2: Methodology & Architecture

### Topics to Cover

#### Slide 1: Two-Phase Pipeline Overview
- **Phase 1:** Train wavCSE jointly on 3 tasks → produce unified embedding
- **Phase 2:** Freeze wavCSE → extract embeddings → evaluate on 12 unseen tasks
- Diagram: Raw Audio → Upstream → Downstream → Embedding

#### Slide 2: Architecture Diagram (THE key slide)
- Full architecture walkthrough:
  - Input audio → WavLM Large → 25 hidden layer outputs
  - Recursive Layer Selection → keep 16 layers
  - Layer-wise Concatenation → Temporal Pooling (mean)
  - FC1 (1024→512) → Layer-wise Pooling (LNP)
  - FC2 (512→2000) → Unified Embedding
  - Three classification heads: KS (12), SID (1251), ER (4)

#### Slide 3: Modification 1 — Recursive Layer Selection
- Problem: 25 layers is expensive (315M params, 1.175 GB)
- Method: Recursive Feature Elimination (RFE) approach
  1. Start with all 25, apply Weighted Average Pooling
  2. Check learned weights → remove least important layer
  3. Retrain, repeat until 1 layer remains
  4. Select subset with highest average accuracy
- Show weight distribution: lower layers dominate
- Result: 16 layers selected (0-14 + 17), 24% memory savings

#### Slide 4: Modification 2 — Layer-wise Pooling Strategy
- After layer selection, what's the best way to combine the 16 layers?
- Evaluated 10 pooling strategies: WAP, Max, Mean, Mixed, Gated, LNP, LSE, SMP, Auto, SAP
- Show comparison table
- **Winner: Learned-Norm Pooling (LNP)**
- Why LNP works: non-linear transformation adapts to statistical distribution
- KS: 98.81, SID: 97.59, ER: 79.39

#### Slide 5: Modification 3 — Feature-based MTL
- Shared backbone (projector, pooling, hidden layers)
- Task-specific heads (one linear classifier per task)
- Equal-weighting loss: `L = (L_KS + L_SID + L_ER) / 3`
- Composite dataset construction (no single dataset has all 3 annotations)
- Index pattern: `[label, -1, -1]` mechanism for per-task loss masking

#### Slide 6: Training Details
- Optimizer: AdamW, scheduler: ReduceLROnPlateau
- Batch size: 2048 (Phase 1), 64 (Phase 2)
- Regularization: L1 (1e-7), L2 (1e-5), Dropout (0.4, 0.6)
- GPU: NVIDIA Quadro RTX 6000 (30 GB)
- Hyperparameter tuning: grid search + Bayesian optimization

#### Slide 7: Pooling at Two Levels (summary)
| Level | What | Method | Why |
|-------|------|--------|-----|
| Frame (temporal) | Collapse time dim | Mean | Simple, temporal structure unchanged |
| Layer (transformer) | Aggregate across layers | LNP | Non-linear, adapts to distribution |

### Key Points to Emphasize
- The 3 modifications are **independent contributions** — each is broadly applicable
- Recursive selection is **principled** (not arbitrary pruning)
- LNP is a **finding** not an assumption — they tested 10 methods
- The index pattern is **clever engineering** for composite MTL datasets

---

## Member 3: Experimental Setup & Phase 1 Results

### Topics to Cover

#### Slide 1: Experimental Design
- **Phase 1:** Optimize wavCSE architecture on 3 training tasks
  - Datasets: Speech Commands v1 (KS), VoxCeleb1 (SID), IEMOCAP (ER)
  - Grid search + Bayesian optimization
  - Compare pooling strategies
  - Validate against SOTA baselines
- **Phase 2:** Evaluate frozen embedding on 12 new tasks

#### Slide 2: Datasets Used
- **Phase 1 Training:**
  - Google Speech Commands v1: 12 keyword classes
  - VoxCeleb1: 1,251 speakers
  - IEMOCAP: 4 emotions (neutral, happy, angry, sad)
- **Phase 2 Evaluation:** 12 tasks, 21 languages (preview table)

#### Slide 3: Recursive Layer Selection Results
- Show Figure 2: Layer-wise importance weights
  - Lower layers (0-6): high weights → general acoustic info
  - Upper layers (18-24): low weights → task-specific features
- Show Figure 3: Accuracy across elimination rounds
  - Round 1 (all 25): baseline
  - Round 10 (16 layers): **peak performance**
  - After round 15: sharp decline (too few layers)
- **Key insight:** More layers ≠ better; upper layers can hurt generalization

#### Slide 4: Pooling Strategy Comparison
- Table 2 from paper: 10 strategies compared
- Highlight top 3: LNP, LSE, Auto
- LNP wins on KS and ER, second on SID
- **Why LNP > WAP:** Non-linear > linear for layer aggregation
- Practical takeaway: don't default to weighted average

#### Slide 5: Phase 1 — Comparison with SOTA
- Table 3 from paper: wavCSE vs. task-specific + SSL baselines
- KS: 98.81 (wavCSE) vs. 98.55 (Vygon et al., task-specific SOTA)
- SID: 97.59 (wavCSE) vs. 95.65 (Hu et al., task-specific SOTA)
- ER: 79.39 (wavCSE) vs. 79.10 (Peng et al., task-specific SOTA)
- **wavCSE beats ALL baselines on ALL 3 tasks**
- Single model doing 3 tasks beats specialized single-task models!

#### Slide 6: Key Takeaways from Phase 1
1. Lower SSL layers = more general, upper = more specialized
2. 16/25 layers is the sweet spot for WavLM Large
3. LNP pooling best for layer aggregation
4. MTL with equal weighting works well for 3 diverse tasks
5. A unified model CAN beat task-specific models

### Key Points to Emphasize
- The **systematic empirical approach**: test 10 pooling methods, not just pick one
- **Efficiency + performance**: saving 24% memory while improving accuracy
- **Surprising result**: joint training beats individual training

---

## Member 4: Phase 2 Results, Conclusions & FYP Relevance

### Topics to Cover

#### Slide 1: Phase 2 Overview
- wavCSE model is **frozen** after Phase 1
- Extract the unified embedding (FC2 output)
- Train lightweight task-specific models on 12 downstream tasks
- Tasks span classification, regression, and verification
- 21 languages including low-resource Indian languages

#### Slide 2: Phase 2 — The 12 Tasks
- Show Table 1: all 12 tasks with datasets and languages
- **Linguistic tasks:** KS, SLI (Language ID), IC (Intent)
- **Paralinguistic tasks:** SID, SV, SGR (Gender), SAR (Age), SDR (Dialect), ER, VR/AR/DR (Affective dimensions)
- Metrics: Accuracy, MAE, EER, CCC

#### Slide 3: Phase 2 — Results Summary
- **9 out of 12 tasks: wavCSE > SOTA baselines**
- Biggest wins:
  - SDR (Dialect Recognition): 51.27 vs. 16.00 (3.2× improvement!)
  - VR (Valence): CCC 0.67 vs. 0.35
  - SLI (Language ID): 99.23 vs. 94.30
  - ER (Greek): 89.26 vs. 84.85
- Competitive but slightly behind:
  - SID: 97.33 vs. 97.96 (0.6% gap, baseline uses multilingual data)
  - SV: 16.87 vs. 11.99 EER (lacks fine-grained speaker cues)
  - IC: 71.00 vs. 99.31 (major gap: needs deeper semantics)

#### Slide 4: Analysis — What Worked and What Didn't
- **Strong on:** Dialect, emotion, language, gender, age — paralinguistic features transfer well
- **Weak on:** Deep semantics (intent classification), fine-grained speaker verification
- **Why the gaps?**
  - WavLM pre-trained on English only → limits multilingual speaker nuance
  - Training tasks (KS, SID, ER) don't include semantic tasks → no pressure to learn deep meaning
  - Fixed-size vector may lose fine-grained speaker-discriminative info

#### Slide 5: Contributions Recap
1. **Unified speech embedding** — first of its kind, works across linguistic + paralinguistic
2. **Recursive layer selection** — 24% memory savings, general method for any SSL pipeline
3. **Learned-norm pooling** — non-linear layer aggregation > weighted average
4. **Comprehensive evaluation** — SOTA on 9/12 tasks, 21 languages
5. **Modular, reproducible framework**

#### Slide 6: Limitations
- Only classification, regression, verification (no generation: ASR, TTS)
- No data augmentation (by design)
- English-only upstream (WavLM) → cross-lingual by transfer not design
- Semantic gap remains (intent classification)
- Equal-weighting only (no task imbalance handling)

#### Slide 7: Relevance to Our FYP
- **wavCSE is our baseline** — we extend it in 3 directions:
  1. **Compare feature-based vs. parameter-based MTL** — wavCSE only does feature-based
  2. **Study task imbalance** — wavCSE only uses equal weighting
  3. **Build modular framework** — systematic analysis of MTL design choices
- Our goal: understand **WHY** different strategies work, not just **THAT** they work
- Target: ACL 2027

### Key Points to Emphasize
- **Generalization is real**: trained on 3 tasks (English), works on 12 tasks (21 languages)
- **Honest about limitations**: 3 tasks where it doesn't win, with clear hypotheses
- **Clear path forward**: our FYP addresses the limitations directly

---

## Workload Distribution Summary

```
Member 1  ████████████████████████████  25%  Motivation & Background
Member 2  ████████████████████████████  25%  Methodology & Architecture
Member 3  ████████████████████████████  25%  Phase 1 Experiments & Results
Member 4  ████████████████████████████  25%  Phase 2 Results & FYP Connection
```

Each member:
- Prepares 5-7 slides
- Presents for 6-8 minutes
- Should be able to answer questions on their section

---

## Preparation Checklist

### For All Members
- [ ] Read the full paper (PDF at `fyp/2025.ijcnlp-long.48.pdf`)
- [ ] Review the codebase (`fyp/wavCSE/`)
- [ ] Read the docs in `fyp/docs/`
- [ ] Prepare your slides
- [ ] Practice your section (~7 min target)
- [ ] Prepare 2-3 backup slides for likely questions

### For Member 1 (Motivation)
- [ ] Understand the speech representation learning landscape
- [ ] Know the key prior work: wav2vec 2.0, HuBERT, WavLM, SUPERB
- [ ] Practice explaining "why unified embeddings matter" in 1 minute

### For Member 2 (Architecture)
- [ ] Trace through the code: `downstream_model.py`, both `pooling.py` files
- [ ] Understand the index pattern mechanism in `load_embedding.py`
- [ ] Practice drawing the architecture diagram from memory

### For Member 3 (Phase 1 Results)
- [ ] Understand Figures 2 & 3 and Table 2 & 3
- [ ] Be ready to explain the recursive elimination process
- [ ] Know why LNP beats WAP technically

### For Member 4 (Phase 2 Results + FYP)
- [ ] Understand all 12 downstream tasks
- [ ] Be ready to explain the 3 "failure" cases
- [ ] Connect clearly to our FYP project description
- [ ] Know the 3 reference papers for our FYP

---

## Suggested Slide Templates

### Architecture Overview (Member 2, most complex slide)
```
┌─────────────────────────────────────────┐
│           Input Audio (.wav)             │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│     WavLM Large (Pre-trained, Frozen)    │
│   CNN Encoder → Transformer ×24         │
│   Output: 25 hidden layer representations│
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│   ★ Recursive Layer Selection           │
│   25 layers → 16 layers (0-14 + 17)     │
│   24% memory reduction                  │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│   Temporal Pooling (mean) + FC1 (512)   │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│   ★ Layer-wise Pooling (Learned-Norm)   │
│   + FC2 (2000) = Unified Embedding      │
└─────────────────┬───────────────────────┘
                  ▼
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│ KS Head │ │SID Head │ │ ER Head │
│  (12)   │ │ (1251)  │ │  (4)    │
└─────────┘ └─────────┘ └─────────┘
```

---

## Notes for the Presentation

1. **Keep technical depth accessible** — the supervisor knows the field but group members should understand everything
2. **Use the figures from the paper** — Figure 1 (architecture), Figure 2 (weights), Figure 3 (elimination)
3. **Highlight numbers** — 9/12 tasks, 24% savings, 21 languages, 3 contributions
4. **Connect to FYP** — make it clear why we're studying THIS paper
5. **Anticipate questions:**
   - "Why only 16 layers, not 15 or 17?"
   - "How does this compare to Whisper embeddings?"
   - "Could you add more tasks to the MTL training?"
   - "What's the computational cost of recursive selection?"
