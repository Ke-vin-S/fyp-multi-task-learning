# Member 4 Presentation Transcript

> Key Findings, Phase 2 Results, Limitations, Conclusion & Future Work

---

## Part 1: Key Findings

**Finding 1: Lower SSL layers matter more for generalization.**

When we look at the learned weights from weighted average pooling, layers 0 through 6 consistently get the highest weights. These lower layers capture general acoustic patterns — things like pitch, rhythm, basic phonetics — that every task needs. The upper layers, especially 18 through 24, get near-zero weights every round. They've become too specialized to WavLM's English pre-training and they don't transfer. We end up keeping 16 of the 25 layers, removing the nine noisiest ones, and accuracy actually *improves*. More layers is not better.

**Finding 2: Non-linear pooling beats linear pooling for layer aggregation.**

We tested ten different ways to combine the selected layers. Weighted average pooling — the standard approach in SUPERB — came sixth. Learned-norm pooling came first. The difference is that LNP applies a non-linear transformation that adapts to the statistical distribution of each layer's output. It can suppress noisy layers and amplify informative ones, which a simple weighted average cannot do. This is a broadly applicable finding — any SSL-based speech pipeline could swap WAP for LNP and likely see gains.

**Finding 3: Multi-task training produces a better representation than single-task training.**

This is the headline. One model, trained jointly on three tasks, beats three separate models each specialized for one task. The MTL model scores 98.81 on keyword spotting against the single-task SOTA of 98.55. On speaker identification, 97.59 against 95.65. On emotion recognition, 79.39 against 79.10. Joint training didn't compromise any task — it improved all of them.

---

## Part 2: Phase 2 Results

Now, Phase 1 proved the model works on the tasks it was trained on. Phase 2 asks the real question: does this embedding transfer to tasks and languages it has never seen?

We freeze the entire wavCSE model after Phase 1. We extract the 2000-dimensional embedding from the output of FC2. Then for each of twelve downstream tasks, we train an extremely simple model — usually a single linear layer — on top of that frozen embedding. These twelve tasks span classification, regression, and verification, across twenty-one languages including low-resource Indian languages.

The result: wavCSE embeddings outperform strong task-specific baselines on nine of the twelve tasks.

Let me walk through the highlights. On dialect recognition, wavCSE scores 51.27 against a baseline of 16.00. That's a threefold improvement — the embedding captures accent and dialect information remarkably well. On spoken language identification across six European languages, 99.23 against 94.30. On gender recognition, 99.84 against 99.00. On emotion recognition in Greek, 89.26 against 84.85 from emotion2vec, which is a model specifically designed for emotion.

The affective dimension tasks — valence, activation, and dominance — all show strong gains. These measure the pleasantness, energy level, and sense of control in speech. wavCSE achieves a CCC of 0.67 on valence against a baseline of 0.35. That's nearly double. This tells us the embedding captures subtle emotional nuance, not just coarse emotion categories.

Age regression also improves — mean absolute error drops from 5.97 years to 3.79 years. The embedding learns something about speaker age even though age was never a training task in Phase 1.

Now, honesty requires me to discuss the three tasks where wavCSE did not win.

Speaker identification on Kathbath, a twelve-language Indian dataset, comes in at 97.33 against 97.96. This is a marginal loss — less than one percentage point. The winning baseline uses IndicWav2Vec, which was pre-trained specifically on Indian languages. Our WavLM backbone was trained only on English. Given that mismatch, being within half a percent is actually impressive.

Speaker verification on Chinese data is a clearer loss — equal error rate of 16.87 against 11.99 from x-vector. The fixed-size embedding loses fine-grained speaker-discriminative detail that x-vector preserves. This is an inherent tension: compressing a variable-length utterance into a fixed 2000-dimensional vector inevitably discards some speaker-specific nuance.

The largest gap is intent classification — 71.00 against 99.31 from WavLM Large. The embedding captures surface-level linguistic content well enough for keyword spotting, but intent classification requires deeper semantic understanding. You need to know that "turn on the lamp" and "make it brighter" mean the same thing. The Phase 1 training tasks — keywords, speakers, emotions — never pushed the model to learn that kind of semantic abstraction.

---

## Part 3: Limitations

**First, no generative tasks.** The embedding has been evaluated only on classification, regression, and verification. We don't know whether it would help with speech recognition, speech synthesis, or text-to-speech. The features useful for discrimination might not be the features useful for generation. The authors explicitly leave this for future work.

**Second, the semantic gap is real.** Intent classification at 71 percent tells us the embedding understands what words were spoken but not what those words mean. If we want a truly general-purpose speech embedding, we need to include semantically demanding tasks during Phase 1 training. The authors acknowledge this and suggest adding intent-related training tasks in future work.

**Third, no data augmentation.** The model learns from raw audio only — by design. The authors wanted to test whether genuinely generalizable features emerge from clean training. But this means the model is not exposed to noise, reverberation, channel variation, or different recording conditions. Real-world robustness was not tested.

**Fourth, English-only pre-training.** WavLM Large was trained on 94,000 hours of English speech. Despite this, the embedding transfers to twenty-one languages — which is actually a positive result. But the two speaker-related losses, SID and SV on non-English data, suggest that a multilingual upstream model would close those gaps.

**Fifth, only equal-weighting loss was tested.** All three Phase 1 tasks contribute equally to the total loss, regardless of dataset size or task difficulty. The paper does not explore whether, for example, giving more weight to emotion recognition would improve the downstream emotion transfer while potentially hurting other tasks. This tradeoff between task balance and overall performance is exactly what our FYP will investigate.

---

## Part 4: Transition to Our FYP

So to summarize where wavCSE leaves us and where we pick up: wavCSE proved that feature-based multi-task learning with equal weighting produces a strong unified speech embedding. It established the baseline. What it did not do — and what our project will do — is ask *why* this works compared to other ways of sharing between tasks, what happens when tasks are imbalanced, and whether alternative MTL strategies like parameter sharing could do even better. We're taking the framework, the backbone, and the tasks that wavCSE established, and we're adding the systematic comparison that turns a good model into a real understanding of multi-task learning for speech.

---

## Part 5: Conclusion Slide Points

### Slide Title: Conclusion

**What wavCSE achieved:**
- Produced a fixed-size (2000-dim) unified speech embedding
- Single embedding works for both linguistic and paralinguistic tasks
- Joint training on 3 tasks (KS + SID + ER) → outperforms specialized single-task SOTA models
- Embedding transfers to 9 of 12 unseen tasks across 21 languages

**Three architectural contributions:**
- Recursive layer selection: 25 → 16 layers, 24% memory reduction, accuracy improvement
- Learned-norm pooling: non-linear layer aggregation, beats weighted average across all tasks
- Feature-based MTL framework: modular, reproducible, extensible

### Slide Title: Future Work

**From the paper:**
- Add semantically-oriented tasks during training to close the intent classification gap
- Extend beyond classification/regression/verification to generative tasks (ASR, TTS)
- Incorporate multilingual upstream pre-training for better cross-lingual speaker performance
- Explore data augmentation for real-world robustness

**From our FYP:**
- Compare feature-based vs. parameter-based MTL — which sharing strategy works best for speech?
- Analyze task imbalance effects — what happens when tasks have unequal data or difficulty?
- Implement and evaluate loss-balancing strategies (Uncertainty weighting, GradNorm, DWA)
- Build a modular framework for systematic MTL experimentation in speech
- Target: ACL 2027
