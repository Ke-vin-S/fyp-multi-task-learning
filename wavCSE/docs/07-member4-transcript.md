# Member 4 Presentation Transcript

> Slides: 19 (Phase 2 Results) → 09 (Key Findings) → 20 (Limitations) → 28 (Conclusion)
>
> Note: Experimental setup (datasets, hyperparameters, Phase 1 pipeline) is covered by Member 3.

---

## Slide 19: Phase 2 Results

"Phase 2: freeze wavCSE and evaluate the extracted fixed-size embedding on 12 downstream tasks across classification, regression, and verification — 21 languages. As you can see, wavCSE outperforms SOTA baselines on **9 out of 12 tasks**."

**Key wins:**

"Across classification tasks — keyword spotting, language identification, gender recognition — wavCSE improves on every baseline. Age regression is particularly notable: MAE drops to 3.79 from 5.97, even though age was never a training task in Phase 1."

"Dialect recognition is the standout: **51.27 against 16.00** of the baseline — and that is a three fold improvement compared to the baseline.
Dialect sits at the intersection of phonetic content and speaker characteristics, and wavCSE was trained on both."

"Emotion recognition in Greek beats emotion2vec — ; here specialized emotion model was beaten using a frozen general embedding."

"The affective dimensions such as valence, activation, dominance all improve, with valence nearly doubling the baseline: **CCC 0.67 against 0.35.**"

**The three losses:**

"Speaker identification on — twelve Indian languages — 97.33 against 97.96. Marginal, and the winning baseline was pre-trained on Indian languages, while our WavLM backbone is English-only."
uses IndicWav2Vec

"Speaker verification on Chinese data: EER 16.87 against x-vector's 11.99.
Fixed-size embeddings lose fine-grained speaker-discriminative cues."

"Intent classification: **71.00 against WavLM's 99.31** — the largest gap.
even though Keywords need surface-level word recognition, intent needs semantic understanding. The Phase 1 tasks never demanded that." so that is very much justified

---

## Slide 09: Key Findings

**1. Lower layers carry more transferable information.** first six layers consistently get the highest weights. Upper layers were always eliminated early. Keeping 16 of 25 layers improves accuracy —  that shows more layers is not better.

**2. Non-linear pooling (LNP) outperforms linear (WAP).** there was Ten strategies tested. Weighted average pooling  - which was the SUPERB standard — came sixth. Learned-norm pooling came first. LNP adapts non-linearly to each layer's distribution, which WAP cannot do. Applicable to any SSL pipeline.

**3. MTL with equal weighting works.** Joint training on three tasks beats specialized single-task SOTA on all three, and transfers to 9 of 12 unseen tasks. Equal weighting — each task contributes equally regardless of dataset size — is simple and effective here.

**4. Strong linguistic and paralinguistic transfer.** The embedding handles both domains — keywords and language ID on one side, emotion, gender, age, and dialect on the other — across 21 languages.

**5. Weakness in semantic tasks.** Intent classification at 71% shows the embedding captures *what* words were spoken but not *what they mean*. Closing this gap requires semantically-oriented training tasks.

---

## Slide 20: Analysis of Limitations

**No generative tasks.** Only classification, regression, and verification were evaluated. No TTS, no speech recognition.
here The features for discrimination may not transfer to generation.

**Speaker verification gap.**
The fixed-size embedding lacks fine-grained speaker cues that dedicated embeddings like x-vector preserve.
Compression inevitably loses some nuance.

**Semantic gap.** Intent classification at 71% again shows room for improvement. Phase 1 training tasks — keywords, speakers, emotions — don't require deep semantic understanding.

**English-centric pre-training.**
WavLM trained on English speech. Despite this, the embedding transfers to 21 languages — this is a positive result.
But the loss  on non-English speaker verification  suggests multilingual pre-training would help.

---

## Slide 28: Conclusion

- Fixed-size 2000-dim unified embedding — vector worked  for linguistic AND paralinguistic tasks
- Joint MTL training  on 3 tasks (KS + SID + ER) beats single-task SOTA on all three
- Embedding transfers to **9 of 12 unseen tasks** across **21 languages**

**Three reusable contributions:**

- Recursive layer selection: where we reduced 25→16 layers, uses less memory **and** shows better accuracy
- Learned-norm pooling: non-linear > linear for layer aggregation
- Modular, reproducible two-stage framework

wavCSE established the baseline for feature-based MTL on speech. Our FYP takes the next step — comparing feature-based vs. parameter-based MTL, studying task imbalance, and building a systematic framework for understanding how MTL design choices affect each task.
