# Analyzing the Impact of Multi-task Learning Strategies on Speech Task Performance

Final year project — target venue **ACL 2027**.

Systematic comparison of **parameter-based MTL** strategies for speech tasks (content, speaker, emotion, intent), built on a speech LLM backbone. Extends the [wavCSE](https://github.com/Synergy-io/wavCSE) baseline (IJCNLP 2025) by adapting MTL architectures from computer vision to speech processing.

## Structure

```
fyp/
├── docs/          Project description, paper reading list
├── weekly/        Supervisor meeting notes and presentations
├── wavCSE/        wavCSE baseline + our improvements (PMR, TSM, GBC)
└── LibMTL/        Reference MTL library
```

## Benchmarks

1. **Pass wavCSE** — parameter-based MTL must beat the feature-learning approach
2. **Pass single-task baselines** — prove multi-task training is better than isolated models

## People

- **Supervisor:** Dr. Uthayasanker Thayasivam
- **Co-supervisor:** Braveenan Sritharan (wavCSE author)
