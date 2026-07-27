# Expected Q&A — Fallback Answers

---

## Q1: What is a logit?

A logit is the raw, unnormalized score a model assigns to each class before applying softmax. It's a real number — can be positive, negative, any magnitude. Softmax converts logits to probabilities that sum to one, and argmax picks the highest.

Example: for a 4-class emotion task, the model might output logits `[2.1, -0.5, 0.3, 1.7]`. After softmax those become probabilities `[0.52, 0.04, 0.08, 0.36]`. Argmax picks class 0 — "happy." The logit is the intermediate representation, not the final prediction.

In equations, when you see `logit_t = W_t · h + b_t`, that's the per-class raw score vector for task t. It's what gets fed into cross-entropy loss.

---

## Q2: What is a convex combination of the K latents?

A convex combination is a weighted sum where all weights are non-negative and sum to 1. In TSM, each task `t` has a row in the structure matrix A: `[A_t1, A_t2, ..., A_tK]`. After softmax, these are all ≥ 0 and sum to 1. So task `t`'s logit is something like:

```
logit_t = 0.4 · g_1(h) + 0.35 · g_3(h) + 0.25 · g_5(h) + 0.0 · (everything else)
```

Three properties make this important. Non-negative weights means a latent can only contribute, never subtract — it can't "un-explain" a task. Sum to one means the output stays bounded — no single latent can dominate arbitrarily. And it's interpretable: you can say "task t is 40% latent 1, 35% latent 3, and 25% latent 5."

It's the same intuition as mixing paints. You have K base colors. Every task is some mixture of them. The mixture proportions tell you which base colors matter for that task. The L1 penalty then pushes most of those proportions to zero, so each task ends up using only a few latents.

---

## Q3: What specific problem of wavCSE does each architecture solve?

**GBC** solves the problem of **imbalanced bias estimation**. In the original wavCSE, each task head's bias is learned from only that task's data. Emotion recognition has ~5K samples; speaker ID has ~150K. The emotion bias has high variance simply because it sees fewer updates. The global bias coupling vector receives gradients from every training example regardless of task, so it converges with lower variance. The data-rich speaker task indirectly stabilizes the data-poor emotion task's bias estimate through the shared coupling. Reason: Zhang et al. (2010) showed that a shared global bias parameter converges faster and with higher precision than any per-class bias because it acts as a statistical hub — updated more frequently.

**TSM** solves the problem of **unknown task relatedness and negative transfer**. The original wavCSE assumes nothing about task relationships, but also provides no mechanism to discover or exploit them. If sharing the backbone hurts one task while helping another, there's no way to detect or stop it. TSM replaces independent heads with a shared latent classifier basis and a sparse structure matrix A. L1 regularization zeroes out spurious cross-task connections. The non-zero pattern of A reveals which tasks share underlying structure. Reason: Ciliberto et al. (2015) showed that sparsity on the task structure matrix prevents negative transfer by eliminating unhelpful cross-task influence — demonstrated on Animals with Attributes where training color and limb-shape attributes jointly hurt other MTL methods but SKMTL was unaffected.

**PMR** solves the problem of **no formal statistical model of task relationships**. Neither the original wavCSE, GBC, nor TSM provides a rigorous answer to "how are these tasks related?" GBC couples through a heuristic hub; TSM couples through a deterministic mixture. PMR models task parameters as draws from a multivariate Gaussian and learns the precision matrix — where a zero entry means conditional independence. This gives a proper probabilistic interpretation and produces a visualizable dependency graph. Reason: Gonçalves et al. (2016) showed that precision matrix estimation recovers meaningful domain structure purely from data — terrain clusters in landmine detection, climate connectivity patterns — without any spatial or domain metadata.

---

## Q4: Why three architectures? Why not just pick one?

They answer different questions. GBC asks "can minimal coupling help imbalanced tasks?" TSM asks "can we discover task structure from data?" PMR asks "can we model task relationships probabilistically?" Each tests a different hypothesis about what kind of coupling matters. Running all three gives you a comparative picture — you can say whether the coupling mechanism itself matters, or whether any coupling at all is sufficient.

---

## Q5: What does "conditional independence" mean in PMR?

Two tasks are conditionally independent given all other tasks if, once you know the parameters of every other task, knowing task i's parameters gives you zero additional information about task j's parameters. Any apparent correlation between them is fully explained through their relationships with the remaining tasks.

In the precision matrix Omega, Omega_ij = 0 encodes exactly this. It's stronger than correlation. Two tasks could be highly correlated (because they both correlate with a third task) but conditionally independent (no direct edge between them in the graph).

---

## Q6: What's the difference between covariance and precision?

Covariance tells you whether two tasks vary together. Precision tells you whether they have a direct relationship after accounting for all other tasks. If task A and task B both correlate with task C, the covariance matrix will show A and B as related. But the precision matrix will correctly show A and B as conditionally independent — their apparent relationship was entirely mediated through C. This is why we learn a precision matrix rather than a covariance matrix.

---

## Q7: How do you choose K for TSM?

K — the number of latent classifiers — is a hyperparameter. If K is too small, different tasks are forced to share latents even when they shouldn't, causing negative transfer. If K is too large, you lose the benefit of sharing — each task essentially gets its own private latents and the structure matrix becomes sparse but uninformative.

In practice, K is set via cross-validation or chosen based on the expected number of task clusters. With three tasks (content, speaker, emotion), K between 4 and 8 is a reasonable starting range. You can also inspect the learned A matrix — if every task uses a disjoint set of latents, K is too large; if every task uses all latents, K is too small.

---

## Q8: Why does PMR need a warmup? Doesn't that slow training?

Without warmup, Omega starts fitting structure to randomly-initialized task head parameters. Those random weights encode no meaningful task relationships, so Omega would learn spurious dependencies that then bias the rest of training. It's like trying to draw a map of task relationships before the tasks themselves have been defined.

The warmup cost is negligible — usually three epochs of standard training — and the total training time with PMR is only about five percent more than the original.

---

## Q9: Can GBC and TSM or PMR be combined?

Yes, they target different parts of the model. GBC modifies the bias term, TSM restructures the heads, and PMR adds a loss term. You could combine GBC's global bias coupling with PMR's precision matrix regularization — they're orthogonal. TSM and PMR could also theoretically be combined, though TSM replaces the independent heads that PMR's parameter matrix W is constructed from, so W would need to be built from the latent classifier parameters instead.

We're testing them separately first to isolate each paper's contribution.

---

## Q10: What happens if tasks are completely unrelated?

GBC would still couple them — that's its main weakness. The global bias connects everything, so unrelated tasks still exert some influence on each other through the shared coupling space. Whether that influence is harmful or just neutral depends on the data.

TSM would isolate them. The L1 penalty would push the structure matrix entries to zero for any latent that doesn't help a given task, so unrelated tasks would end up using disjoint sets of latent classifiers. Ciliberto demonstrated this on Animals with Attributes, where training color attributes and limb-shape attributes jointly caused negative transfer for other MTL methods but SKMTL was unaffected.

PMR would also isolate them. Unrelated tasks would get zero entries in Omega, meaning no coupling force in the trace term. They'd train essentially independently, just sharing the backbone.

---

## Q11: Why preserve wavCSE's layer selection and pooling? Why not improve those too?

Those are wavCSE's novel contributions — the recursive feature selection across transformer layers and the extensive pooling strategy comparisons. Changing them would make this a different paper. We're specifically testing whether the task-head design can be improved while keeping the representation-learning pipeline intact. It's a cleaner ablation — one variable changed at a time.

---

## Q12: How does this relate to just using different loss weights per task?

Loss weighting and task coupling solve different problems. Loss weighting addresses the optimization question: "how much should each task contribute to the gradient?" Coupling addresses the representational question: "should the parameters of these tasks be related, and if so, how?"

You could use both. Weighted loss helps with imbalanced gradient magnitudes. GBC/TSM/PMR helps with parameter-level task structure. They're complementary.

---

## Q13: Why not just use a multi-head attention over tasks instead?

You could. Multi-head attention over tasks is another valid coupling mechanism — it's essentially what TSM does but with learned attention weights instead of a fixed K latent basis and sparsity. The difference is interpretability. TSM's A matrix gives you a fixed, sparse task-latent mapping you can read after training. Attention weights vary per input and are harder to summarize into a single task structure statement. Different design goals — TSM prioritizes interpretable structure discovery over per-sample flexibility.
