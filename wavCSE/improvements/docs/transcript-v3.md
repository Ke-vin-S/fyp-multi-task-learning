# wavCSE Architectural Improvements — Transcript v3

**Natural spoken version. Three segments, ~2 min each.**

---

## Slides 3-4 — Global Bias Coupling (GBC)

**Paper:** Zhang et al. 2010 — BOMC

"So let's start with Global Bias Coupling, which is based on Zhang et al. from 2010.

The integration is really simple. You leave the entire forward path alone — backbone, projector, hidden layer, weight matrices, all of it. The only thing that changes is the bias term. You add a global coupling vector and project it into each task's output space through these small per-task layers. That's it. Drop-in, no structural surgery.

Why does this work? Zhang noticed something interesting about multi-label classification. They had this parameter called a global bias, and because it was shared across all classes, it got a gradient update from every single training example. Every sample, regardless of which class it belonged to. That means it converged faster and with much lower variance than any per-class parameter that only saw its own examples.

We adapted that idea from multi-label to multi-task. Instead of one shared scalar, we use one shared vector. And the formula for each task's logit becomes: weight matrix times the hidden representation — that's the standard part — plus the task's own private bias, plus the projected global coupling. The key thing is that the weight matrices stay independent. You're only coupling the intercepts. It's the minimal possible change that still lets tasks influence each other.

Why would you choose this over the others? Because it pools statistical strength across imbalanced tasks. Speaker ID has over a hundred thousand training samples. Emotion recognition has maybe five thousand. That emotion bias estimate has high variance simply because it doesn't see enough data. But the global coupling vector gets updated by every sample from both tasks. So the data-rich task indirectly stabilizes the data-poor one. And this is the lightest change of all three — same training loop, negligible extra parameters.

What do we expect? Better accuracy on those data-poor tasks. More stable convergence if you rerun with different seeds. And you can actually inspect the projection weights — their magnitude tells you how strongly each task leans on the shared coupling, which is itself an interpretable signal about cross-task dependence."

---

## Slides 5-6 — Task Structure Matrix (TSM)

**Paper:** Ciliberto et al. 2015 — SKMTL

"Now the Task Structure Matrix, from Ciliberto et al. 2015. This one is more ambitious.

The backbone and hidden layer stay the same, but we completely replace the independent task heads. Instead of three separate linear classifiers — one for keyword spotting, one for speaker ID, one for emotion — we introduce K shared latent classifiers plus a structure matrix we call A. The latent classifiers are full linear classifiers that each produce logits from the shared representation. The A matrix is T by K — tasks by latents — and each row is softmax-normalized so every task is a convex combination of the K latents.

The approach is elegant. Each latent classifier learns a distinct decision pattern. One might specialize in content-like distinctions, another in speaker-like patterns, another in affective features. The A matrix determines how much of each latent each task uses. And here's the crucial part: we put L1 regularization on A. That pushes small entries exactly to zero. So after training, the non-zero pattern of A literally is the task structure you've discovered. If keyword spotting uses latents 1 and 3, and emotion also uses 3, you know they share that underlying pattern. If speaker ID uses completely different latents, it's independent. No assumptions — the data determines the groupings.

Training uses alternating minimization. Most epochs are standard supervised training with A fixed through the softmax. Then every few epochs you freeze the model and optimize only A with the sparsity penalty. Ciliberto proved this converges to a global optimum, so it's not a heuristic — you have formal guarantees.

Why choose this? It discovers task groupings from data without you having to specify anything about which tasks should be related. The L1 sparsity actively blocks negative transfer — if a latent is harmful for a task, its entry goes to zero and the task simply stops using it. And the alternating optimization has those convergence proofs.

Expected outcome: you can read the rows of A and see exactly which latents each task uses. Unrelated tasks naturally get isolated to their own latent classifiers. And there's a nice side benefit — when K is less than T, the shared latent basis uses fewer parameters than T independent large heads."

---

## Slides 7-8 — Precision Matrix Regularization (PMR)

**Paper:** Gonçalves et al. 2016 — MSSL

"Finally, Precision Matrix Regularization, from Gonçalves et al. 2016. This takes a completely different philosophy from the first two.

The architecture and forward pass are untouched. 100% identical to the original wavCSE. PMR modifies only the training loss. You add a learnable precision matrix — that's Omega, the inverse covariance — and it regularizes the task parameters through three additional loss terms.

Here's the theoretical foundation. Gonçalves modeled the task parameters as draws from a multivariate Gaussian where the precision is Omega. In a Gaussian, the precision matrix has a powerful property: Omega-ij equals zero means tasks i and j are conditionally independent given all other tasks. That's not correlation — it's a much stronger, more rigorous statement. If Omega-12 is zero, knowing the parameters of task 1 tells you nothing about task 2 beyond what you could already infer from all other tasks. That's a proper probabilistic model, not a heuristic coupling.

Now the loss. We construct a parameter matrix W by flattening each task head's weights and biases into rows. Then we add three terms. First, minus log determinant of Omega — this is a barrier function, it blows up if Omega tries to collapse to zero. Second, the trace of W times Omega times W transpose — this is the coupling term, it pushes task parameters closer together or further apart depending on Omega's entries. Third, L1 penalty on the off-diagonal entries of Omega, which enforces sparsity so only the meaningful relationships survive. We parameterize Omega through its Cholesky factor — L times L transpose — which guarantees positive semi-definiteness by construction without needing the expensive ADMM projections Gonçalves originally used.

Crucially, we use a warmup. For the first few epochs, no PMR at all — just standard cross-entropy. Once the heads have converged to reasonable values, we enable the PMR terms. If you don't do this, Omega latches onto noise in the randomly initialized parameters.

Why choose PMR? It's the only one that gives you a formal probabilistic model — conditional independence, not heuristic coupling. Since the forward pass is unchanged, inference is byte-for-byte identical to the original. You can take an already-trained wavCSE model, fine-tune it with PMR, and deploy it without any pipeline changes.

Expected outcome: a visualizable dependency graph.Unrelated tasks get zero Omega entries, meaning zero coupling force — controlled negative transfer. And Omega itself is a research artifact. You can compare the learned structure graph across runs and across datasets."

---

**Total: ~6 minutes across three segments.**
