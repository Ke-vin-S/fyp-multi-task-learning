# wavCSE Architectural Improvements — Presentation Transcript

**Duration:** ~8 minutes | **9 slides** | **Audience:** Technical (ML/speech research)

---

## Slide 1 — Title (30 seconds)

"Good morning. I'm presenting three architectural improvements to wavCSE, a multi-task speech representation learning framework from IJCNLP 2025. Each improvement draws from an established MTL paper — Zhang et al. 2010 on Bayesian online multi-label classification, Ciliberto et al. 2015 on sparse kernel MTL, and Gonçalves et al. 2016 on multi-task sparse structure learning. All three are implemented as drop-in variants, preserving wavCSE's core contributions — recursive layer selection and pooling strategies."

---

## Slide 2 — Problem Statement (40 seconds)

"The original wavCSE has independent task classification heads — keyword spotting, speaker ID, emotion recognition — each learning in isolation. Three issues: equal loss weighting regardless of task difficulty or data size, no detection or mitigation of negative transfer, and no interpretable task structure. The heads are black-box silos. Our question: can we add principled task-coupling mechanisms without touching wavCSE's novel contributions?"

---

## Slide 3 — GBC: Integration & Approach (50 seconds)

"Global Bias Coupling, from Zhang et al. 2010. The forward path is unchanged except for the bias term. A global coupling vector is added to each task's logits via per-task projection layers.

The approach: a shared bias hub that receives gradients from every training example and therefore converges with lower variance than any per-task bias. The formula is straightforward — each task's logit is the weight matrix times the hidden representation, plus the task's private bias, plus the projected global coupling.

This is bias-only coupling. Weight matrices stay independent. Only the intercepts are linked. It's the lightest possible architectural change that still enables cross-task interaction. No alternating training, no extra loss terms."

---

## Slide 4 — GBC: Why & Expected Outcome (40 seconds)

"Why GBC? It pools statistical strength across imbalanced tasks. Speaker ID has over a hundred thousand samples; emotion recognition has around five thousand. The global bias is updated by every sample from both, so the emotion task's bias estimate benefits from speaker data.

It's the lightest change of all three architectures — negligible parameter increase, same training loop.

What we expect: better accuracy on data-poor tasks through reduced bias variance, more stable convergence across random seeds, and interpretable projection weights that show which tasks depend most on the shared coupling."

---

## Slide 5 — TSM: Integration & Approach (50 seconds)

"Task Structure Matrix, from Ciliberto et al. 2015. The backbone and hidden layer are unchanged, but the independent task heads are replaced entirely by K shared latent classifiers plus a structure matrix A.

The approach: tasks decompose as sparse convex combinations of shared latent classifiers. Each latent classifier learns a distinct decision pattern from the shared representation. The structure matrix A, with softmax-normalized rows, determines how each task mixes these latents. L1 regularization on A pushes spurious entries to zero.

After training, the non-zero pattern of A is the discovered task structure. If keyword spotting and emotion both use latent 3, they share that underlying pattern. If speaker ID uses completely different latents, it's independent.

Training uses alternating minimization — supervised steps with A fixed, then unsupervised steps optimizing only A with the sparsity penalty. Ciliberto proved this converges to a global optimum."

---

## Slide 6 — TSM: Why & Expected Outcome (40 seconds)

"Why TSM? It discovers task groupings from data with no prior assumptions about which tasks are related. L1 sparsity provides explicit negative transfer protection — harmful cross-task connections are zeroed out. And the alternating optimization has formal convergence guarantees.

Expected outcome: interpretable task clusters visible in the rows of A. Unrelated tasks get isolated to their own latent classifiers rather than being forced to share. And when K is less than T, the shared basis is more parameter-efficient than T independent heads.

The tradeoff: replaces the original architecture, and A is asymmetric so there's no graphical model interpretation."

---

## Slide 7 — PMR: Integration & Approach (50 seconds)

"Precision Matrix Regularization, from Gonçalves et al. 2016. This takes a fundamentally different approach — the architecture and forward pass are completely unchanged. A precision matrix Omega is added to the training loss only.

The approach models task parameters as draws from a multivariate Gaussian with sparse precision Omega. Omega-ij equals zero means tasks i and j are conditionally independent given all other tasks — that's a rigorous statistical interpretation, not just correlation.

We construct a parameter matrix W by flattening each task head's weights and biases into rows. Three terms are added to the loss: a log-determinant barrier preventing Omega from collapsing, a trace term coupling task parameters through Omega, and L1 sparsity on off-diagonal entries. Omega is parameterized via Cholesky — L times L transpose — to guarantee positive semi-definiteness by construction.

Training includes a warmup — standard training for the first few epochs, then the PMR terms are enabled — so heads converge before structure is imposed."

---

## Slide 8 — PMR: Why & Expected Outcome (40 seconds)

"Why PMR? It provides a formal probabilistic model of task relationships — conditional independence, not heuristics. Inference is byte-for-byte identical to the original model since PMR is purely a training regularizer. And the warmup prevents fitting spurious structure to random initial parameters.

Expected outcome: a visualizable dependency graph by thresholding Omega — draw edges where entries are non-zero. Controlled negative transfer — unrelated tasks get zero Omega entries, meaning no coupling force acts on them. And Omega is a reproducible artifact that can be compared across runs and datasets.

The tradeoff: Omega parameters scale quadratically, though negligible for three tasks. And it only regularizes parameters — unlike TSM, it doesn't change what the model can represent."

---

## Slide 9 — Comparison (60 seconds)

"To summarize across three dimensions.

Integration: GBC touches only the bias term. TSM replaces the task heads. PMR modifies only the loss function.

Coupling mechanism: GBC uses a shared bias hub. TSM uses a convex combination of latent classifiers. PMR uses a precision matrix with conditional independence semantics.

Training: GBC is identical to the original. TSM adds alternating steps. PMR adds warmup plus an extra loss term per batch.

The discovered structure is implicit in GBC, explicit in the A matrix for TSM, and a formal dependency graph for PMR.

GBC is best when you want minimal change. TSM when you want interpretable task groupings. PMR when you need a formal statistical model.

Next steps: run all three on the combined keyword spotting, speaker ID, and emotion recognition task set with consistent seeds. Compare per-task accuracy, training dynamics, and the interpretability of each method's structure. And ablate the hidden layer expansion to see whether it's necessary for these coupling approaches.

Thank you. Questions?"
