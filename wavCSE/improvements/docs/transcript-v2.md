# wavCSE Architectural Improvements — Transcript v2

**Covers slides 3-8 only (architecture pairs). Each pair spoken as one continuous segment.**

---

## Slides 3-4 — Global Bias Coupling (GBC) | ~2 min

**Paper:** Zhang et al. 2010 — BOMC

"Global Bias Coupling, from Zhang et al. 2010. This is the lightest architectural change of the three. The forward path is unchanged except for the bias term. A global coupling vector is added to each task's logits via per-task projection layers. That's the integration — drop it in without touching the backbone, the projector, the hidden layer, or the weight matrices.

The approach is a shared bias hub that receives gradients from every training example and therefore converges with lower variance than any per-task bias. The formula is straightforward — logit for task t equals the weight matrix times the hidden representation, plus the task's private bias, plus the projected global coupling. This is bias-only coupling. Weight matrices stay independent. Only the intercepts are linked.

Why use it? It pools statistical strength across imbalanced tasks. Speaker ID has over a hundred thousand samples; emotion recognition has around five thousand. The global bias is updated by every sample from both, so the emotion task's bias estimate benefits from speaker data. It's the lightest change — no alternating training, no extra loss terms, negligible parameter increase.

What we expect: better accuracy on data-poor tasks through reduced bias variance, more stable convergence across random seeds, and interpretable projection weights that show which tasks depend most on the shared coupling."

---

## Slides 5-6 — Task Structure Matrix (TSM) | ~2 min

**Paper:** Ciliberto et al. 2015 — SKMTL

"Task Structure Matrix, from Ciliberto et al. 2015. The backbone and hidden layer are unchanged, but the independent task heads are replaced entirely by K shared latent classifiers plus a structure matrix A.

The approach: tasks decompose as sparse convex combinations of shared latent classifiers. Each latent classifier learns a distinct decision pattern from the shared representation. The structure matrix A, with softmax-normalized rows, determines how each task mixes these latents. L1 regularization on A pushes spurious entries to zero. After training, the non-zero pattern of A is the discovered task structure — if keyword spotting and emotion both use latent 3, they share that underlying pattern. If speaker ID uses completely different latents, it's independent.

Training uses alternating minimization — supervised steps with A fixed, then unsupervised steps optimizing only A with the sparsity penalty. Ciliberto proved this converges to a global optimum. The key formula: each task's logit is the weighted sum over K latents, A-soft times the latent output, plus a task bias, with an L1 penalty on A.

Why use it? It discovers task groupings from data with no prior assumptions. L1 sparsity provides explicit negative transfer protection — harmful cross-task connections are zeroed out. The alternating optimization has formal convergence guarantees.

Expected outcome: interpretable task clusters visible in the rows of A. Unrelated tasks get isolated to their own latent classifiers. And when K is less than T, the shared basis is more parameter-efficient than T independent heads."

---

## Slides 7-8 — Precision Matrix Regularization (PMR) | ~2 min

**Paper:** Gonçalves et al. 2016 — MSSL

"Precision Matrix Regularization, from Gonçalves et al. 2016. This takes a fundamentally different approach — the architecture and forward pass are completely unchanged. A precision matrix Omega is added to the training loss only.

The approach models task parameters as draws from a multivariate Gaussian with sparse precision Omega. Omega-ij equals zero means tasks i and j are conditionally independent given all other tasks — a rigorous statistical interpretation, not just correlation. We construct a parameter matrix W by flattening each task head's weights and biases into rows. Three terms are added to the loss: a log-determinant barrier preventing Omega from collapsing, a trace term coupling task parameters through Omega, and L1 sparsity on off-diagonal entries. Omega is parameterized via Cholesky to guarantee positive semi-definiteness by construction. Training includes a warmup — standard training for the first few epochs, then the PMR terms are enabled — so heads converge before structure is imposed.

Why use it? It provides a formal probabilistic model of task relationships — conditional independence, not heuristics. Inference is byte-for-byte identical to the original model since PMR is purely a training regularizer. The warmup prevents fitting spurious structure to random initial parameters.

Expected outcome: a visualizable dependency graph by thresholding Omega — draw edges where entries are non-zero. Controlled negative transfer — unrelated tasks get zero Omega entries, meaning no coupling force acts on them. Omega is a reproducible artifact that can be compared across runs and datasets."
