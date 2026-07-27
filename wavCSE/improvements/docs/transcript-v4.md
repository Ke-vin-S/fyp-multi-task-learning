# wavCSE Architectural Improvements — Transcript v4

**Shortened spoken version. Three segments, ~1 min each. ~3.5 min total.**

---

## Slides 3-4 — Global Bias Coupling (GBC)

**Paper:** Zhang et al. 2010 — BOMC

"Global Bias Coupling, from Zhang et al. 2010. The lightest change of the three.

Integration: forward path is untouched except the bias term. A global coupling vector is projected into each task's logits through small per-task layers. Backbone, hidden layer, weight matrices — all unchanged.

Approach: a shared bias hub that gets gradients from every training example, so it converges with lower variance than any per-task bias. Formula — logit is weight times hidden, plus task-private bias, plus projected global coupling. Weight matrices stay independent. Only intercepts are linked. That's the minimal change that still enables cross-task interaction.

Why: pools statistical strength across imbalanced tasks. Speaker ID has 150K samples, emotion has 5K. The global coupling is updated by both, so the data-poor emotion task's bias benefits from speaker data. Same training loop, negligible extra parameters.

Expected: better accuracy on data-poor tasks, more stable convergence across seeds, and projection weight magnitudes reveal which tasks depend most on shared coupling."

---

## Slides 5-6 — Task Structure Matrix (TSM)

**Paper:** Ciliberto et al. 2015 — SKMTL

"Task Structure Matrix, from Ciliberto et al. 2015.

Integration: backbone and hidden layer stay, but independent task heads are replaced by K shared latent classifiers plus a structure matrix A.

Approach: each latent classifier learns a distinct decision pattern. A softmax-normalized matrix A — T by K — determines how each task mixes these latents as a convex combination. L1 regularization on A zeroes out spurious entries. After training, the non-zero pattern of A is the discovered task structure. Training alternates: supervised steps with A fixed, then unsupervised steps optimizing only A with the sparsity penalty. Ciliberto proved this converges to a global optimum. Formula: each task's logit is the A-weighted sum over K latents, plus task bias, with an L1 penalty on A.

Why: discovers task groupings from data with no assumptions. L1 explicitly blocks negative transfer — harmful connections go to zero. Alternating optimization has convergence guarantees.

Expected: interpretable task clusters in rows of A. Unrelated tasks isolated to their own latents. Parameter-efficient when K is less than T."

---

## Slides 7-8 — Precision Matrix Regularization (PMR)

**Paper:** Gonçalves et al. 2016 — MSSL

"Precision Matrix Regularization, from Gonçalves et al. 2016. Different philosophy entirely.

Integration: forward pass completely unchanged. A precision matrix Omega is added to the training loss only. Architecture is byte-identical to original at inference.

Approach: task parameters are modeled as draws from a Gaussian with precision Omega. Omega-ij equals zero means tasks i and j are conditionally independent — rigorous, not heuristic. We flatten task head weights into a matrix W. Three terms are added to the loss: log-determinant barrier preventing Omega collapse, trace term coupling parameters through Omega, and L1 on off-diagonals for sparsity. Omega is parameterized via Cholesky to guarantee PSD. Warmup — first few epochs no PMR — so heads converge before structure is imposed.

Why: formal probabilistic model of task relationships. Inference identical to original. Warmup prevents fitting noise.

Expected: visualizable dependency graph by thresholding Omega. Controlled negative transfer — unrelated tasks get zero entries. Omega is reproducible across runs and datasets."
