# PCGrad: Gradient Surgery for Multi-Task Learning

- **Authors**: Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn
- **Venue/Year**: NeurIPS 2020
- **arXiv**: 2001.06782
- **Citations**: ~1200
- **Category**: Gradient
- **TL;DR**: Projects conflicting gradients (those with negative cosine similarity) onto the orthogonal subspace of each other, removing the conflicting component while preserving the non-conflicting part, serving as the most widely adopted baseline for gradient-based MTL methods.

## Key Method

- At each training step, for every pair of tasks (i, j), compute the cosine similarity between their gradients g_i and g_j. If cos(g_i, g_j) < 0 (they conflict), project g_i onto the orthogonal complement of g_j: g_i' = g_i - (g_i^T g_j / ||g_j||^2) * g_j.
- Each gradient is "surgically" modified by removing the components that conflict with other tasks. The projection is applied sequentially: for each task i, its gradient is projected onto the orthogonal complement of the sum of all conflicting task gradients.
- The algorithm is simple to implement: (a) compute per-task gradients, (b) for each task i, iteratively subtract the projection of g_i onto each conflicting g_j, (c) average the modified gradients. This is O(k^2) in the number of tasks per iteration.
- The resulting update direction is guaranteed to have non-negative inner product with each task's gradient, meaning every step improves (or at least does not harm) each task in the direction of steepest descent.
- No hyperparameters to tune beyond standard learning rate scheduling. The method is a "drop-in" replacement for gradient averaging in any MTL training loop.

## Relevance to wavCSE

PCGrad is the most natural starting point for modifying wavCSE's training. wavCSE currently uses simple gradient averaging, which is known to fail when gradients conflict. PCGrad directly addresses this by removing conflicting components. For wavCSE's 3-task setup, the overhead of O(k^2) = 9 gradient projections per step is minimal. The key question is whether the speaker gradient (which likely dominates in magnitude) is conflicting with content and emotion — PCGrad would remove the conflicting components and preserve only the aligned signal. Since PCGrad requires no hyperparameter tuning and no architecture changes, it is the easiest gradient-based method to integrate into wavCSE.

## Implementation Complexity

**Very Low.** The simplest gradient-based MTL intervention. Requires collecting per-task gradients and applying the projection formula. Can be implemented in ~30 lines of PyTorch. The authors' code is widely available and integrated into LibMTL. No hyperparameters, no architecture changes, no loss modifications.

## Strengths

- Extremely simple: conceptually clear, easy to implement, and easy to debug.
- No hyperparameters (beyond existing training hyperparameters like learning rate). This eliminates a major source of complexity in MTL.
- Widely validated: the standard baseline for all subsequent gradient-based MTL methods (CAGrad, Nash-MTL, etc.), with hundreds of citations demonstrating broad applicability.
- The "zero-conflict" guarantee is intuitive and testable: after PCGrad, cos(g_i', g_j') >= 0 for all pairs.
- Works with any task count, any architecture, any loss function — truly a drop-in replacement for gradient averaging.
- Often sufficient: for many MTL problems, PCGrad's simple conflict removal provides most of the benefit, and more complex methods add marginal gains.

## Weaknesses / Limitations

- The sequential projection order matters and is not principled: different orderings can produce different final gradients. The paper uses random ordering, but this introduces variance.
- Removing conflicting components is aggressive: some conflict may be informative (e.g., indicating a genuine tradeoff that should be explored). PCGrad assumes all conflict is harmful.
- Does not handle gradient magnitude differences: if speaker's gradient is 10x larger than emotion's, even after removing conflicting components, the residual speaker gradient may still dominate.
- The method can produce very small updates when gradients are heavily conflicting, potentially slowing convergence in high-conflict regimes.
- More sophisticated methods (CAGrad, Nash-MTL) generally outperform PCGrad on challenging benchmarks, suggesting the simple projection is suboptimal.

## Key Experimental Results

On Multi-MNIST (2 tasks), PCGrad reduces digit classification error from 3.2% (uniform averaging) to 2.4% and color classification error from 7.1% to 5.2%. On NBA player trajectory prediction (2 tasks: offensive and defensive player prediction), PCGrad improves average displacement error by 5-8%. On a 7-task reinforcement learning setup, PCGrad improves average success rate by 10-15% over uniform averaging. The improvements are consistent across tasks but often smaller than more recent methods.

## Integration Plan for wavCSE

1. Modify the wavCSE training loop to collect per-task gradients: run backward for each task separately (or use `torch.autograd.grad` with `retain_graph=True`), gathering g_content, g_speaker, g_emotion from the shared parameters (projector and hidden layer).
2. Implement the PCGrad projection: for each of the 3 task gradients, iteratively project it onto the orthogonal complement of any task with negative cosine similarity.
3. Average the modified gradients and apply the update. This replaces the current single backward pass with the PCGrad procedure.
4. Train wavCSE with PCGrad using identical hyperparameters (learning rate, batch size, epochs) as the baseline for fair comparison.
5. Report per-task accuracy for content, speaker, and emotion. Measure the cosine similarity before and after PCGrad to verify conflict reduction. This serves as the primary gradient-based baseline against which CAGrad, Nash-MTL, and ExcessMTL can be compared.
