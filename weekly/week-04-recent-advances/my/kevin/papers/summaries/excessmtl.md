# ExcessMTL: Excess Risk Optimization for Multi-Task Learning

- **Authors**: Feiyang Ye, Baijiong Lin, Yu Zhang
- **Venue/Year**: ICML 2024
- **arXiv**: 2405.12003
- **Citations**: ~20
- **Category**: Gradient + Theory
- **TL;DR**: Proposes optimizing excess risk (gap between current loss and the task's optimal loss) instead of raw loss, linking gradient conflicts to excess risk theoretically and deriving a practical weighted gradient aggregation method.

## Key Method

- Defines excess risk for task i as: e_i = L_i(theta) - L_i^*, where L_i^* is the optimal loss for task i (possibly estimated). The total objective is sum_i w_i * e_i, the weighted sum of excess risks rather than raw losses.
- The key theoretical contribution: gradient conflicts (negative cosine similarity between task gradients) directly increase excess risk. Specifically, the excess risk after a gradient step is bounded by a term involving pairwise gradient conflicts. This provides a formal link between gradient alignment and generalization.
- From this theory, derives a practical algorithm: compute per-task gradients, then find weights w_i that minimize an upper bound on the excess risk. The optimization is a quadratic program over the simplex with a closed-form solution when the bound is tight.
- The algorithm alternates between: (a) estimating L_i^* via a running minimum of recent task losses (exponential moving average), and (b) computing excess-optimal gradient weights via the derived formula.
- ExcessMTL generalizes several existing methods: when L_i^* = 0 for all tasks, it reduces to MGDA-style optimization. The excess risk framing provides a principled justification for dynamic weighting that responds to each task's proximity to optimality.

## Relevance to wavCSE

ExcessMTL's framing is particularly relevant to wavCSE's task imbalance problem. Content (12 classes), speaker (1251 classes), and emotion (4 classes) converge at different rates — speaker likely reaches a lower per-class loss due to more data per class, while emotion with only 4 classes may plateau earlier. ExcessMTL would dynamically weight tasks based on their distance from optimality, naturally de-emphasizing tasks that are "good enough" and focusing on those with more room for improvement. This adaptive behavior could prevent the speaker task from dominating training once it has converged, allowing content and emotion to continue improving. The theoretical link between gradient conflicts and excess risk also provides a principled framework for analyzing wavCSE's task interactions.

## Implementation Complexity

**Medium.** Requires tracking running estimates of optimal loss per task (an exponential moving average of the minimum loss seen so far) and implementing the quadratic programming weight computation. The QP is small (3 variables for wavCSE) and can be solved analytically. No architectural changes. The main complexity is in correctly estimating L_i^* — noisy estimates could destabilize training.

## Strengths

- Strong theoretical foundation: provides the first formal link between gradient conflict and excess risk, justifying gradient-based MTL methods from a generalization perspective.
- The excess risk framing is intuitive: tasks closer to optimality should receive less weight — this matches practical intuitions about curriculum learning and adaptive weighting.
- Generalizes existing methods (MGDA, PCGrad) under a unified theoretical framework, providing insight into why they work.
- The adaptive weighting naturally handles tasks that converge at different rates, which is a common problem in practice.
- The QP formulation is efficient (closed-form for small task counts) and adds negligible overhead per iteration.

## Weaknesses / Limitations

- Performance depends on accurate estimation of L_i^*. If the running minimum estimate is noisy (especially early in training), the excess risk values may be unreliable.
- The theory assumes convex losses or near-convex regions of the loss landscape — the guarantees weaken for highly non-convex settings like deep networks.
- The method adds another set of hyperparameters: the momentum rate for the running minimum estimator and potentially a clipping threshold for excess risk values.
- Only validated on computer vision benchmarks (Multi-MNIST, NYUv2, Office-31, Office-Home) — no speech or sequence modeling evaluation.
- For tasks with fundamentally different loss scales (cross-entropy for 1251-class classification vs. 4-class), the excess risk values differ by orders of magnitude, requiring additional normalization.

## Key Experimental Results

On Multi-MNIST (2 tasks), ExcessMTL reduces test error by 1.2% over uniform averaging and 0.5% over PCGrad. On NYUv2 (3 tasks), it achieves 1.8% average improvement over uniform averaging. The authors validate the theoretical claims by showing that excess risk optimization empirically reduces gradient conflict — the average cosine similarity between task gradients increases by 15-20% under ExcessMTL compared to uniform weighting.

## Integration Plan for wavCSE

1. Implement the excess risk estimator: maintain an exponential moving average of minimum loss per task, initialized after a warmup period (5 epochs of standard training to get reasonable loss values).
2. Implement the excess-risk weight computation: given per-task gradients and excess risk estimates, solve the 3-variable QP to get optimal weights.
3. Modify the wavCSE training loop to use excess-risk-weighted gradient aggregation instead of uniform averaging. Keep all other hyperparameters identical.
4. Train wavCSE with ExcessMTL and monitor per-task excess risk over time. Verify that tasks with lower excess risk (closer to optimality) receive lower weights, and vice versa.
5. Compare final per-task accuracy against uniform averaging, dynamically-weighted averaging (RLW), and CAGrad. Report both average performance and worst-task performance (since ExcessMTL should protect the worst task).
