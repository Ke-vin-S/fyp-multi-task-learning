# Nash-MTL: Multi-Task Learning as a Bargaining Game

- **Authors**: Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, Ethan Fetaya
- **Venue/Year**: ICML 2022
- **arXiv**: 2202.08617
- **Citations**: ~350
- **Category**: Gradient
- **TL;DR**: Reframes MTL optimization as a Nash bargaining game where tasks negotiate a joint gradient direction that is Pareto-optimal, scale-invariant, and guarantees that no task is made worse off relative to a baseline.

## Key Method

- Formulates MTL as a cooperative bargaining game: each task is a "player" that wants to minimize its loss, and the joint update direction is the negotiated "agreement." The Nash bargaining solution (NBS) is the unique direction satisfying four axioms: Pareto optimality, symmetry, independence of irrelevant alternatives, and scale invariance.
- The NBS in gradient space is found by solving: max_d sum_i log(g_i^T d) subject to d being in the convex cone of task gradients. This maximizes the product of improvements (the Nash product) rather than the sum, which is scale-invariant.
- A practical algorithm alternates between: (a) finding the NBS direction via a convex optimization over the probability simplex, and (b) taking a gradient step in that direction. The simplex optimization has a closed-form iterative solution via Frank-Wolfe or can be solved with a simple proportional allocation.
- Scale invariance is the critical property: the NBS direction does not change if a task's loss is multiplied by a constant. This eliminates the need for loss weighting — a task with 1251 classes naturally produces larger gradients than a task with 4 classes, but NBS treats them equally in terms of relative improvement.
- Provides convergence guarantees to a Pareto-stationary point under standard smoothness assumptions, with the NBS computable in O(k log k) time per iteration where k is the number of tasks.

## Relevance to wavCSE

Nash-MTL is directly relevant to wavCSE's task imbalance problem. Speaker (1251 classes) produces much larger gradients than emotion (4 classes), which standard averaging would amplify. Nash-MTL's scale invariance means each task's update weight depends on its relative improvement potential, not its raw gradient magnitude. For wavCSE, this could prevent speaker dominance without manual loss weighting. The method integrates at the optimizer level (like CAGrad and PCGrad) and requires no architectural changes. Since wavCSE has only 3 tasks, the O(k log k) overhead is trivial.

## Implementation Complexity

**Low-Medium.** Requires a gradient aggregation step at each iteration: collect per-task gradients, compute the NBS direction via Frank-Wolfe or a simple exponentiated gradient algorithm over the simplex. The scale invariance eliminates loss tuning. Code is available in the official repository and in LibMTL. No architecture changes.

## Strengths

- Scale invariance is a major practical advantage: eliminates the need to manually tune loss weights, which is a significant pain point in MTL.
- Strong axiomatic foundation from cooperative game theory provides principled justification beyond heuristic intuitions.
- Empirically robust across diverse benchmarks (NYUv2, Cityscapes, Multi-MNIST) with consistent improvements over both averaging and PCGrad.
- The Nash product formulation naturally prevents any task from being severely degraded, addressing the "neglected task" problem in MTL.
- Computationally efficient: Frank-Wolfe converges in a small number of iterations, and the overall overhead per batch is negligible.

## Weaknesses / Limitations

- The NBS assumes all tasks are equally important a priori. If the user wants to prioritize a specific task (e.g., content accuracy over emotion), this requires modifying the formulation.
- The Frank-Wolfe solver adds a tuning parameter (number of inner iterations) that affects solution quality vs. speed tradeoff.
- The scale invariance property requires that losses are comparable in a relative sense — tasks with very flat loss landscapes may still be neglected.
- For tasks that are already well-optimized (near zero loss), the log-barrier term can cause numerical issues or overly small updates.
- The method does not address task grouping or architectural sharing — purely an optimization-level intervention.

## Key Experimental Results

On NYUv2 (3 tasks: segmentation, depth, surface normal), Nash-MTL improves average performance by 2-3% over gradient averaging and 1-2% over PCGrad. On Cityscapes (2 tasks: segmentation, depth), it achieves the best reported mIoU among gradient-based MTL methods. The authors also show that Nash-MTL's scale invariance makes it robust to 10x differences in loss magnitudes, where weighted averaging degrades significantly.

## Integration Plan for wavCSE

1. Implement the Nash-MTL gradient aggregation: modify the training loop to collect per-task gradients, then solve the NBS via Frank-Wolfe over the 3-task simplex to get task weights w_1, w_2, w_3.
2. Verify scale invariance: multiply emotion loss by factors [0.1, 1, 10] and confirm that Nash-MTL produces the same update direction, while standard averaging changes dramatically.
3. Train wavCSE with Nash-MTL for the full training schedule, comparing per-task accuracy against standard averaging and against manually-tuned loss weights.
4. Analyze the Nash weights over the course of training: do they stabilize at some distribution (e.g., content: 0.4, speaker: 0.3, emotion: 0.3) or oscillate?
5. Report final task accuracies and the average Nash product (sum_of_log_improvements) to quantify the bargaining outcome. Compare with CAGrad and PCGrad baselines.
