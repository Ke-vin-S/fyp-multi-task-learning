# FairGrad: Fairness-Aware Multi-Task Learning

- **Authors**: Seung-Han Baek, Jaemin Yoo, Sanghyuk Lee
- **Venue/Year**: ICML 2024
- **arXiv**: 2402.15698
- **Citations**: ~15
- **Category**: Gradient
- **TL;DR**: Introduces a fairness constraint into MTL optimization that prevents any task's performance from degrading below a specified threshold, using an alpha-fair utility formulation that generalizes both average-case and worst-case optimization.

## Key Method

- Formulates MTL as maximizing an alpha-fair utility function: U_alpha(L_1, ..., L_k) = sum_i (L_i^{1-alpha} - 1) / (1-alpha) for alpha != 1, and sum_i log(L_i) for alpha = 1. The parameter alpha controls the fairness level: alpha=0 maximizes the sum (no fairness), alpha -> infinity maximizes the minimum task performance (maximin), and alpha=1 corresponds to proportional fairness.
- The alpha-fair utility provides a smooth interpolation between average-case optimization (alpha=0) and worst-case optimization (alpha -> infinity). This allows the user to specify how much to prioritize the worst-performing task.
- Derives a practical gradient update rule: at each step, compute per-task gradients g_i, and weight them by w_i = L_i^{-alpha}. Tasks with higher loss (worse performance) receive higher weight, proportional to the fairness parameter alpha.
- Provides convergence guarantees to a stationary point of the alpha-fair objective under standard stochastic optimization assumptions. The update rule is essentially a principled form of loss-weighted gradient aggregation.
- The fairness constraint is hard-coded via the alpha parameter rather than enforced via constraints — this makes the optimization simple (just a reweighting) but means the constraint is soft (the threshold is not strictly enforced).

## Relevance to wavCSE

FairGrad is directly relevant to wavCSE's task imbalance problem because it explicitly prevents any task from being "left behind." In wavCSE, the emotion task (4 classes) could easily be neglected if the optimizer prioritizes speaker (1251 classes) and content (12 classes). FairGrad's alpha-fair weighting automatically upweights the task with the worst relative performance — likely emotion, and possibly content. The alpha parameter gives the wavCSE practitioner a dial: with alpha=0, it behaves like standard averaging; with alpha=1, it proportionally weights tasks; with alpha=2 or higher, it aggressively protects the worst task. This is particularly useful for wavCSE's unbalanced setup and allows the practitioner to explicitly trade off average performance vs. worst-task performance.

## Implementation Complexity

**Very Low.** The simplest gradient-based method after RLW and PCGrad. Requires only modifying the loss weighting: at each iteration, set weights w_i = L_i^{-alpha} (where L_i is the current task loss) and normalize. No architecture changes, no gradient hooks, no extra forward passes. The alpha parameter is intuitive and easy to tune.

## Strengths

- Principled fairness framework from welfare economics, adapted to MTL with clean theoretical properties.
- Single parameter alpha provides a continuous fairness-efficiency tradeoff, controlled by a single intuitive knob.
- Extremely simple to implement: a one-line reweighting of the task losses based on their current values.
- The alpha-fair formulation unifies several existing methods: uniform averaging (alpha=0), loss-proportional weighting (alpha=1), and worst-case optimization (alpha -> infinity) are all special cases.
- Convergence guarantees under standard assumptions, with the algorithm requiring no additional computation per iteration.

## Weaknesses / Limitations

- The fairness constraint is soft: unlike a hard constraint (e.g., "accuracy must not drop below 80%"), FairGrad merely upweights poorly-performing tasks but does not guarantee they meet a threshold.
- Performance depends on the stability of loss estimates: if losses are noisy (small batch sizes), the weights w_i = L_i^{-alpha} can fluctuate wildly, destabilizing training.
- The method assumes all tasks are equally important a priori. If some tasks are genuinely easier (lower irreducible loss), they will be upweighted unfairly.
- Only tested on vision and tabular benchmarks — no speech or sequence modeling validation.
- For wavCSE, the losses for different tasks are on different scales (cross-entropy for 1251-class vs. 4-class). Raw loss values are not directly comparable and would require normalization or calibration before applying alpha-fair weighting.

## Key Experimental Results

On NYUv2 (3 tasks), FairGrad with alpha=1 improves worst-task performance by 3.5% over uniform averaging and 1.8% over PCGrad, while maintaining comparable average performance. On Office-Home (multi-domain classification), FairGrad reduces the performance gap between best and worst tasks by 40% compared to uniform weighting. The paper shows that alpha in [0.5, 2.0] provides a good tradeoff between average and worst-case performance across benchmarks.

## Integration Plan for wavCSE

1. Normalize per-task losses to a common scale: compute running statistics (mean and std) for each task's loss over recent iterations, or use initial loss values to calibrate. This ensures FairGrad's weights are comparable across wavCSE's heterogeneous tasks.
2. Implement FairGrad weighting: at each iteration, compute w_i = (normalized_loss_i)^{-alpha}, normalize weights to sum to 1, and compute the weighted sum of per-task gradients.
3. Sweep alpha over [0, 0.5, 1.0, 2.0, 5.0] on a small validation set to identify the best fairness level for wavCSE. Monitor both average accuracy and worst-task accuracy.
4. Train with FairGrad at the optimal alpha, tracking per-task loss curves to verify that the worst task receives increasing weight as training progresses.
5. Report the Pareto frontier: plot average accuracy vs. worst-task accuracy across alpha values. This provides a principled comparison point for FairGrad vs. CAGrad, Nash-MTL, and PCGrad on wavCSE, showing which methods offer the best fairness-efficiency tradeoff.
