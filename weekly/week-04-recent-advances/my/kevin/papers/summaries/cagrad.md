# CAGrad: Conflict-Averse Gradient Descent for Multi-Task Learning

- **Authors**: Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu
- **Venue/Year**: NeurIPS 2021
- **arXiv**: 2110.14048
- **Citations**: ~500
- **Category**: Gradient
- **TL;DR**: Formulates multi-task optimization as a min-max problem that finds a gradient update direction minimizing the average task loss while remaining within a ball around the average gradient, provably converging to a Pareto-optimal solution.

## Key Method

- At each training step, computes per-task gradients g_1, ..., g_k. The naive approach averages them: g_avg = (1/k) * sum(g_i). CAGrad instead solves: max_{d in R^m} min_i (g_i^T d) subject to ||d - g_avg|| <= c * ||g_avg||, where c is a hyperparameter controlling the "radius" of allowed deviation.
- Intuitively, CAGrad finds a direction d within a ball of radius c around the average gradient that maximizes the worst-case improvement across all tasks. When c=0, it reduces to simple averaging. When c is large, it can deviate arbitrarily to find conflict-free directions.
- The optimization has a closed-form solution: d* = g_avg + c * ||g_avg|| * v*, where v* is the solution to a simpler convex optimization over a sphere. This avoids expensive inner-loop iterations.
- Proof that CAGrad converges to a Pareto-optimal or Pareto-stationary point under standard assumptions, with a convergence rate of O(1/sqrt(T)) for non-convex objectives — matching standard SGD.
- The hyperparameter c controls conservatism: small c (0.1-0.3) gives conservative updates close to averaging; larger c (0.5-0.9) allows aggressive conflict resolution but may destabilize training.

## Relevance to wavCSE

wavCSE's three tasks (content, speaker, emotion) likely have conflicting gradients: speaker gradient directions (which must separate 1251 classes) likely dominate the average, potentially overwriting content and emotion signals. CAGrad can directly address this by finding a gradient direction that accounts for all three tasks' gradients rather than averaging them. For wavCSE specifically, CAGrad's learned direction could prevent the speaker task from dominating the update, since the min-max objective explicitly protects weak tasks. The projector (512-d) and head layers would see the most benefit since the WavLM backbone is frozen.

## Implementation Complexity

**Low.** A self-contained gradient manipulation method with a closed-form solution — no architectural changes, no hyperparameter tuning beyond c. Can be implemented in ~50 lines of PyTorch by modifying the optimizer step. Code available in the authors' repository and integrated into LibMTL.

## Strengths

- Strong theoretical foundation: provable Pareto optimality convergence, matching SGD rates.
- Closed-form solution makes it nearly as fast as simple gradient averaging — negligible overhead per iteration.
- Single hyperparameter (c) with intuitive meaning: degree of conservatism vs. conflict resolution.
- Consistent improvements over PCGrad and MGDA across a wide range of benchmarks (classification, regression, reinforcement learning).
- Works with any architecture and any task count — no per-task engineering.

## Weaknesses / Limitations

- The ball constraint around the average gradient may be too restrictive: if all tasks conflict strongly, no direction within the ball is good, and CAGrad defaults to near-averaging behavior.
- Performance is sensitive to c — requires tuning. For wavCSE's imbalanced tasks, the optimal c may differ from the paper's typical range (0.2-0.5).
- Assumes all tasks contribute equally to the min-max objective. For wavCSE, the speaker task's gradient is naturally larger (1251 classes), which may still dominate the min-max solution.
- Does not address task-specific learning rate or gradient scaling — only direction. Magnitude differences between tasks (e.g., speaker vs. emotion) may persist.
- The min-max formulation can be overly conservative, potentially slowing convergence for well-aligned tasks.

## Key Experimental Results

On the Multi-MNIST and NYUv2 benchmarks, CAGrad improves average task performance by 1-3% over gradient averaging and 0.5-2% over PCGrad. On a 7-task robot manipulation RL benchmark, CAGrad achieves 15-25% higher success rates than naive averaging. The method consistently finds Pareto-dominating solutions — improving one task without degrading others.

## Integration Plan for wavCSE

1. Implement the CAGrad optimizer wrapper that captures per-task gradients during backward, computes the min-max direction, and applies the update. This replaces the current simple averaging in the wavCSE training loop.
2. Sweep over c values [0.1, 0.3, 0.5, 0.7, 0.9] on a small validation set (10% of training) to identify the best conservatism level for the 3-task wavCSE setup.
3. Train the full wavCSE model with CAGrad at the optimal c, tracking per-task losses to verify that content and emotion losses decrease more than under standard averaging.
4. Compare final task accuracies across all three tasks, paying particular attention to content and emotion (which are likely being suppressed by speaker).
5. Analyze the learned gradient directions: collect CAGrad vs. averaging gradient angles to quantify how much CAGrad actually deviates from averaging in the wavCSE setting.
