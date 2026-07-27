# Grad-TAG: Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity

- **Authors**: Dongyue Li, Aneesh Sharma, Hongyang R. Zhang
- **Venue/Year**: KDD 2024
- **arXiv**: 2409.06091
- **Citations**: ~15 (2024)
- **Category**: Task Relation + Gradient
- **TL;DR**: Uses a gradient-based linearization of the training dynamics to estimate task affinities without full retraining, enabling 45x FLOPs reduction and SDP-based optimal task clustering.

## Key Method

- Linearizes the joint training dynamics using a first-order Taylor expansion around initialization, yielding a closed-form estimate of how each task's loss changes when trained jointly — no need to actually train all task combinations.
- From this linearization, constructs a task affinity matrix where each entry measures the pairwise effect of task A on task B's loss. The matrix is computed from per-task gradient samples at a single shared parameter point.
- Formulates task grouping as a semidefinite programming (SDP) problem: given an affinity matrix, find clusters that maximize intra-cluster affinity while respecting a cluster size constraint (e.g., GPU memory limits).
- The overall pipeline: (1) train a single shared model briefly to get gradient snapshots, (2) compute the linearized affinity matrix in one forward-backward pass per task, (3) solve the SDP to assign tasks to groups, (4) train each group with its own MTL model.
- Achieves 45x fewer FLOPs compared to brute-force pairwise training, with theoretical guarantees on the linearization error bound.

## Relevance to wavCSE

The frozen WavLM backbone + projector architecture in wavCSE means the trainable parameters are only in the projection and head layers (~0.5M parameters). Grad-TAG's gradient-based affinity estimation could be computed extremely cheaply on these layers, identifying whether content (12 classes), speaker (1251), and emotion (4) should be trained jointly or in sub-groups. Given wavCSE's severe task imbalance (speaker dominates by number of classes), Grad-TAG could reveal whether speaker should be isolated from content/emotion. The SDP clustering could also suggest adding a fourth supervision task (e.g., gender, duration) if affinity analysis shows it would not interfere.

## Implementation Complexity

**Low-Medium.** The gradient linearization requires collecting per-task gradients over a validation batch — straightforward in PyTorch with `torch.autograd.grad`. The SDP solver can use CVXPY with a standard SDP backend (SCS/MOSEK). No architectural changes needed, only a training-time analysis step.

## Strengths

- Dramatic reduction in computation: 45x fewer FLOPs than pairwise exhaustive search, making task grouping practical for large task sets.
- Theoretical grounding with error bounds on the linearized affinity estimates, connecting to the Neural Tangent Kernel literature.
- Flexible clustering framework: the SDP formulation handles arbitrary cluster size constraints (memory limits, latency budgets).
- Model-agnostic: works with any architecture that produces per-task gradients at shared parameters.
- Validation on both synthetic benchmarks and real-world recommendation datasets shows the grouped models outperform both single-task and all-tasks-joint baselines.

## Weaknesses / Limitations

- Linearization accuracy degrades for tasks whose training dynamics are highly nonlinear (e.g., tasks requiring deep feature reorganization). The NTK approximation assumes the model stays near initialization.
- Requires a meaningful shared parameter set from which to compute gradients — tasks with disjoint parameter sets need separate treatment.
- The SDP solver introduces overhead and may not scale to hundreds of tasks (though the paper handles dozens).
- No guarantee that the optimal grouping found at initialization remains optimal throughout training — task relationships can shift.
- The method computes pairwise affinities but does not model higher-order interactions (triplets of tasks may behave differently than pairwise sums).

## Key Experimental Results

On a 12-task setup from Taskonomy, Grad-TAG achieves the same performance as brute-force pairwise grouping while using 45x fewer FLOPs. The SDP-clustered groups outperform random grouping by 3-5% and full joint training by 2-3% on average, confirming that careful task grouping matters and that linearized affinities are sufficient for good groupings.

## Integration Plan for wavCSE

1. Add a `compute_linearized_affinity` function that, given the shared projector (512-d), samples a validation batch for each wavCSE task and computes per-task gradients with `torch.autograd.grad`.
2. Assemble the 3x3 affinity matrix (content, speaker, emotion) and assess whether speaker dominates the gradient geometry as hypothesized.
3. If the affinity matrix shows negative interference, implement the SDP clustering via CVXPY to determine optimal grouping (e.g., {content, emotion} vs {speaker}).
4. Train wavCSE under the recommended grouping(s) and compare task-wise accuracy against the current joint-3-task baseline.
5. Optionally extend to a 4- or 5-task setup (adding e.g., gender, speech rate) and use Grad-TAG to determine which combinations benefit from joint training.
