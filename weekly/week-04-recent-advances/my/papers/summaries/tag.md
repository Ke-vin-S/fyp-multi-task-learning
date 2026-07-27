# TAG: Efficiently Identifying Task Groupings for Multi-Task Learning

- **Authors**: Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, Chelsea Finn
- **Venue/Year**: NeurIPS 2021
- **arXiv**: 2211.00288
- **Citations**: ~250
- **Category**: Task Relation
- **TL;DR**: Computers task affinity scores from a single jointly-trained model by measuring how each task's gradient changes the loss on other tasks, reducing the cost of task grouping from O(n^2) training runs to O(1) forward passes.

## Key Method

- Trains a single multi-task model to convergence, then at a frozen parameter checkpoint computes per-task gradient vectors. To measure affinity from task A to task B: subtract the gradient update direction for task A from the shared parameters, and measure the change in task B's loss — no retraining needed.
- The key insight is that the gradient from task A defines an update direction; applying that direction and checking the loss on task B reveals whether tasks are aligned (task B's loss decreases) or conflicting (task B's loss increases).
- Produces an asymmetric affinity matrix (affinity_{A->B} != affinity_{B->A}) from just the joint-trained model, plus O(k) gradient computations where k is the number of tasks.
- Uses spectral clustering on the symmetrized affinity matrix to produce task groupings. The clustering does not enforce equal-sized groups, allowing some groups to have many tasks while others have few.
- Validated on both synthetic label-shift tasks and large-scale production recommendation systems (YouTube, Google Play), showing that TAG-grouped models outperform both full joint training and random groupings.

## Relevance to wavCSE

TAG is directly applicable to wavCSE's 3-task setup. With the frozen WavLM backbone acting as a fixed feature extractor, the gradient dynamics on the projector and heads are relatively tractable. TAG can reveal the asymmetric relationships among content, speaker, and emotion: e.g., perhaps learning speaker identity helps emotion recognition (prosodic features overlap) but hurts content (speaker variation distracts from phoneme classification). The asymmetric matrix from TAG would quantify this directional interference. Since TAG requires only a trained joint model, wavCSE's current checkpoint already satisfies the input requirement — no additional training runs needed.

## Implementation Complexity

**Low.** Requires only forward passes through the trained model with gradient computation hooks. No architectural changes. The spectral clustering is a one-line `sklearn.cluster.SpectralClustering` call. TAG is one of the easiest task-relation methods to integrate into an existing pipeline.

## Strengths

- Extremely efficient: requires only a single already-trained joint model, not O(n^2) pairwise training runs. This is the key advantage over Standley et al. (ICML 2020).
- Asymmetric affinity signals directional relationships — e.g., "speaker helps emotion" but "emotion hurts speaker" — which is lost in symmetric methods.
- Empirically validated at production scale (YouTube, Google Play recommendation systems), demonstrating robustness.
- No special architecture or loss function modifications during training — purely an analysis tool applied post-hoc.
- The groupings found by TAG transfer: the clusters discovered on one joint model generalize to new initializations.

## Weaknesses / Limitations

- The affinity estimates are only valid at the converged solution from joint training. If tasks are severely conflicted, the joint model may settle in a poor region, making the affinities unrepresentative of well-separated training regimes.
- Assumes the model capacity is sufficient that gradient information at a single point captures task relationships. With highly limited shared parameters, the gradients may all be noisy.
- Spectral clustering requires choosing the number of groups a priori, which is often non-trivial.
- Since wavCSE has only 3 tasks, the "grouping" question is binary (split vs. join) — the full clustering machinery is overkill for this scale.

## Key Experimental Results

On the Taskonomy benchmark with 26 tasks, TAG groupings achieve competitive or better performance compared to brute-force pairwise evaluation, while requiring orders of magnitude less computation. On a YouTube production recommendation system, TAG-grouped models improve offline metrics by 0.3-0.8% over joint training across multiple task types (engagement, satisfaction, etc.), consistent gains at scale.

## Integration Plan for wavCSE

1. Take the existing converged wavCSE checkpoint (frozen WavLM + trained projector + heads), compute per-task gradients on a held-out validation batch for content, speaker, and emotion.
2. Implement the TAG affinity formula: for each ordered pair (A, B), compute the change in task B's loss after a small gradient step from task A — measure `L_B(theta - alpha * grad_A) - L_B(theta)`.
3. Construct the 3x3 asymmetric affinity matrix and analyze whether the relationships are largely positive (collaborative) or negative (conflicting).
4. Based on the affinities, train alternative configurations: e.g., if speaker negatively impacts content, isolate speaker with a separate projector or head branch.
5. Compare the TAG-predicted grouping against the actual performance gap to validate whether TAG's recommendations generalize to wavCSE's speech setting.
