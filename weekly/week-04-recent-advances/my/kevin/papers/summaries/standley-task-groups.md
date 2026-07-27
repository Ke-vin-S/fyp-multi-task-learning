# Which Tasks Should Be Learned Together in Multi-Task Learning?

- **Authors**: Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, Silvio Savarese
- **Venue/Year**: ICML 2020
- **arXiv**: 1905.07575
- **Citations**: ~550
- **Category**: Task Relation
- **TL;DR**: Systematic framework that trains all task pairs to build a task affinity matrix, then uses hierarchical clustering to determine optimal multi-task groupings, validated across dozens of visual tasks.

## Key Method

- Exhaustive pairwise evaluation: trains every pair of tasks (O(n^2) total runs) alongside single-task baselines. The performance of each pair relative to single-task performance defines the pairwise affinity score.
- Constructs an n x n symmetric affinity matrix where entry (i, j) captures the benefit (or cost) of training task i and task j together vs. separately. Affinity is measured as the average performance improvement across both tasks.
- Applies hierarchical agglomerative clustering on this affinity matrix to build a dendrogram of task groupings. The optimal cut point is determined by a held-out validation set — groups with high intra-cluster affinity and low inter-cluster affinity are preferred.
- Demonstrates the method on the Taskonomy dataset (26 visual tasks spanning geometry, semantics, and scene understanding), producing interpretable clusters: e.g., depth/normal/reshading cluster tightly, while object classification is more isolated.
- The paper also analyzes the effect of model capacity per task: as capacity decreases, the benefit of grouping increases (tasks are forced to share), but at high capacity, grouping can hurt.

## Relevance to wavCSE

Standley et al.'s framework provides the intellectual foundation for task grouping. For wavCSE, the central question is whether content (12 classes), speaker (1251 classes), and emotion (4 classes) should be trained jointly or in any 2-task combination. The exhaustive approach would require training all 3 choose 2 = 3 pairwise models plus 3 single-task models. Given wavCSE's small architecture (~0.5M trainable params), this is computationally cheap. The key insight from Standley is that capacity matters: with wavCSE's small projector, forcing the three tasks to share may hurt, and the pairwise affinities would reveal which pair(s) benefit from sharing.

## Implementation Complexity

**Low-Medium.** Straightforward to implement but requires running multiple training jobs (single-task and pair-task). Each run is cheap for wavCSE (small model, frozen backbone). The hierarchical clustering is trivial with scipy. The main labor is setting up the training sweeps and analyzing results.

## Strengths

- The most comprehensive empirical validation of task grouping principles, spanning 26 diverse tasks — provides strong evidence that careful grouping matters.
- Introduces the concept of "taskonomy" — building a taxonomy of task relationships — that has influenced nearly all subsequent task-grouping work.
- Analyzes the capacity-grouping tradeoff: smaller models benefit more from grouping, larger models can afford separation. This is directly relevant to practitioners choosing model scale.
- Hierarchical clustering produces a human-interpretable dendrogram that reveals task structure and can guide future task selection.
- The framework is agnostic to architecture, loss function, and training setup — purely an experimental design methodology.

## Weaknesses / Limitations

- O(n^2) training runs is prohibitively expensive for large task sets (e.g., 100+ tasks). This is the limitation that TAG (Fifty et al.) was designed to address.
- Pairwise affinities may not capture higher-order task interactions — three tasks together may behave differently than the sum of their pairwise relationships.
- The method requires defining a metric for "affinity" that aggregates performance across tasks, which may obscure task-specific tradeoffs.
- The clustering is static (performed once before training) and does not adapt to changing relationships during learning.
- For wavCSE's 3-task setup, the O(n^2) cost is trivial, but the binary nature of the experiment (split or join) may be better served by simpler methods.

## Key Experimental Results

On Taskonomy, grouping tasks by the discovered hierarchy yields a 2-3% average improvement over training all 26 tasks jointly, and a 7-10% improvement over single-task baselines. The optimal grouping is not "all together" or "all separate" — it lies in intermediate clusters (e.g., grouping depth, normal, and reshading together while keeping semantic segmentation separate), confirming that task grouping is a nuanced problem.

## Integration Plan for wavCSE

1. Establish single-task baselines: train separate wavCSE models for content-only, speaker-only, and emotion-only (each with the same frozen WavLM backbone and projector, but task-specific head).
2. Train all 3 pairwise models: content+speaker, content+emotion, speaker+emotion, using the standard wavCSE joint training loss (simple average).
3. Compute pairwise affinities: for each pair (i,j), measure the average relative improvement over single-task across both tasks.
4. Build the 3x3 affinity matrix and apply hierarchical clustering to determine whether 2+1 grouping (one pair together, one separate) outperforms all-3-joint.
5. Train the recommended configuration and measure per-task accuracy against the current all-3-joint baseline, quantifying the benefit (or cost) of selective sharing.
