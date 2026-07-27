# AdaShare: Learning What to Share for Multi-Task Learning

- **Authors**: Ximeng Sun, Rameswar Panda, Rogerio Feris
- **Venue/Year**: CVPR 2020
- **arXiv**: 1911.12190
- **Citations**: ~300
- **Category**: Task Relation + Architecture
- **TL;DR**: Learns task-specific binary masks over shared network blocks via Gumbel-sigmoid relaxation, enabling each task to selectively use only the layers it needs from a shared backbone while skipping irrelevant computation.

## Key Method

- Divides the network into sequential blocks (e.g., ResNet stages, or transformer layers). Each task learns a binary mask per block indicating whether that task's input should be routed through the block.
- A task that skips a block uses the previous block's output directly (identity shortcut). This means tasks can effectively follow different computational paths through a shared network.
- The binary masks are learned via a Gumbel-sigmoid relaxation: during training, the hard 0/1 decisions are replaced with continuous sigmoid values with Gumbel noise, making the mask selection differentiable. During inference, masks are thresholded to binary.
- A regularizer controls the total computational cost: sum over tasks of (number of blocks used / total blocks). The lambda hyperparameter trades off task performance vs. computational efficiency.
- The masks are input-dependent: conditioned on the input features, not just the task identity. This allows adaptive computation where hard examples use more blocks.
- The resulting sharing pattern is highly structured: the first few layers are almost always shared (general feature extraction), while later layers diverge by task. The exact pattern emerges from optimization, not manual design.

## Relevance to wavCSE

AdaShare's approach is particularly relevant because wavCSE uses a frozen WavLM-24 backbone. The question is: which of the 24 transformer layers should the three tasks (content, speaker, emotion) share? Currently, wavCSE uses layer pooling (attention-weighted sum over the 24 layers) followed by a shared projector. AdaShare suggests that tasks might benefit from using different subsets of WavLM layers. For example, content task might prefer higher-level semantic layers (layers 20-24), speaker task might prefer middle layers with phonetic information (layers 10-16), and emotion might prefer early layers with prosodic features (layers 1-8). AdaShare's binary masks could be applied to WavLM's layer outputs before the layer pooling step.

## Implementation Complexity

**Medium-High.** Requires modifying the wavCSE architecture to insert Gumbel-sigmoid gates between WavLM layers and the layer pooling module. The Gumbel-sigmoid relaxation is well-studied but requires careful tuning of the temperature schedule. The computational cost regularizer adds an extra loss term. Training may be less stable due to the discrete nature of the masks.

## Strengths

- Learned sharing patterns are often more effective than manually designed architectures — the optimization discovers non-obvious sharing structures.
- The computational cost regularizer provides a Pareto frontier: users can trade accuracy for speed at inference time.
- Input-dependent masks enable adaptive computation, which is useful for variable-difficulty inputs (common in speech).
- The Gumbel-sigmoid relaxation provides a clean gradient estimator; the approach is architecture-agnostic and can be applied to any sequential backbone.
- Empirically, the learned masks show interpretable structure: earlier layers are shared more, later layers diverge — validating intuitions about feature hierarchy.

## Weaknesses / Limitations

- Training with Gumbel-sigmoid can be unstable: the temperature annealing schedule requires tuning, and the discrete nature causes gradient variance.
- The method introduces O(tasks * blocks) additional parameters for the gating functions, adding memory overhead.
- The computational cost regularizer introduces a hyperparameter (lambda) that must be tuned per use case.
- The masks are learned for each block independently but do not account for sequential dependencies between blocks.
- For wavCSE, the frozen WavLM backbone cannot be fine-tuned, so the masks only control which layers' outputs are used for the pooling — the backbone weights themselves remain fixed, limiting the benefit of selective routing.

## Key Experimental Results

On NYUv2 (3 tasks), AdaShare achieves 1.2% average improvement over hard parameter sharing with 20% less computation (as measured by FLOPs). On Taskonomy (26 tasks), AdaShare matches the performance of fully-shared baselines while using 40-60% fewer FLOPs, demonstrating significant computational savings without accuracy loss.

## Integration Plan for wavCSE

1. Extend wavCSE's layer pooling module: instead of attention-weighted pooling over all 24 WavLM layers, insert Gumbel-sigmoid gates that learn a task-specific binary mask per layer.
2. Each task gets its own set of 24 binary mask parameters. During training, the masks are relaxed with Gumbel noise; during inference, they are thresholded and fixed.
3. Add a computational cost regularizer lambda * sum(mask_ij) to the loss, where lambda is tuned on a held-out set.
4. Train the full model end-to-end (the WavLM backbone remains frozen; only the gating and projection parameters are learned).
5. Analyze the learned masks: do content, speaker, and emotion select different WavLM layers? Do the masks match linguistic intuitions (content -> late layers, emotion -> early layers)? Compare performance against the uniform layer pooling baseline.
