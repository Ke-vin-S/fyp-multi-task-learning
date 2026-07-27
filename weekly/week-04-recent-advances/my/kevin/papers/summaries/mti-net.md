# MTI-Net: Multi-Scale Multi-Task Interaction Network

- **Authors**: Simon Vandenhende, Stamatios Georgoulis, Luc Van Gool
- **Venue/Year**: ECCV 2020
- **arXiv**: 2003.12908
- **Citations**: ~250
- **Category**: Task Relation + Architecture
- **TL;DR**: Introduces cross-task attention at multiple resolution scales where tasks modulate each other's feature maps through learnable attention, enabling dense multi-scale task interactions rather than sharing a single feature representation.

## Key Method

- Constructs a shared encoder backbone (e.g., ResNet/HRNet) that produces feature maps at multiple resolutions. Each task has its own decoder that processes these multi-scale features. The key innovation is the cross-task attention module inserted between encoder and decoder.
- The cross-task attention module works as follows: for each pair of tasks (A, B), compute an attention map from task A's features to modulate task B's features. This is done at each resolution scale independently. The attention is learned, not hand-designed, and can attend to spatial regions relevant to the source task.
- The interaction is symmetric but directional: task A's features are used to compute attention weights applied to task B's features, and vice versa. This allows each task to "look at" regions that are important for the other task.
- Multi-scale aggregation: interactions are computed at 4 resolution scales (1/4, 1/8, 1/16, 1/32 of input), and the resulting task features are fused across scales. This ensures both fine-grained and global interactions.
- The architecture is trained end-to-end with task-specific loss functions. The cross-task attention modules add approximately 15-20% more parameters compared to a hard-parameter-sharing baseline with task-specific decoders.
- Ablation studies show that: (a) cross-task attention outperforms simple feature concatenation or summation, (b) multi-scale interaction is better than single-scale, and (c) bidirectional interaction outperforms unidirectional.

## Relevance to wavCSE

MTI-Net's cross-task attention is architecturally complementary to all the gradient-based methods discussed elsewhere. While wavCSE currently uses a single shared projector followed by task-specific heads, MTI-Net suggests an alternative: task-specific feature extractors that cross-attend to each other. Applied to wavCSE, this could mean: instead of a single projector (1024->512), each task (content, speaker, emotion) has its own projector, and cross-attention modules allow the emotional content branch to attend to the speaker branch's features (which may carry prosodic information useful for emotion), while the content branch attends to different acoustic properties. The multi-scale aspect maps naturally to WavLM's hierarchical representations (24 layers producing features at different temporal/linguistic granularities).

## Implementation Complexity

**High.** MTI-Net's architecture is substantially more complex than wavCSE's current design. Requires implementing multi-scale task-specific branches, cross-attention modules for each task pair at each scale, and multi-scale feature fusion. The parameter count increase is significant. Training stability may require careful tuning. No off-the-shelf implementation for speech/sequence models exists — would need to adapt from the vision codebase.

## Strengths

- The most architecturally ambitious of the papers surveyed: enables rich, bidirectional, multi-scale task interactions rather than simple feature sharing.
- Cross-task attention is interpretable: attention maps reveal which regions/features one task finds useful in another task's representation.
- Comprehensive ablation studies provide clear design guidelines (multi-scale > single-scale, bidirectional > unidirectional).
- Strong empirical performance on semantic segmentation, depth estimation, and surface normal prediction — the NYUv2 multi-task benchmark.
- The architecture naturally handles tasks with different spatial/temporal resolutions, which is relevant for speech tasks with different granularities.

## Weaknesses / Limitations

- High architectural complexity: the cross-task attention modules add significant parameters and FLOPs, reducing the efficiency advantage of MTL.
- Designed for dense prediction tasks (pixel-level outputs) in computer vision. Adapting to speech classification tasks (utterance-level or pooled representations) requires significant rethinking.
- The cross-attention assumes spatial structure (2D feature maps); for speech, the structure is 1D temporal, which simplifies some aspects but changes the attention mechanism.
- Training stability can be challenging: joint training of multiple task-specific branches with cross-attention is more prone to optimization difficulties.
- The method does not address task imbalance or gradient conflict directly — it assumes the architecture naturally resolves these issues, which may not hold for wavCSE's severely imbalanced tasks.

## Key Experimental Results

On NYUv2 (13-class segmentation, depth estimation, surface normal prediction), MTI-Net achieves 41.27% mIoU (vs. 38.31% for hard sharing), 0.5257 depth error (vs. 0.5627), and 23.18° normal error (vs. 25.01°). On Cityscapes, MTI-Net improves segmentation mIoU by 2.6% and depth error by 1.8% over the cross-stitch baseline. Cross-task attention improves all tasks simultaneously — no task is harmed.

## Integration Plan for wavCSE

1. Redesign wavCSE's architecture: instead of a single shared projector, create three task-specific projectors (1024->512) for content, speaker, and emotion, each processing the pooled WavLM output independently.
2. Implement a 1D cross-attention module: for each pair of task projector outputs, compute attention weights from task A's features applied to task B's features (and vice versa). Since wavCSE uses utterance-level features (pooled over time), the attention operates on the 512-d feature vectors.
3. Design a multi-scale variant: use WavLM's layer outputs at different granularities (e.g., layers 8, 16, 24 as three "scales") and compute cross-task attention at each scale, then fuse via weighted sum.
4. Add task-specific hidden layers (512->2000) and classification heads (12, 1251, 4 classes) on top of the attended features.
5. Train end-to-end with the standard weighted loss, comparing per-task accuracy against the shared-projector baseline. Analyze the cross-attention weights to determine which task pairs benefit most from mutual feature modulation.
