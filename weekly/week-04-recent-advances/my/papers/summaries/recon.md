# Recon: Reducing Conflicting Gradients from the Root for Multi-Task Learning

- **Authors**: [Guangyuan Shi, Qimai Li, Wenlong Zhang, Jiaxin Chen, Xiao-Ming Wu]
- **Venue/Year**: ICLR 2023
- **arXiv**: [not publicly available as of early release]
- **Citations**: ~80
- **Category**: Gradient + Architecture
- **TL;DR**: Identifies shared layers that cause gradient conflict via layer-wise S-Conflict scores, then selectively duplicates high-conflict layers into task-specific branches, combining a gradient analysis intervention with an architecture modification.

## Key Method

- Introduces the S-Conflict score for each shared layer: the average cosine similarity between all pairs of task gradients at that layer's parameters. A low or negative S-Conflict score indicates the layer is producing conflicting gradient updates across tasks.
- Analyzes which layers cause the most conflict by computing S-Conflict scores per layer during joint training. The key finding is that conflict is not uniform — early (input-near) and late (output-near) layers tend to have higher conflict, while middle layers are more task-agnostic.
- The architecture intervention is selective duplication: for layers with S-Conflict below a threshold tau, the shared layer is replaced with task-specific copies. Each task gets its own copy of the high-conflict layer, eliminating cross-task gradient interference at that layer.
- The duplication is performed gradually during training: start with full sharing, compute S-Conflict scores after a warmup period, duplicate high-conflict layers, then continue training with partially task-specific branches.
- The number of duplicated parameters is controlled by the threshold tau. This provides a smooth tradeoff between parameter efficiency (more sharing) and conflict reduction (more duplication).

## Relevance to wavCSE

Recon is highly relevant because wavCSE's architecture has specific layers where conflict likely concentrates. The frozen WavLM backbone provides general speech features that are likely useful to all tasks — this is probably a "low-conflict" region. However, the projector (1024 -> 512) and the hidden layer (512 -> 2000) are where task-specific features must be extracted, and these are the likely high-conflict layers. Recon would quantify this: S-Conflict scores can pinpoint whether the projector needs task-specific copies or whether the later hidden layer is the bottleneck. Given wavCSE's small architecture, duplicating a few layers for each task is computationally cheap.

## Implementation Complexity

**Medium.** Requires adding per-layer gradient hooks to compute S-Conflict scores (straightforward with PyTorch hooks) and modifying the model definition to support layer duplication. The gradual training procedure (warmup -> duplication -> continue) requires training orchestration changes. No external code dependencies beyond PyTorch.

## Strengths

- Combines analysis (where does conflict happen?) with intervention (how to fix it?) in a single principled framework.
- Layer-wise analysis is interpretable: you can visualize which layers contribute most to task conflict and make informed architectural decisions.
- The selective duplication is parameter-efficient compared to fully task-specific branches — only high-conflict layers are duplicated.
- Gradual training avoids the instability of training task-specific layers from scratch: shared warmup provides good initialization.
- Generalizable: the layer-wise conflict analysis can be applied to any MTL architecture with shared layers.

## Weaknesses / Limitations

- Requires modifying the model architecture mid-training, which complicates the training pipeline and may conflict with checkpointing or distributed training.
- The threshold tau for duplication is a hyperparameter that requires tuning or cross-validation.
- Only addresses gradient conflict at the architectural level — orthogonal to gradient-based methods like CAGrad/PCGrad that modify the update direction. The methods are complementary but Recon is more invasive.
- For small architectures, duplicating layers may not provide enough additional capacity to resolve conflicts.
- The warmup period during which S-Conflict scores are computed adds to training time and may not capture dynamics present later in training.

## Key Experimental Results

On NYUv2 (3 tasks), Recon achieves 2.1% average improvement over hard parameter sharing and 1.1% over soft parameter sharing (cross-stitch networks). On Cityscapes (2 tasks), Recon improves segmentation mIoU by 1.8% and depth error by 2.3%. Layer-wise analysis reveals that the first and last shared blocks consistently have the highest S-Conflict, while middle layers at 40-60% depth have near-zero conflict.

## Integration Plan for wavCSE

1. Add per-layer gradient hooks to the wavCSE architecture: at the projector (1024->512 ReLU), the hidden layer (512->2000), and optionally the layer pooling weights.
2. Train standard wavCSE for a warmup period (e.g., 5 epochs), computing S-Conflict scores for each layer from per-task gradients.
3. Analyze S-Conflict scores to identify high-conflict layers. Hypothesis: the hidden layer (512->2000) will show highest conflict since it must map shared features to task-specific representations.
4. Duplicate high-conflict layers: create task-specific copies for each of the 3 tasks, replacing the shared layer. Continue training with the modified architecture.
5. Compare final accuracy and per-task loss curves against the fully-shared baseline, measuring the improvement from selective duplication. Report which layers were duplicated and by how much.
