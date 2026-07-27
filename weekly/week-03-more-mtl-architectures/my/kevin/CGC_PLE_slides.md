# Customized Gate Control (CGC)

*Reference: Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations" — ACM RecSys 2020 (Best Paper)*

---

## Architecture & Mechanism

- Extends MMoE by adding **task-specific expert networks** alongside the shared expert pool.
- Shared experts learn features common to all tasks; task-specific experts capture **task-private patterns**.
- Each task has its own **linear-softmax gate** that selects over both shared and its own specific experts.
- Expert outputs are combined via weighted sum (`einsum`) and passed to task-specific decoders.
- Single-level gating: one gate per task applied after the full encoder forward pass.

## Benefit & Outcome

- Explicit separation of shared vs. task-private feature extraction capacity.
- Each task can draw from shared knowledge while maintaining dedicated expert pathways.
- Reduces negative transfer by giving tasks independent experts that are isolated from other tasks.
- Simple extension of MMoE — inherits multi-input support and all MMoE gating utilities.

---

## LibMTL Implementation

- Creates multiple shared experts and task-specific experts using independent encoder instances.
- A separate linear-softmax gate learns expert weights for each task over the combined shared + task-specific pool.
- Weighted expert representations are combined via einsum and passed to task-specific decoders.

---

### Diagram Description

*CGC architecture with shared and task-specific experts*

- **Input** feeds into:
- **Shared Experts** (Expert 1, Expert 2) and **Task-Specific Experts** (Task-A Expert, Task-B Expert)
- **Gate A** weights the outputs of Shared Experts + Task-A Expert via a learned softmax selector.
- **Gate B** weights the outputs of Shared Experts + Task-B Expert via a learned softmax selector.
- The weighted representations are combined via einsum and fed into **Decoder A** and **Decoder B**.
- Decoders produce final task outputs (**ŷ_A** and **ŷ_B**).

---

# Progressive Layered Extraction (PLE)

*Reference: Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations" — ACM RecSys 2020 (Best Paper)*

---

## Architecture & Mechanism

- Applies the shared + task-specific expert pattern **at every layer** of the backbone, not just the top.
- The encoder (ResNet) is decomposed into 5 stages: `conv1`, `layer1`, `layer2`, `layer3`, `layer4`.
- At each stage, there are **K shared layer-modules** and **M_t task-specific layer-modules** per task.
- A per-task gate at each stage selects over the shared + task-specific outputs for that stage.
- The gated output becomes the input to the **next stage's** experts — progressive, multi-level extraction.

## Benefit & Outcome

- Shared and task-specific features are separated at **every abstraction level** (low-level to high-level).
- Each task's pathway through the network is independently gated at every stage.
- More fine-grained control than CGC — tasks can share early features but diverge later (or vice versa).
- Empirically outperforms both MMoE and CGC on multi-domain recommendation tasks.

---

## LibMTL Implementation

- Decomposes the ResNet encoder into 5 sequential stages and creates shared + task-specific copies at each stage.
- A separate linear-softmax gate at each stage learns expert weights for each task over the combined pool.
- Gated representations flow progressively: each stage's output feeds the next stage's experts for multi-level extraction.

---

### Diagram Description

*PLE architecture with progressive multi-stage extraction*

- **Input** feeds into **Stage 0** (conv1+bn+relu+pool), which contains:
- **Shared Layers** (Layer[0]₁, Layer[0]₂) and **Task-Specific Layers** (Task-A Layer[0]₁...₂, Task-B Layer[0]₁...₂)
- **Gate A[0]** and **Gate B[0]** weight the shared + task-specific outputs at Stage 0.
- The gated representations flow into **Stage 1** (layer1), which repeats the same pattern with its own shared and specific layers.
- **Stages 2–4** (layer2, layer3, layer4) continue the progressive extraction identically.
- After **Stage 4**, the final representations pass through **Decoder A** and **Decoder B** to produce **ŷ_A** and **ŷ_B**.

---

## CGC vs. PLE — Quick Comparison

| | CGC | PLE |
|---|-----|-----|
| **Gating levels** | 1 (top-level only) | 5 (one per ResNet stage) |
| **Expert granularity** | Full encoders | Layer-group modules per stage |
| **Parameter count** | Higher (full encoder copies) | More fine-grained per-layer copies |
| **Multi-input support** | Yes | No |
| **Backbone requirement** | Any encoder class | ResNet only |
| **LibMTL base class** | `MMoE` | `AbsArchitecture` (direct) |

### Key Limitations

- CGC: single-level gating — lower-layer features cannot benefit from shared/specific separation.
- PLE: ResNet-only (hard-coded `conv1`, `bn1`, `relu`, `maxpool`, `layer1`–`layer4` decomposition); no multi-input support.
