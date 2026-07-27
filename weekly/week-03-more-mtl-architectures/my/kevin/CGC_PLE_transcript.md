# CGC & PLE — Presentation Transcript (~4 minutes)

---

## Slide 1: CGC (Content)

CGC stands for Customized Gate Control, from Tang et al., RecSys 2020. It's an extension of MMoE. The key idea is simple: MMoE only has shared experts — all tasks pull from the same pool. CGC adds **task-specific experts** on top. So each task gets its own private expert networks alongside the shared ones.

A per-task gate — just a linear layer followed by softmax — learns to weight the shared experts and that task's own specific experts. The weighted sum gives each task a customized representation, which then goes through its decoder.

The benefit is explicit separation of shared versus task-private capacity. Each task can lean on shared knowledge when it helps, but has its own pathway when it needs independence. Implementation-wise in LibMTL, it inherits from MMoE, so it reuses the shared expert pool and just adds a separate set of task-specific expert networks for each task, then redefines the gates to cover the larger combined pool of shared plus task-private experts.

---

## Slide 2: CGC Diagram

Let me walk through the diagram. At the top we have the **Input**, which feeds into four expert networks in parallel. On the left, the two blue boxes are **Shared Experts** — these process the input for all tasks. On the right, the green box is the **Task-A specific expert** and the orange box is the **Task-B specific expert** — these are private to each task.

Below the experts, **Gate A** takes the outputs from both shared experts plus the Task-A expert, applies a learned softmax weighting, and produces a combined representation for Task A. Similarly, **Gate B** combines the shared experts with the Task-B expert.

Those weighted representations then flow down through the **Decoder** for each task, and finally produce the predictions — **ŷ_A** and **ŷ_B**. The key thing to notice is that the shared experts feed into *both* gates, while each task-specific expert only feeds its own gate.

---

## Slide 3: PLE (Content)

PLE — Progressive Layered Extraction — comes from the same paper and won Best Paper at RecSys 2020. The motivation is that CGC only applies gating once, at the top. But what about lower layers? Different levels of abstraction might benefit from different sharing patterns.

So PLE takes the shared-plus-specific expert idea and applies it **at every layer** of the backbone. In LibMTL, the ResNet is decomposed into 5 stages — conv1 through layer4. At each stage, you have K copies of that stage's layers for shared processing, and M copies for each task. A per-task gate at each stage selects over shared and specific, and the gated output becomes the input for the *next* stage.

This is progressive extraction — the network separates shared from task-specific information at every abstraction level, from low-level features to high-level semantics. The cost is that it's ResNet-only — the decomposition hard-codes ResNet's layer names — and it doesn't support multi-input.

---

## Slide 4: PLE Diagram

This diagram shows the multi-stage flow. **Input** enters at the top and feeds into **Stage 0** — that's conv1 plus batch norm, ReLU, and max pooling. Inside Stage 0, the dashed box contains shared layers in blue, Task-A layers in green, and Task-B layers in orange. The purple diamonds are the per-task gates — Gate A at stage 0 and Gate B at stage 0 — each selecting over the shared and its own specific layers.

The gated output flows down to **Stage 1** — that's ResNet layer1 — which has the exact same structure: its own shared layers, task-specific layers, and gates. This pattern repeats through stages 2 and 3, shown with the dotted lines, and then through **Stage 4** — the final ResNet layer.

After Stage 4's gates, the representations hit the **Decoders** and produce the final outputs. The critical insight is those vertical arrows — each stage's output feeds the next stage's experts, so the separation of shared and specific information compounds progressively through the network.
