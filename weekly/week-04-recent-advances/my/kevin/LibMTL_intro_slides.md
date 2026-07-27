# Slide 1: LibMTL — A Python Library for Deep Multi-Task Learning

## LibMTL Overview (Lin & Zhang, JMLR 2023)

### What is LibMTL?
- A **unified, modular Python library** for deep multi-task learning
- Implements **24 weighting strategies** + **8 architectures** in a composable design
- Built on PyTorch, fully open-source (v1.1.5)
- Published in JMLR (Journal of Machine Learning Research) 2023

### Design Philosophy: Two-Axis Separation

LibMTL cleanly separates MTL into two **orthogonal, composable axes**:

```
┌─────────────────────────────────────────────────────────┐
│                     LibMTL                               │
│                                                         │
│   ┌───────────────────┐    ┌─────────────────────────┐  │
│   │   ARCHITECTURE    │    │      WEIGHTING          │  │
│   │                   │    │                         │  │
│   │ How to share?     │    │ How to balance losses?  │  │
│   │ (network design)  │    │ (optimization design)   │  │
│   │                   │    │                         │  │
│   │ • HPS             │    │ • EW  • UW  • DWA      │  │
│   │ • Cross-Stitch    │    │ • GLS • RLW             │  │
│   │ • MMoE            │    │ • MGDA  • PCGrad        │  │
│   │ • CGC             │◄──►│ • GradNorm • GradDrop   │  │
│   │ • PLE             │    │ • CAGrad • IMTL         │  │
│   │ • MTAN            │    │ • Nash-MTL • MoCo       │  │
│   │ • DSelect-k       │    │ • FAMO • MoDo • ...     │  │
│   │ • LTB             │    │                         │  │
│   └───────────────────┘    └─────────────────────────┘  │
│                         │                               │
│                  COMPOSE via                             │
│           Multiple Inheritance                          │
│                         │                               │
│         ┌───────────────▼──────────────┐                │
│         │      MTLmodel(Arch, W)       │                │
│         │   arch.forward()  →  preds   │                │
│         │   weight.backward() → grads  │                │
│         └──────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Core Idea
Any of the 24 weighting methods can be paired with any of the 8 architectures — **192 possible combinations** from a single config change.

---

# Slide 2: LibMTL — Package Structure

## How the Library is Organized

```
LibMTL/
├── config.py          # CLI argument parsing + prepare_args()
├── trainer.py         # Central orchestrator (train/test loop)
├── loss.py            # AbsLoss, CELoss, L1Loss, MSELoss
├── metrics.py         # AbsMetric, AccMetric, L1Metric
├── _record.py         # Performance tracking per epoch
├── utils.py           # set_random_seed, set_device, count_parameters
│
├── architecture/      # "How to share?" — 8 MTL network designs
│   ├── abstract_arch.py
│   ├── HPS.py, Cross_stitch.py, MMoE.py, MTAN.py
│   ├── CGC.py, PLE.py, LTB.py, DSelect_k.py
│
├── weighting/         # "How to balance?" — 24 loss/gradient strategies
│   ├── abstract_weighting.py
│   ├── EW.py, UW.py, GradNorm.py, MGDA.py, DWA.py, GLS.py
│   ├── PCGrad.py, GradDrop.py, IMTL.py, GradVac.py, CAGrad.py
│   ├── Nash_MTL.py, RLW.py, MoCo.py, Aligned_MTL.py
│   ├── DB_MTL.py, STCH.py, ExcessMTL.py, FairGrad.py
│   ├── FAMO.py, MoDo.py, SDMGrad.py, UPGrad.py
│
└── model/             # Backbone networks
    ├── resnet.py           # ResNet-18/34/50/101/152 + variants
    └── resnet_dilated.py   # Dilated ResNet for dense prediction
```

---

# Slide 3: LibMTL — Composition & Data Flow

## How Architecture + Weighting Compose at Runtime

### Step 1: Dynamic Class Creation (`trainer.py:111`)

```python
# Multiple inheritance composes Arch + Weighting at runtime
class MTLmodel(architecture_class, weighting_class):
    def __init__(self, ...):
        super().__init__(...)   # chains through both base classes
        self.init_param()       # init any trainable weighting params
```

Python MRO ensures:
- `architecture_class.__init__` → sets up encoder, decoders, expert networks
- `weighting_class.__init__` → sets up weighting-specific state

### Step 2: Training Loop (single backward pass)

```
┌──────────────────────────────────────────────────────────────┐
│                   One Training Iteration                      │
│                                                              │
│  1. Forward Pass                                             │
│     inputs ──► Architecture.forward() ──► {task: pred}       │
│                                                              │
│  2. Loss Computation                                         │
│     {task: pred} + {task: gt} ──► loss_fn ──► train_losses  │
│                                                              │
│  3. Weighted Backward                                        │
│     train_losses ──► Weighting.backward() ──► combined grads │
│     (calls _get_grads → _backward_new_grads internally)      │
│                                                              │
│  4. Optimizer Step                                           │
│     optimizer.step()                                         │
└──────────────────────────────────────────────────────────────┘
```

### Step 3: The Weighting Method Interface

```python
class AbsWeighting:
    def backward(self, losses, **kwargs):
        """
        Input:  losses = [loss_task1, loss_task2, ...]   (Tensor)
        Output: np.array of task weights or None

        Internally calls:
          - _get_grads(losses)       → per-task gradient vectors
          - _backward_new_grads(w)   → apply weighted combination
        OR
          - combined_loss.backward() → simple sum-based methods
        """
```

### Architecture Interface

```python
class AbsArchitecture:
    def forward(self, inputs, task_name=None):
        """
        Input:  inputs (batch tensor)
        Output: {task_name: prediction_tensor}

        Pattern:
          shared_repr = self.encoder(inputs)
          return {task: self.decoders[task](shared_repr)
                  for task in self.task_name}
        """
```

---

# Slide 4: LibMTL — Extension Points

## How Users Extend LibMTL for Custom Research

### Three Extension Points

```
                    ┌──────────────────┐
                    │   User's Code     │
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ Custom Weighting │ │ Custom Arch  │ │ Custom Loss/Metric│
│                  │ │              │ │                  │
│ Subclass         │ │ Subclass     │ │ Subclass         │
│ AbsWeighting     │ │AbsArchitecture│ │ AbsLoss/AbsMetric│
│                  │ │              │ │                  │
│ Implement:       │ │ Implement:   │ │ Implement:       │
│ backward(losses) │ │ forward()    │ │ compute_loss()   │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

### Example: Custom Weighting Method

```python
from LibMTL.weighting import AbsWeighting

class MyWeighting(AbsWeighting):
    def backward(self, losses, **kwargs):
        # 1. Get per-task gradients
        grads = self._get_grads(losses, mode='param_grad')

        # 2. Compute task weights (your algorithm)
        weights = my_algorithm(grads)

        # 3. Apply weighted gradients
        self._backward_new_grads(weights)

        return weights.cpu().numpy()
```

### Example: Custom Architecture

```python
from LibMTL.architecture import AbsArchitecture

class MyArch(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders,
                 rep_grad, multi_input, device, **kwargs):
        super().__init__(task_name, encoder_class, decoders,
                         rep_grad, multi_input, device, **kwargs)
        # Custom network components here

    def forward(self, inputs, task_name=None):
        # Custom forward logic
        feat = self.encoder(inputs)
        return {task: self.decoders[task](feat)
                for task in self.task_name}
```

### Key Configuration Arguments

| Argument | Purpose | Example |
|----------|---------|---------|
| `--weighting` | Loss balancing method | `EW`, `CAGrad`, `MGDA` |
| `--arch` | MTL architecture | `HPS`, `MMoE`, `CGC`, `PLE` |
| `--rep_grad` | Gradient mode | `True` (rep-grad) or `False` (param-grad) |
| `--multi_input` | Separate inputs per task | `True` or `False` |
| `--img_size` | Input dimensions (CGC/PLE) | `3 288 384` |
| `--num_experts` | Expert counts (CGC/PLE) | `2 1 1` (shared + per-task) |

### Two Gradient Modes

| Mode | `rep_grad=True` | `rep_grad=False` |
|------|-----------------|-------------------|
| **Gradient computed w.r.t.** | Shared representation | Shared parameters |
| **Efficiency** | More efficient | Standard |
| **Methods supported** | EW, UW, MGDA, PCGrad, GradVac, CAGrad | All methods |
