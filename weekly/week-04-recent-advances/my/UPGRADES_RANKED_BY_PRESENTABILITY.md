# Task Relationship Upgrades — Ranked by Presentation Impact

> For presenting to seniors. Ranking criteria: visual elegance, conceptual clarity, novelty for speech, and how well it tells a story.

---

## Tier 1: Show-Stoppers (lead with these)

These are the slides that make people lean forward. Each one produces a visual artifact that is instantly understandable and has never been shown for speech before.

### 1. Task Affinity Matrix (Grad-TAG / TAG)

**The slide**: A single 3×3 heatmap. Rows and columns: Content, Speaker, Emotion. Green = positive affinity (tasks help each other). Red = negative affinity (tasks compete). One glance tells the whole story.

**Why it's elegant**: Zero equations needed. A color tells you whether two speech tasks should be trained together or apart. Hypothesis: Content↔Speaker is green (+0.6), Content↔Emotion is red (−0.2). If true, the visual IS the contribution.

**The story**: "We ran one training pass, extracted gradients, and computed this. For the first time, we can SEE which speech tasks help each other. Content and speaker reinforce each other. Emotion is the outlier — it actually competes with content. This single matrix explains why equal-weight training underperforms."

**Visual artifact**: A 3×3 heatmap with annotated values. Side-by-side: before (equal weights, assumed all related) vs after (data-driven affinities, emotion is isolated).

**Code**: ~50 lines in trainer, hooks into `loss.backward()`. Output: one matplotlib heatmap saved to results.

---

### 2. Cross-Task Attention Graph (MTI-Net style)

**The slide**: Three task nodes (Content, Speaker, Emotion) connected by directed edges. Edge thickness = learned attention weight. A dynamic visualization — show the graph at epoch 1 vs epoch 50. At epoch 1, edges are uniform. By epoch 50, Content↔Speaker is thick, Content→Emotion is thin.

**Why it's elegant**: It's a graph. Everyone understands graphs. The edges ARE the task relationships, learned from data. You can animate it over training epochs.

**The story**: "What if each task could decide how much to listen to every other task? We add cross-task attention before the classifier heads. Content attends to Speaker's features. Emotion attends to Content. The attention weights are learned — and they're interpretable. Look at epoch 50: Emotion barely listens to Content. The model discovered on its own what Grad-TAG told us analytically."

**Visual artifact**: Animated or side-by-side graph. Nodes = tasks, edges = learned attention weights. Epoch progression shows the model discovering task structure.

**Code**: ~100 lines. New `CrossTaskAttention` module between hidden layer and classifiers. Returns attention weights for visualization.

---

### 3. Learned Task Correlation Matrix (Kanakis et al., ECCV 2020)

**The slide**: A symmetric 3×3 matrix Ω with values in [−1, 1]. Ω_ij = how correlated the classifier weights of task i and task j become. Below it: the BEFORE (independent heads, Ω = identity) and AFTER (correlated heads, Ω learned).

**Why it's elegant**: It's the simplest possible upgrade — one 3×3 parameter matrix added to the classifier heads. The matrix IS the result. No new layers, no new loss terms. Just "let the heads talk to each other through a learned correlation."

**The story**: "Currently, wavCSE's three classifier heads are independent. Content's head doesn't know Speaker's head exists. We add one 3×3 matrix Ω. During training, Ω_ij learns how much head i should borrow from head j's weights. The result: a correlation matrix we can read directly. If Ω_content,emotion ≈ 0, the model confirms they need separate representations. If Ω_content,speaker ≈ 0.7, they share statistical strength."

**Visual artifact**: The Ω matrix as a heatmap. Side-by-side with the Grad-TAG affinity matrix — do they agree? If yes, two independent methods confirm the same task structure. Powerful.

**Code**: ~30 lines. Add `nn.Parameter(torch.eye(3))` to DownstreamMultiTaskModel. Modify classifier forward to use correlated weights.

---

### 4. Layer-Wise Conflict Heatmap (Recon)

**The slide**: A horizontal bar representing wavCSE's architecture (WavLM layers 1-24 on the left, projector, layer pooling, hidden layer, three heads on the right). Each bar segment is colored by S-Conflict score: green (low conflict, safe to share) through orange to red (high conflict, should be task-specific).

**Why it's elegant**: It's a diagnostic. Like a thermal camera for your neural network. You point it at wavCSE and it tells you exactly where the problems are. Hypothesis: WavLM layers 1-20 are green (general speech features), layers 21-24 start showing orange (task-specific features emerging), the hidden layer is red (conflict central).

**The story**: "We don't have to guess which layers should be shared vs task-specific. Recon computes a per-layer conflict score from gradient signals. Green means share — all three tasks benefit from those features. Red means split — the tasks are fighting over those parameters. For our architecture, the frozen WavLM is mostly green. The trainable hidden layer? Red. That tells us exactly where to add task-specific capacity."

**Visual artifact**: The architecture diagram as a heatmap. One look shows where sharing works and where it doesn't. This is the kind of figure that goes in a paper.

**Code**: ~120 lines. Add gradient hooks per layer, compute cosine similarity of per-task gradients, accumulate S-Conflict scores, generate heatmap.

---

## Tier 2: Theoretically Elegant (strong supporting slides)

These don't produce a single visual artifact, but the ideas are mathematically beautiful and the "before vs after" story is clean.

### 5. Nash-MTL — "The Bargaining Game"

**The slide**: Left: three stick figures labeled Content, Speaker, Emotion pulling a rope in three directions. Label: "Equal weighting — Speaker (strongest) wins." Right: same three figures sitting at a table with a pie chart. Label: "Nash bargaining — each task gets a fair share."

**Why it's elegant**: Game theory is inherently compelling. "Bargaining" is a metaphor anyone understands. The math is clean — solve a small quadratic program each step, get scale-invariant weights. The speaker's 1251 classes no longer dominate just because its gradient is bigger.

**The story**: "Equal weighting is like letting the strongest person at the table take what they want. Speaker has 1251 classes. Its gradient is 100× larger than Emotion's. Nash-MTL reframes the problem: each training step is a bargaining round. Every task gets a weight proportional to how much it would benefit, not how loud it shouts. The result: Emotion's accuracy goes up without hurting Speaker."

**Visual artifact**: Before/after bar chart: per-task accuracy under equal weighting vs Nash-MTL. The Emotion bar visibly rises while Content and Speaker stay flat or improve slightly.

**Code**: ~100 lines. Solve a small QP per gradient step. Already in LibMTL.

---

### 6. CAGrad — "The Consensus Seeker"

**The slide**: A 2D vector diagram. Three colored arrows (task gradients) pointing in different directions. Left: "Averaging — the average direction (gray) satisfies no one." Right: "CAGrad — finds a direction inside the circle that improves the worst-off task."

**Why it's elegant**: The geometry is intuitive. You can draw it. "Stay close to the average, but tilt toward the task that's struggling." The single hyperparameter c (ball radius) has a physical meaning: c=0 is pure averaging, c=1 is maximum conflict resolution.

**The story**: "CAGrad asks: what if we find a gradient direction within a small ball around the average that helps the worst-performing task? It's a min-max optimization with a closed-form solution. One parameter to tune. At c=0.3, it nudges the update toward helping Emotion without derailing Content or Speaker. The result: Pareto improvements — one task gets better, others stay the same."

**Visual artifact**: The vector diagram. Before/after loss curves showing more stable convergence.

**Code**: ~60 lines. Closed-form gradient aggregation. Simpler than Nash-MTL.

---

## Tier 3: Essential but Supporting (mention briefly)

These are important for rigor but won't make the most exciting slides.

### 7. PCGrad — "Gradient Surgery"

The baseline. "When two task gradients point in opposite directions, we project out the conflicting component." Simple, visualizable with before/after vector arrows. Use as the "this is where we started" comparison point.

**Code**: ~80 lines. In LibMTL.

### 8. RLW — "The Random Baseline"

"Here's the sanity check: we randomly sample task weights each batch from a Dirichlet distribution. If our sophisticated method doesn't beat random weighting, it's not worth the complexity." A single slide showing RLW as the horizontal line that all methods must clear.

**Code**: ~10 lines.

### 9. Affinity-Over-Time Animation

Plot the pairwise gradient cosine similarity (Content-Speaker, Content-Emotion, Speaker-Emotion) as three time series over training epochs. Shows when task relationships stabilize. Animated line chart.

**Code**: ~40 lines, hooks into the logging already in trainer_utils.py.

### 10. Layer Duplication (Recon — action step)

The engineering follow-through on the diagnostic from #4. "Based on the conflict heatmap, we duplicate the high-conflict hidden layer into three task-specific copies. This is the fix." Less elegant conceptually but the logical conclusion.

**Code**: ~150 lines.

---

## Recommended Slide Deck Flow

```
Slide 1:  The Affinity Matrix (#1)           ← Hook: "Look what we discovered"
Slide 2:  The Conflict Heatmap (#4)          ← Diagnostic: "Here's WHERE the problem is"
Slide 3:  Cross-Task Attention (#2)          ← Solution A: "Let tasks learn to listen"
Slide 4:  Task Correlation Matrix (#3)       ← Solution B: "Or, one simple matrix"  
Slide 5:  Nash Bargaining (#5)               ← Theory: "Why equal weighting is unfair"
Slide 6:  CAGrad Geometry (#6)               ← Theory: "How to tilt the gradient"
Slide 7:  Before/After Results              ← The payoff
Slide 8:  The Full Pipeline                 ← How everything fits together
```

This tells a story: **Discover** (slides 1-2) → **Solve** (slides 3-4) → **Understand** (slides 5-6) → **Validate** (slide 7).

---

## The 30-Second Elevator Pitch

"wavCSE trains three speech tasks with equal weighting. We have no idea if they help or hurt each other. I found 45 post-2020 papers on task relationship learning. The key insight: we can compute a task affinity matrix from gradient signals — a single 3×3 heatmap that shows content and speaker reinforce each other, while emotion competes with both. Armed with this, we replace equal weighting with Nash bargaining, add cross-task attention between heads, and use layer-wise conflict analysis to decide what to share. Four elegant upgrades, each producing an interpretable visual, none requiring us to retrain the backbone. First time this has been done for speech. ACL 2027."
