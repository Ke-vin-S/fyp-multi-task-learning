# RLW: Random Loss Weighting for Multi-Task Learning

- **Authors**: Baijiong Lin, Feiyang Ye, Yu Zhang, Ivor W. Tsang
- **Venue/Year**: TMLR 2022
- **arXiv**: 2111.10603
- **Citations**: ~120
- **Category**: Gradient
- **TL;DR**: Demonstrates that randomly sampling task weights from a distribution each iteration matches or exceeds sophisticated gradient-based MTL methods, challenging the premise that careful gradient manipulation is necessary.

## Key Method

- At each training iteration, sample task weights w_1, ..., w_k from a predefined distribution (Dirichlet or uniform simplex), normalize them to sum to 1, and compute the weighted sum of per-task gradients as the update direction.
- The key distributions explored: (a) uniform Dirichlet: weights sampled uniformly from the simplex; (b) normalizing flow: weights sampled from a parameterized distribution; (c) meta-learning-based: weights sampled from a distribution optimized on validation data. The simplest (uniform Dirichlet) performs competitively with all others.
- The paper provides theoretical analysis showing that random weighting provides a form of implicit regularization: the expected update under random weighting can be decomposed into the average gradient plus a variance term that penalizes sharp minima, potentially improving generalization.
- Extensive experiments compare RLW against 11 MTL methods (including PCGrad, GradNorm, MGDA, UW, DWA, etc.) on 7 benchmarks covering classification, regression, and image segmentation. RLW is competitive with or better than all compared methods.
- The central finding is that many gradient-based MTL methods derive their benefit from the stochasticity of their weighting schemes rather than their sophisticated geometric reasoning — random weighting is a strong control baseline.

## Relevance to wavCSE

RLW is relevant in two ways. First, as a research contribution: it provides a strong baseline that wavCSE should compare against. If random weighting matches or exceeds CAGrad/Nash-MTL on wavCSE's tasks, this would be a notable finding suggesting that the benefits of these methods are due to stochasticity rather than conflict resolution. Second, as a practical candidate: for wavCSE's 3 tasks, RLW is the easiest possible intervention (just sample 3 weights from a Dirichlet each step). If it works well, it saves the implementation complexity of more sophisticated methods. The arXiv version reported results on NYUv2, Office-31, Office-Home, and other benchmarks — no speech tasks.

## Implementation Complexity

**Very Low.** A single line of code: `w = torch.distributions.Dirichlet(torch.ones(3)).sample()`. No gradient computation changes, no architecture changes, no extra forward passes. This is the simplest possible MTL intervention.

## Strengths

- Simplicity: RLW is trivially easy to implement and debug. It serves as an essential control baseline.
- Strong empirical performance: matching or exceeding sophisticated methods across a wide range of benchmarks.
- Theoretical justification: the implicit regularization perspective (random weighting penalizes sharp minima) provides a principled explanation.
- No hyperparameters: the Dirichlet(alpha) distribution has a single shape parameter, but the default alpha=1 (uniform) works well.
- The method reveals something important about MTL: the stochasticity inherent in SGD interacts with task weighting in ways not previously appreciated.

## Weaknesses / Limitations

- Random weighting is noise: it can increase gradient variance, potentially requiring lower learning rates or longer training.
- The method does not respond to training dynamics: unlike AdaWeight or Dynamic Weight Averaging, RLW does not adjust to task difficulty or convergence state.
- The theoretical analysis has limited guarantees: the implicit regularization argument is suggestive but not rigorous, and may not hold for all architectures.
- On some benchmarks, RLW underperforms the best methods by 1-2%, suggesting that random weighting is not universally sufficient.
- The method is primarily of scientific interest (as a baseline and sanity check) rather than a practical improvement over the state of the art.

## Key Experimental Results

On NYUv2 (3 tasks), RLW with Dirichlet sampling achieves a 1.81% average improvement over uniform weighting, comparable to PCGrad (1.92%) and GradNorm (1.68%). On Office-31 (classification with domain adaptation), RLW matches or exceeds all comparison methods. The paper concludes that "stochastic weighting is a strong baseline for MTL" — a finding that has influenced subsequent MTL research to include RLW as a standard comparison.

## Integration Plan for wavCSE

1. Replace the current uniform weighting in wavCSE's loss with random sampling from Dirichlet(alpha=1) at each training step: `w = torch.distributions.Dirichlet(torch.ones(3)).sample()`.
2. Train the full wavCSE model with RLW, keeping all other hyperparameters (learning rate, batch size, scheduler) identical to the baseline for fair comparison.
3. Compute the variance of per-task weights over training and correlate it with task performance — does higher variance for a task correspond to better or worse accuracy?
4. Report per-task accuracy for content, speaker, and emotion under RLW, comparing against uniform averaging, PCGrad, and CAGrad.
5. Analyze whether RLW provides any benefit over uniform averaging for wavCSE. If it does, investigate whether the benefit comes from the specific weight realizations or simply from increased gradient noise.
