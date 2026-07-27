const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "FYP - MTL for Speech";
pres.title = "wavCSE Architectural Improvements";

const C = {
  bg:       "0D1117",
  surface:  "161B22",
  border:   "30363D",
  text:     "C9D1D9",
  muted:    "8B949E",
  white:    "FFFFFF",
  green:    "3FB950",
  purple:   "A371F7",
  blue:     "58A6FF",
  red:      "F85149",
  orange:   "D2991D",
};

const fontTitle = "Calibri";
const fontBody = "Calibri";

function sectionLabel(slide, text, color, y) {
  slide.addText(text, {
    x: 0.6, y: y || 0.4, w: 4, h: 0.35,
    fontSize: 10, fontFace: fontBody, color,
    bold: true, charSpacing: 3, margin: 0,
  });
}

function accentBar(slide, x, y, w, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.04, fill: { color } });
}

function card(slide, x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.surface },
    line: { color: C.border, width: 0.5 },
  });
}

function diagramPlaceholder(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: "FFFFFF", transparency: 95 },
    line: { color: C.border, width: 1, dashType: "dash" },
  });
  if (label) {
    slide.addText(label, {
      x, y, w, h,
      fontSize: 11, fontFace: fontBody, color: C.muted,
      align: "center", valign: "middle",
    });
  }
}

function sectionHeader(slide, y, color, label, title) {
  sectionLabel(slide, label, color, y);
  slide.addText(title, {
    x: 0.6, y: y + 0.45, w: 9, h: 0.6,
    fontSize: 28, fontFace: fontTitle, color, bold: true, margin: 0,
  });
  accentBar(slide, 0.6, y + 1.0, 2.0, color);
}

function bigStatement(slide, text, x, y, w, color) {
  slide.addText(text, {
    x, y, w, h: 1.5,
    fontSize: 22, fontFace: fontTitle, color: color || C.text,
    bold: true, valign: "middle", margin: 0,
  });
}

function oneLiner(slide, text, x, y, w) {
  slide.addText(text, {
    x, y, w, h: 0.35,
    fontSize: 13, fontFace: fontBody, color: C.muted,
    valign: "middle", margin: 0,
  });
}

// ═══════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.5, w: 0.06, h: 1.2, fill: { color: C.green } });

  s.addText("wavCSE Architectural Improvements", {
    x: 0.9, y: 1.5, w: 8.5, h: 0.7,
    fontSize: 38, fontFace: fontTitle, color: C.white, bold: true, margin: 0,
  });
  s.addText("Benchmarking Multi-Task Learning Enhancements for Speech", {
    x: 0.9, y: 2.25, w: 8.5, h: 0.4,
    fontSize: 16, fontFace: fontBody, color: C.muted, margin: 0,
  });

  const badges = [
    { text: "Zhang et al. 2010 (BOMC)", color: C.green },
    { text: "Ciliberto et al. 2015 (SKMTL)", color: C.purple },
    { text: "Goncalves et al. 2016 (MSSL)", color: C.blue },
  ];
  badges.forEach((b, i) => {
    const bx = 0.6 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, {
      x: bx, y: 3.5, w: 2.9, h: 0.45,
      fill: { color: C.surface }, line: { color: C.border, width: 0.5 },
    });
    s.addText(b.text, {
      x: bx, y: 3.5, w: 2.9, h: 0.45,
      fontSize: 10, fontFace: fontBody, color: b.color,
      align: "center", valign: "middle",
    });
  });

  s.addText("FYP — Multi-Task Learning for Speech  |  ACL 2027 Target", {
    x: 0.6, y: 4.8, w: 9, h: 0.3,
    fontSize: 10, fontFace: fontBody, color: C.muted, margin: 0,
  });
}

// ═══════════════════════════════════════════
// SLIDE 2 — PROBLEM STATEMENT
// ═══════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  sectionHeader(s, 0.4, C.orange, "PROBLEM STATEMENT", "Original wavCSE Limitation");

  // One core statement
  bigStatement(s, "Task classification heads are completely independent.\nNo mechanism to share statistical strength or discover\ninter-task relationships.", 0.6, 1.5, 8.8, C.red);

  // Three bullets
  s.addText([
    { text: "Equal loss weighting regardless of task difficulty or data size", options: { bullet: true, breakLine: true, fontSize: 15, color: C.muted } },
    { text: "Potential for negative transfer — no detection or mitigation", options: { bullet: true, breakLine: true, fontSize: 15, color: C.muted } },
    { text: "No interpretable task structure — tasks are black-box silos", options: { bullet: true, fontSize: 15, color: C.muted } },
  ], {
    x: 0.9, y: 3.2, w: 8.5, h: 1.5,
    fontFace: fontBody, valign: "top", margin: 0, paraSpaceAfter: 12,
  });

  diagramPlaceholder(s, 5.5, 1.55, 4.1, 2.4,
    "[ Place original wavCSE architecture diagram ]\nRed = the part we modify");
}

// ═══════════════════════════════════════════
// HELPER: Build an architecture slide pair
// ═══════════════════════════════════════════
function archConceptSlide(shortLabel, paperRef, color, title, oneLinerText, leftTitle, leftBullets, rightTitle, rightBullets) {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  sectionLabel(s, "ARCHITECTURE " + shortLabel + "  /  " + paperRef, color);
  s.addText(title, {
    x: 0.6, y: 0.85, w: 9, h: 0.6,
    fontSize: 28, fontFace: fontTitle, color, bold: true, margin: 0,
  });
  accentBar(s, 0.6, 1.4, 2.5, color);

  // One-liner
  s.addText(oneLinerText, {
    x: 0.6, y: 1.65, w: 9, h: 0.45,
    fontSize: 15, fontFace: fontBody, color: C.text, italic: true, margin: 0,
  });

  // Left column
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 2.3, w: 0.05, h: 0.35, fill: { color } });
  s.addText(leftTitle, {
    x: 0.7, y: 2.3, w: 4.2, h: 0.35,
    fontSize: 16, fontFace: fontBody, color: C.white, bold: true, margin: 0,
  });
  s.addText(leftBullets.map((b, i) => ({
    text: b,
    options: { bullet: true, breakLine: i < leftBullets.length - 1, fontSize: 14, color: C.muted },
  })), {
    x: 0.7, y: 2.8, w: 4.2, h: 2.3,
    fontFace: fontBody, valign: "top", margin: 0, paraSpaceAfter: 12,
  });

  // Right column
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 2.3, w: 0.05, h: 0.35, fill: { color } });
  s.addText(rightTitle, {
    x: 5.6, y: 2.3, w: 4.2, h: 0.35,
    fontSize: 16, fontFace: fontBody, color: C.white, bold: true, margin: 0,
  });
  s.addText(rightBullets.map((b, i) => ({
    text: b,
    options: { bullet: true, breakLine: i < rightBullets.length - 1, fontSize: 14, color: C.muted },
  })), {
    x: 5.6, y: 2.8, w: 4.2, h: 2.3,
    fontFace: fontBody, valign: "top", margin: 0, paraSpaceAfter: 12,
  });

  return s;
}

function archRationaleSlide(shortLabel, color, leftTitle, leftBullets, rightTitle, rightBullets, tradeoff) {
  const s = pres.addSlide();
  s.background = { color: C.bg };

  sectionLabel(s, shortLabel + "  —  WHY USE IT & EXPECTED OUTCOME", color);

  // Left
  s.addShape(pres.shapes.RECTANGLE, { x: 0.4, y: 1.0, w: 0.05, h: 0.35, fill: { color } });
  s.addText(leftTitle, {
    x: 0.7, y: 1.0, w: 4.2, h: 0.35,
    fontSize: 16, fontFace: fontBody, color: C.white, bold: true, margin: 0,
  });
  s.addText(leftBullets.map((b, i) => ({
    text: b,
    options: { bullet: true, breakLine: i < leftBullets.length - 1, fontSize: 14, color: C.muted },
  })), {
    x: 0.7, y: 1.5, w: 4.2, h: 2.6,
    fontFace: fontBody, valign: "top", margin: 0, paraSpaceAfter: 12,
  });

  // Right
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 1.0, w: 0.05, h: 0.35, fill: { color } });
  s.addText(rightTitle, {
    x: 5.6, y: 1.0, w: 4.2, h: 0.35,
    fontSize: 16, fontFace: fontBody, color: C.white, bold: true, margin: 0,
  });
  s.addText(rightBullets.map((b, i) => ({
    text: b,
    options: { bullet: true, breakLine: i < rightBullets.length - 1, fontSize: 14, color: C.muted },
  })), {
    x: 5.6, y: 1.5, w: 4.2, h: 2.6,
    fontFace: fontBody, valign: "top", margin: 0, paraSpaceAfter: 12,
  });

  // Tradeoff bar
  card(s, 0.4, 4.3, 9.4, 0.6);
  s.addText([
    { text: "Tradeoff: ", options: { bold: true, color: C.orange, fontSize: 13 } },
    { text: tradeoff, options: { fontSize: 13, color: C.muted } },
  ], {
    x: 0.7, y: 4.3, w: 8.8, h: 0.6,
    fontFace: fontBody, valign: "middle", margin: 0,
  });

  return s;
}

// ═══════════════════════════════════════════
// SLIDES 3-4: GBC
// ═══════════════════════════════════════════
archConceptSlide("1", "ZHANG ET AL. 2010 (BOMC)", C.green,
  "Global Bias Coupling (GBC)",
  "A shared bias vector couples all task heads — the lightest possible change to the architecture.",

  "Integration into wavCSE",
  [
    "Modifies only the bias term of each task head — backbone, projector, hidden layer, and weight matrices are untouched",
    "Adds a learnable global coupling vector, a projector into the coupling space, and per-task layers that translate it to each task's output",
    "Training loop is identical — no alternating steps, no extra loss terms",
  ],

  "Approach",
  [
    "Zhang et al. showed a global bias acts as a statistical hub — updated by every example, converging faster than any per-class bias",
    "Adapted from multi-label to multi-task: one shared coupling vector instead of one shared scalar",
    "Bias-only coupling — weight matrices stay independent, only intercepts are linked",
  ]);

archRationaleSlide("GBC", C.green,
  "Why Use It",
  [
    "Pools statistical strength — data-rich tasks help stabilize bias estimates for data-poor tasks",
    "Implicit regularization — tasks cannot drift independently through the shared coupling",
    "Minimal overhead — lightest change of all three architectures",
  ],

  "Expected Outcome",
  [
    "Improved accuracy on data-poor tasks through reduced bias variance",
    "More stable convergence across different random seeds",
    "Interpretable coupling weights reveal which tasks benefit most from shared information",
  ],

  "Coupling is implicit and soft — no sparsity, so all tasks are coupled to some degree. Less interpretable structure than TSM or PMR.");

// ═══════════════════════════════════════════
// SLIDES 5-6: TSM
// ═══════════════════════════════════════════
archConceptSlide("2", "CILIBERTO ET AL. 2015 (SKMTL)", C.purple,
  "Task Structure Matrix (TSM)",
  "Replaces independent heads with K shared latent classifiers — each task is a sparse mixture of them.",

  "Integration into wavCSE",
  [
    "Replaces the independent task heads — backbone, projector, hidden layer, and pooling are unchanged",
    "Adds K latent classifiers plus a T x K structure matrix A with L1 sparsity",
    "Training alternates: standard supervised epochs, then periodic unsupervised steps that optimize only A",
  ],

  "Approach",
  [
    "Ciliberto et al. showed multi-task predictors decompose into latent basis functions with a structure matrix",
    "Each latent classifier learns a distinct decision pattern — tasks share the ones they need, ignore the rest",
    "L1 sparsity zeroes out spurious relationships — the non-zero pattern of A IS the discovered task structure",
    "Alternating minimization is provably convergent to a global optimum",
  ]);

archRationaleSlide("TSM", C.purple,
  "Why Use It",
  [
    "Discovers task structure from data — no assumptions about which tasks are related",
    "Explicit negative transfer protection — harmful latent entries are zeroed out by L1",
    "Formal convergence guarantees for the alternating optimization",
  ],

  "Expected Outcome",
  [
    "Interpretable task groupings — rows of A show exactly which latents each task uses",
    "Resilience to heterogeneous tasks — unrelated tasks get isolated to their own latents",
    "Potential parameter efficiency when K < T — shared basis with task-specific mixing",
  ],

  "Replaces the original head architecture. Alternating training adds overhead. Matrix A is asymmetric — no graphical model interpretation.");

// ═══════════════════════════════════════════
// SLIDES 7-8: PMR
// ═══════════════════════════════════════════
archConceptSlide("3", "GONCALVES ET AL. 2016 (MSSL)", C.blue,
  "Precision Matrix Regularization (PMR)",
  "Keeps the forward pass identical — adds a precision matrix that regularizes task parameters through the loss.",

  "Integration into wavCSE",
  [
    "Model architecture is completely unchanged — only the training loss is modified",
    "Adds a learnable precision matrix Omega (via Cholesky) with log-det, trace, and L1 loss terms",
    "PMR loss enabled after a warmup period — heads converge first, then structure is imposed",
  ],

  "Approach",
  [
    "Goncalves et al. modeled task parameters as draws from a Gaussian with precision Omega",
    "Omega_ij = 0 means tasks i and j are conditionally independent — a rigorous statistical statement",
    "Task head weights are flattened into a parameter matrix W; Omega couples them through the trace term",
    "Cholesky parameterization guarantees Omega is PSD without expensive ADMM projections",
  ]);

archRationaleSlide("PMR", C.blue,
  "Why Use It",
  [
    "Formal probabilistic model — conditional independence, not just heuristic coupling",
    "Inference-time identical to original — purely a training regularizer",
    "Warmup prevents fitting spurious structure to random initial parameters",
  ],

  "Expected Outcome",
  [
    "Visualizable task dependency graph — threshold Omega to reveal conditional dependence edges",
    "Controlled negative transfer — unrelated tasks get zero Omega entries, meaning no coupling force",
    "Reproducible structure discovery — Omega is a research artifact comparable across runs",
  ],

  "Omega scales quadratically — negligible for 3 tasks. Only regularizes parameters, doesn't change what the model can represent.");

// ═══════════════════════════════════════════
// SLIDE 9 — COMPARISON
// ═══════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: C.bg };

  sectionLabel(s, "COMPARISON & NEXT STEPS", C.white);

  s.addText("Cross-Architecture Summary", {
    x: 0.6, y: 0.85, w: 9, h: 0.6,
    fontSize: 28, fontFace: fontTitle, color: C.white, bold: true, margin: 0,
  });
  accentBar(s, 0.6, 1.4, 2.5, C.white);

  const cols = [
    { name: "GBC", color: C.green, paper: "Zhang 2010", items: [
      "Integration: Bias term only",
      "Coupling: Global bias hub",
      "Training: Same as original",
      "Structure: Implicit (soft)",
      "Best for: Minimal change",
    ]},
    { name: "TSM", color: C.purple, paper: "Ciliberto 2015", items: [
      "Integration: Replaces task heads",
      "Coupling: Latent classifier mix",
      "Training: Alternating steps",
      "Structure: Sparse matrix A",
      "Best for: Task discovery",
    ]},
    { name: "PMR", color: C.blue, paper: "Goncalves 2016", items: [
      "Integration: Loss function only",
      "Coupling: Precision matrix Omega",
      "Training: Extra loss term",
      "Structure: Dependency graph",
      "Best for: Formal model",
    ]},
  ];

  cols.forEach((col, ci) => {
    const cx = 0.4 + ci * 3.15;
    s.addShape(pres.shapes.RECTANGLE, {
      x: cx, y: 1.6, w: 3.0, h: 0.5,
      fill: { color: col.color, transparency: 80 },
      line: { color: col.color, width: 0.5 },
    });
    s.addText([
      { text: col.name + "  ", options: { bold: true, color: col.color, fontSize: 16 } },
      { text: col.paper, options: { color: C.muted, fontSize: 9 } },
    ], { x: cx, y: 1.6, w: 3.0, h: 0.5, fontFace: fontBody, align: "center", valign: "middle", margin: 0 });

    col.items.forEach((item, ii) => {
      const iy = 2.3 + ii * 0.45;
      s.addShape(pres.shapes.RECTANGLE, {
        x: cx, y: iy, w: 3.0, h: 0.38,
        fill: { color: C.surface }, line: { color: C.border, width: 0.3 },
      });
      s.addText(item, {
        x: cx + 0.15, y: iy, w: 2.7, h: 0.38,
        fontSize: 10, fontFace: fontBody, color: C.text, valign: "middle", margin: 0,
      });
    });
  });

  card(s, 0.4, 4.7, 9.4, 0.7);
  s.addText([
    { text: "Next: ", options: { bold: true, color: C.white, fontSize: 13 } },
    { text: "Run all three on ks_si_er with consistent seeds  |  Compare accuracy, training dynamics, interpretability  |  Ablation: test without hidden layer expansion", options: { fontSize: 13, color: C.muted } },
  ], { x: 0.7, y: 4.7, w: 8.8, h: 0.7, fontFace: fontBody, valign: "middle", margin: 0 });
}

// ── Write ──
const outPath = "/home/kevin/projects/fyp/wavCSE/improvements/docs/wavcse-architectural-improvements.pptx";
pres.writeFile({ fileName: outPath }).then(() => console.log("Written to:", outPath));
