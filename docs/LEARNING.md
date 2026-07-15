# LEARNING — dim-stage

*Teaching notes, appended at each stage close. Plain English first; every
jargon term defined at first use. The goal: Kyle can defend every decision.*

## M0 — the fit pilot (2026-07-15)

### The one-paragraph story

We built our own copy of the paper's **Jacobian lens** — a tool that reads a
transformer's *middle* layers by first transporting each middle-layer vector
into the final layer's coordinate system (multiplying by `J_l`, the average
sensitivity matrix) and then decoding it with the model's own output matrix.
We proved our copy computes the *exact same object* as Anthropic's reference
implementation (bit for bit), fitted it to both small Qwen models, and then
asked the frozen question: can this lens recover known "workspace content"
(an evoked concept, a latent reasoning step, a planned rhyme) from the middle
layers at least half the time? **Answer: no, on both models — a clean,
pre-registered null.** That null is the headline result the kickoff predicted
might happen, and it now reshapes the rest of the project.

### Vocabulary that now has meaning from experience

- **Residual stream** — the running vector each transformer layer reads and
  adds to; one per token position. The thing the lens reads.
- **Cotangent / VJP** — backprop's "direction of interest": put a 1 in one
  output coordinate, run backward, and you get that coordinate's sensitivity
  to every input — a **v**ector–**J**acobian **p**roduct. Build `J_l` one row
  (or 8 rows — `dim_batch`) at a time this way.
- **Frobenius norm** — a matrix's overall size (square root of the sum of
  squared entries). "Relative Frobenius distance 0" = the matrices are
  identical.
- **Wilson interval** — the honest range a true rate sits in given only n
  trials; never escapes [0,1]. Our gate turned on its *lower bound*: "even
  pessimistically, is the rate ≥ 0.5?"
- **Newcombe interval** — the same idea for a *difference* between two rates;
  if it straddles 0, no gap may be claimed. This is what made "J-advantage"
  a claim on typo at 0.5B (+43pp, CI [+31, +53]) and *forbade* it at 1.5B.
- **Logit lens (`J = I`)** — the old trick of decoding a middle layer
  directly, no transport. Kept as a standing falsification arm: any J-lens
  claim must beat "doing nothing."
- **Forking paths** — tuning analysis choices after seeing results until
  something passes. Every convention here (band, synonym table, token
  variants, tolerance) was frozen *before* the first official number existed.

### Hard-won, non-obvious lessons

1. **MPS fp32 backward is bitwise deterministic** (this machine, torch
   2.13.0): two reference refits produced *identical* matrices, so the
   run-to-run "noise floor" was exactly 0 and our pre-declared 1e-4 stand-in
   carried the tolerance. Determinism is not hand-waving here — we measured it.
2. **The AGREE gate result was stronger than required** — bitwise identity,
   not just within-tolerance. Two independently written codebases hitting the
   same bits is what "the estimator spec was extracted correctly" looks like.
3. **Batching knobs can be math-neutral and still 25× slower** — dim_batch=32
   replicated the retained graph past MPS unified-memory locality. Measure,
   don't assume.
4. **Even CPU-side work slows an MPS job** (~8% on our fit) — GPU kernels are
   driven by CPU threads. Serial beats concurrent for measurements.
5. **A null with structure beats a bare null.** Abstract-content
   distributions were hard zeros at both scales; surface-adjacent ones sat at
   33–54%; and the Jacobian correction's value *inverted* with scale (strongly
   positive at 0.5B on typo/multilingual, significantly *negative* on typo at
   1.5B as the plain logit lens caught up). Those are reportable observations
   even though no gate claim is made.
6. **Grading conventions the paper doesn't ship are decisions you own.** The
   intermediate→token mapping (leading-space variants, synonym table) had to
   be invented — so it was frozen in code, pre-declared, and rowed in the
   deviations table before any run.

### Recall questions (answers in this repo's docs)

1. Why did the AGREE tolerance need a *measured* noise floor instead of a
   guessed one — and what did we do when the floor came back exactly 0?
2. The 1.5B typo cell shows the J-lens at 17.7% and the logit lens at 44.8%.
   Why can we *report* that reversal but not claim "the J-lens hurts" as a
   gated finding? (Hint: which arm was the gate pre-registered on?)
3. Why does the readability verdict use the Wilson lower bound rather than
   the raw pass@10 percentage — what failure mode does that choose to avoid?
