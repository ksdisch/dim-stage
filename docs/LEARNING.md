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

## M0 — 3B escalation (2026-07-15 evening)

### The one-paragraph story

Both small models came back null, which was the *pre-registered trigger* to try
one bigger model (3B) — the only way to ask "does the workspace become readable
if we add scale?" We froze the 3B analysis band **before** fitting (so we
couldn't move the goalposts after seeing results), discovered the laptop
physically can't fit a 3B model in full precision, rented a cloud GPU for **83
cents** to do just the fit, then graded locally exactly like the other two. The
answer: **still null.** No distribution's readout clears the bar at 0.5B, 1.5B,
or 3B. That's the strongest honest version of this project's headline — a
clean, three-scale, pre-registered *no* to the paper's open question, at the
scales a hobbyist can reach.

### What was learned

7. **A memory ceiling is a hardware fact you measure, not guess.** The Mac's
   24 GB of unified memory (shared by CPU and GPU) holds a 1.5B model's
   full-precision weights (6.2 GB) with room for the backward pass, but a 3B
   model's weights (12.4 GB) sit right at the edge — and the backward pass tips
   it over a "working-set cliff" where the GPU thrashes and runs ~25× slower.
   Halving the batch didn't help because the *weights themselves*, not the
   batch, are what overflow. Lesson: **fp32 fitting on this machine tops out
   between 1.5B and 3B** — a boundary now written into memory, so no future
   session re-discovers it by burning an evening.
8. **Rent, don't fight, when the tool is wrong for the job.** A datacenter GPU
   (CUDA, no such cliff) fit the 3B lens in ~57 minutes at 34.5 s/prompt — the
   same prompt the Mac couldn't finish in 40. The whole detour was ~$0.83 and an
   *owned deviation row*, not a silent move: cross-device numeric noise on a
   corpus-averaged matrix is ~1e-7, orders below any gate's sensitivity, and the
   fit was byte-identical procedure (same code, same prompts). We verified the
   returned file by checksum before trusting it.
9. **A null that survives escalation is stronger than the null that triggered
   it.** The double null was a headline; the triple null *closes the question* —
   it says the emergence point, if one exists, is above 3B, outside the reachable
   range. And the structure sharpened: association and poetry (the abstract,
   not-yet-spoken content — the most workspace-like) are hard zeros at *all
   three* scales, while the Jacobian correction's value turned out **content-
   dependent, not a scale trend** (typo's reversal deepens with scale; order-ops
   *regains* a J-advantage at 3B).
10. **Make the oracle's own output honest.** The gate's console line printed "no
    clear gap" for a statistically-clean *reversal* (the J-lens reading
    significantly *worse* than the plain baseline). We added a distinct
    `J-REVERSAL` label — a finding hidden by a lazy print is still a hidden
    finding, and in a project whose whole worth is faithful measurement, the
    console should never under-report what the data shows.

### Recall questions (answers in this repo's docs)

4. Why did we freeze the 3B band (L14–L32) and merge it *before* the fit ran,
   rather than after we had the lens in hand? What would have been wrong with
   waiting?
5. The 3B fit ran on a rented CUDA GPU while the grading ran on the local Mac.
   Why is that split defensible — what makes the device change a *small* owned
   deviation rather than a result-invalidating one?
6. "association" and "poetry" are hard zeros at 0.5B, 1.5B, *and* 3B. Why is
   that the most interesting part of the null, given what those two
   distributions are asking the lens to read?

## M1 — verbal report + introspection (2026-07-16)

### The one-paragraph story

M1 moved from *reading* the workspace to *writing* it. We built the paper's
two intervention operators — the **swap** (exchange one concept's coordinates
for another inside an activation, touching nothing orthogonal to the pair)
and **steering** (add a scaled concept direction) — and, because Anthropic
shipped no intervention code, gated them with **pre-committed invariant
tests** instead of a reference diff: a rigged tiny model whose Jacobian is
known exactly, with numbers chosen so every post-patch logit is asserted *to
the bit*. Then both protocols ran on all three subjects, descriptive per the
triple null. Swapping the model's ready answer mid-forward barely moves its
spoken report (top-5 rates 10–18% against the paper's 88% anchor). But
*steering a thought in and asking the model to name it* produced the
project's first dose–response curve: at 1.5B the report rate climbs 0 → 30%
with strength while the zero-strength control stays exactly 0/101 — and the
same curve is flat at 0.5B and barely rises at 3B.

### What was learned

11. **When there is no oracle, invariants are the gate.** The AGREE pattern
    needs a reference to diff against; none exists for interventions. The
    replacement discipline: hand-build a model where the *true* answer is
    analytic (dyadic numbers → exact equality, no tolerances to hide a sign
    or transpose bug), and re-verify the core property (coordinates exchange)
    on every *real* application at runtime. The runtime check stayed silent
    across ~550 swaps — that silence is now evidence, not hope.
12. **An intervention can "work" and still miss the headline.** Steering
    moved the median concept's rank ~3× at 0.5B, ~300× at 1.5B, ~10× at 3B —
    directionally right on every subject. Only 1.5B converts movement into
    rank-1 reports. "Did the needle move" and "did it arrive" are different
    questions; report both or the summary lies.
13. **A control earns its keep in one number.** α = 0 giving exactly 0/101 on
    every subject kills the best alternative story ("the prompt begs for
    concept words at an open quote") for free. Every strength cell above it
    inherits that interpretability.
14. **Scale structure keeps refusing to be monotone.** M0's multihop peaked
    at 1.5B and regressed at 3B; the introspection curve does the same, with
    the 1.5B–3B gap CI-clean. At hobby scale, "bigger model" is not a
    monotone knob — and the interesting subject in this repo is repeatedly
    the middle one.

### Recall questions (answers in this repo's docs)

7. The 1.5B introspection curve is the most striking result in the project so
   far — yet it can never gate M1's verdict. Which frozen decision made that
   so, and what forking-paths discipline does it protect?
8. The α = 0 control came back exactly 0/101 on all three subjects. Which
   alternative explanation of the 1.5B curve does that single number
   eliminate?
9. M1's correctness gate is invariant tests instead of M0's AGREE diff. What
   made the AGREE pattern impossible here, and which class of bug do the
   invariants provably catch — and which class can they *not*?
