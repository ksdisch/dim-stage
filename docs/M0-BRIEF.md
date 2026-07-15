# M0 start-of-stage brief — the fit pilot

*2026-07-15 · status: options presented, decisions pending Kyle*
*Sources: the paper (refetched 2026-07-15, `refs/workspace-paper.md`, not committed) and the
reference repo (`anthropics/jacobian-lens`, cloned to `refs/jacobian-lens/`, not committed).*

## What M0 is, in plain terms

M0 answers one question: **can we fit the paper's lens on our laptop, trust our own
implementation of it, and does it read anything at small scale?** Everything after M0
(the three property measurements) assumes a yes.

The object we're building is the **Jacobian lens** (J-lens). Some vocabulary, defined once:

- The **residual stream** is the running vector that flows through a transformer — each
  layer reads it, computes something, and adds its result back in. Think of it as the
  model's working scratchpad, one vector per token position.
- The **unembedding** is the model's final matrix: it turns the last-layer residual vector
  into one score per vocabulary token. Those scores are called **logits** — whichever
  token scores highest is the model's next-token prediction.
- The **logit lens** is the old trick of applying the unembedding to a *middle* layer's
  residual vector, as if it were the last layer. It assumes the middle layers use the
  same coordinate system as the final layer — which they mostly don't, so it produces
  garbage in early layers.
- A **Jacobian** is a matrix of sensitivities: entry (i, j) says how much coordinate i of
  a later vector moves when you nudge coordinate j of an earlier vector. The J-lens fixes
  the logit lens by first multiplying the middle-layer vector by `J_l` — the *average*
  Jacobian of the final layer with respect to layer `l` — which transports it into the
  final layer's coordinate system before unembedding it:

  ```
  lens_l(h) = unembed( J_l @ h ),    J_l = E[ ∂h_final / ∂h_l ]
  ```

- The paper's claim is that what this lens reads out in the **workspace band** — a
  contiguous range of middle layers — behaves like a global workspace: reportable,
  swappable, steerable content. Our project measures whether that band exists and is
  readable in Qwen 0.5B and 1.5B.

M0's deliverables: (1) the hour-one hardware gate, (2) this brief's decisions frozen,
(3) an independent implementation of the fitting procedure, (4) the AGREE cross-check
against the reference implementation, (5) the readability gate on both subjects,
(6) the single-token stimulus pre-filter.

## Design extraction (verbatim, source-cited)

### The estimator — how `J_l` is actually computed

From `refs/jacobian-lens/jlens/fitting.py` module docstring (this is the spec my
independent build must hit):

> Estimator: for each output dimension, inject a one-hot cotangent at *every valid
> target position at once* and backprop. The gradient at source position `p` is then
> `sum_{p' >= p} dh_final[p'] / dh_l[p]`, the sum over later target positions; we take
> the mean over source positions `p`. **This is the reduction used in the paper.**
> A per-position estimator (`dh_final[p] / dh_l[p]` averaged over `p`) gives a slightly
> different `J_l`; both work as a lens.

Plain terms: a **cotangent** is the "direction of interest" you hand to backpropagation —
you set a 1 in one output coordinate and backprop to ask "what input changes move *this*
coordinate?" (the vector–Jacobian product, **VJP**). Doing that once per output dimension
builds `J_l` one row at a time. The reference batches this: it replicates the prompt
`dim_batch` times so each backward pass computes `dim_batch` rows at once.

From `refs/jacobian-lens/README.md`:

> The expectation is over prompts, source positions, and all current-and-future target
> positions in a generic web-text corpus.

Per-prompt `J_l`s are accumulated as a running mean over prompts (`fit()` in
`fitting.py`); the lens for layer `l` is that mean.

### Frozen parameters (no decision needed — copied from the reference)

| Parameter | Value | Source | Note |
|---|---|---|---|
| `skip_first` | 16 | `fitting.py:42` | Positions 0–15 excluded: "early positions act as attention sinks and have atypical residual statistics" |
| Final position | excluded | `valid_position_mask` | "the final position has no next-token target" |
| `max_seq_len` | 128 | `fitting.py` default; README "1000 sequences of 128 tokens" | Prompts truncated to 128 tokens |
| `target_layer` | final layer | `fit()` default | Docstring notes penultimate "can give a better-conditioned J_l" — we keep the default; any change would be a deviation row |
| `source_layers` | every layer below target | `fit()` default | Fit the lens at all layers; the band question is about *reading*, not fitting |
| `J_l` accumulation | fp32, running mean per prompt | `fitting.py` | |
| `dim_batch` | 8 default; **math-neutral knob** | `fitting.py` docstring | Changes batching only, not the estimator ("total backward FLOPs are unchanged") — free to tune for MPS wall-clock |

Cost model (from the docstring): one forward pass + `ceil(d_model / dim_batch)` backward
passes per prompt. At `dim_batch=8`: **112 backwards/prompt** on 0.5B (d_model 896),
**192 backwards/prompt** on 1.5B (d_model 1536).

### The paper's band-finding analysis

Load-bearing discovery — from the paper's Methods:

> Throughout the paper, we report results on 25 evenly spaced layers of the model's
> residual stream **reindexed to the range [0–100] so that layer numbers can be
> interpreted as percentages.**

So the paper's workspace band "~L38 to ~L92" means **38%–92% of network depth**, which
transfers to any depth. The band was identified by converging diagnostics:

1. **CKA block structure** — centered kernel alignment, a score for whether two layers'
   lens-vector geometries look alike, computed for every layer pair. The matrix shows an
   early block (~first third), a long middle block, and a small late block.
2. **Next-token top-k accuracy** of the lens (panel a): near zero early, ticks up at
   workspace start, jumps steeply in final layers → the jump marks the band's *end*
   (readouts become "motor" content aligned with imminent output).
3. **Excess kurtosis** of the readout distribution (panel b) — how sharply peaked the
   logit distribution is versus random; a peaked readout means the lens found something
   specific. Its early rise marks the band's *start*.
4. **Top-1 autocorrelation** across positions vs a shuffled null (panel c) — abstract
   content persists across tokens; token-local content doesn't.
5. **Effective dimensionality** of the lens vectors (panel d).

> The analyses above all identify a similar range of layers, beginning about a third of
> the way through (~L38) and ending shortly before the output (~L92).

None of this band-finding machinery ships in the reference library — the library fits and
applies the lens at every layer; the band is a paper-side analysis choice. That's why
band selection is a decision we must freeze (D2).

### Grading conventions for the lens evals (readability gate inputs)

From `refs/jacobian-lens/data/evaluations/README.md`, verbatim conventions:

> **Hit** — a target token is a *hit* if it appears at lens rank 1 at any (layer,
> position) in the band over the scored span.
>
> Metric: **pass@k** = mean over items of the fraction of `intermediates` whose
> min-over-layers lens rank ≤ k.

Each of the six `lens-eval-*` distributions specifies a **single readout position**
(e.g. the token immediately preceding `target`, or the final prompt token) and takes the
rank minimum **over all layers** at that position. `lens-eval-order-ops` expands each
intermediate to a synonym set (rank = min over single-token synonyms).

### Fit-corpus guidance

From `refs/jacobian-lens/README.md`:

> The paper's lenses use 1000 sequences of 128 tokens from a pretraining-like corpus.
> Quality saturates quickly (§9.3); ~100 prompts is usable.

From the paper §9.3:

> We observe that J-lens beats the logit lens and tuned lens baselines with as few as
> 10 prompts, with modest improvements coming from additional data.

And the reference repo ships its own corpus convention — `jlens/examples.py::
load_wikitext_prompts`: *the first N records of WikiText-103 with ≥600 characters,
streamed in order from HuggingFace*. Fully deterministic — no seed needed.

## Decisions to freeze (Kyle picks each)

### D1 — AGREE-gate metric: how do we decide my build matches the reference?

Both implementations fit on the **same model and the identical prompt list**, so they
compute the same mathematical object; differences beyond floating-point noise mean my
build diverged (KICKOFF risk #4). Three levels of comparison, strictest first:

- **A. Matrix-level (recommended as the deciding gate).** Per fitted layer, relative
  Frobenius distance `‖J_mine − J_ref‖_F / ‖J_ref‖_F` (Frobenius norm = the size of a
  matrix: square root of the sum of squared entries). Gate: max over layers ≤ tolerance,
  where tolerance = **10× the measured run-to-run noise floor of the reference itself**
  (refit the reference twice on the same prompts; their self-distance is the floor — a
  calibrated tolerance, not a guessed one). *Merit:* the most sensitive detector of
  estimator divergence, and it localizes a failure to a layer. *Trade-off:* needs the
  calibration run (cheap: it's a small prompt subset).
- **B. Readout-level.** Top-1 token agreement between the two lenses at sampled
  (layer, position) cells on held-out prompts; Wilson 95% lower bound ≥ 0.95.
  *Merit:* tests what the lens actually says; immune to benign float noise.
  *Trade-off:* coarse — a small systematic divergence could still pass.
- **C. Endpoint-level.** pass@k on the six lens evals within CI of each other (Newcombe
  difference CI contains 0). *Merit:* like-for-like with the paper's own summary.
  *Trade-off:* weakest — two genuinely different lenses can tie in aggregate.

**Recommendation: A decides the gate, B runs as confirmation, C is reported but never
gates.** AGREE = A passes AND B passes.

### D2 — Band selection at small scale

- **A. Proportional transplant.** Fix the band to the paper's percentages: include layer
  `l` iff `0.38 ≤ l/(n_layers−1) ≤ 0.92`. Qwen2.5-0.5B (24 layers, indices 0–23):
  **L9–L21**. Qwen2.5-1.5B (28 layers, indices 0–27): **L11–L24**. *Merit:* zero degrees
  of freedom — no risk of tuning the band until results look good (pre-commitment is a
  forking-paths guard). *Trade-off:* if the band genuinely sits elsewhere at small scale,
  a misplaced band degrades readability and could produce a false null.
- **B. Re-derive the band with the paper's diagnostics.** Compute the per-layer
  diagnostics on our models and place the band where they indicate. *Merit:* most
  faithful; "does band structure even exist at small scale" is itself a finding.
  *Trade-off:* the paper's criteria are qualitative ("rises sharply") — reading a band
  off our own curves before gates are locked opens the forking-paths door; more build
  work before the first result.
- **C. Hybrid (recommended).** Pre-register A's proportional band as the primary analysis
  band. Compute the two cheap diagnostics (excess kurtosis + next-token top-1 accuracy
  per layer) as a *descriptive* arm: report whether small-scale curves agree with the
  transplanted band. If they clearly indicate a different contiguous band, changing bands
  is an owned deviation row with the curves attached — never a silent move.

### D3 — Fit corpus

- **A. WikiText-103, reference-native convention (recommended).** Use exactly
  `load_wikitext_prompts(N)`: first N records ≥600 chars, streamed in order.
  *Merit:* deterministic with no seed machinery; both implementations fit on the
  byte-identical prompt list (which D1 requires anyway); it is the reference repo's own
  example corpus, so the choice is citable. *Trade-off:* Wikipedia prose, not the paper's
  "generic web-text / pretraining-like corpus" — an owned deviation row.
- **B. FineWeb or C4 sample.** *Merit:* closer to "pretraining-like web text."
  *Trade-off:* needs a pinned snapshot + seeded sampling for reproducibility; more moving
  parts for a corpus the paper says barely matters past ~10–100 prompts.

**Corpus size (sub-choice):** start **N=100** (KICKOFF's plan; §9.3 says quality
saturates). Escalate toward the paper's 1000 only if the hour-one gate's measured
wall-clock makes a full fit an overnight job or less. 100 vs 1000 stays a deviation row
either way.

### D4 — Readability-gate condition (the M0 verdict gate)

Per-distribution metric, following the evaluations README verbatim: each intermediate is
one Bernoulli trial — hit iff its min-over-layers rank at the specified readout position
is ≤ k. Hit counts give Wilson CIs; J-lens vs logit-lens differences get Newcombe CIs.
The logit lens here is literally our own lens code with `J = I` — the standing
falsification arm.

Two separate claims need separating (the paper itself notes the logit lens "captures much
of the workspace content" in later layers — so "J-lens beats logit lens" and "a readable
workspace exists" are *different questions*):

- **A. Two-arm gate (recommended).**
  - *Arm 1 — absolute readability:* J-lens pass@10 Wilson 95% lower bound ≥ **0.5** on
    ≥ 3 of 6 distributions → the subject READS; below that on all arms → NULL. (0.5 =
    "recovers the known intermediate at least half the time"; chance under a ~150k-token
    vocabulary is ≈ 0.007%, so the floor is generous but pre-declared, not post-hoc.)
  - *Arm 2 — J-advantage:* Newcombe 95% CI on (J-lens − logit-lens) hit-rate difference,
    per distribution; the J-correction "adds value" where the CI excludes 0.
  - Verdicts are reported per arm — a subject can READ without J-advantage (workspace
    visible even to the logit lens) and that is a finding, not a failure.
- **B. Difference-only gate.** READABLE iff J-lens beats logit lens. *Trade-off:* wrongly
  calls NULL when the workspace is real but logit-lens-visible — conflates the two claims.
- **C. AUC replication.** The paper's normalized pass@k AUC (Figure 52). *Trade-off:* an
  AUC is not a binomial count, so no Wilson/Newcombe CIs without bootstrap machinery —
  violates the project's stats ruler. Report it descriptively for like-for-like with
  Figure 52; don't gate on it.

Also reported descriptively (not gating): band-restricted min-rank (the workspace-specific
reading), and k ∈ {1, 5, 25} alongside the gate's k=10.

### Assumption on record — model variants

Subjects are **Qwen2.5-0.5B-Instruct** and **Qwen2.5-1.5B-Instruct**. Reason: the M1–M3
stimuli are chat-format instruction prompts (`[{"role": ...}]`, "Think of a category.
Answer in one word.") — base models don't reliably follow them, and the lens must be fit
on the very model being read. Costs nothing to change before coding starts; flag if you
want base models instead.

## Hour-one gate spec (runs immediately after decisions freeze)

**Question:** does the backward pass run on MPS (Apple's GPU backend for PyTorch), and at
what wall-clock rate?

- **Procedure:** pin the reference repo as a dev-dependency (cross-check group, exact
  commit SHA), run `jlens.fitting.jacobian_for_prompt` on Qwen2.5-0.5B-Instruct on MPS
  for 2 WikiText prompts (all source layers, `dim_batch=8`, `max_seq_len=128`), model in
  fp32. Record per-prompt wall-clock; extrapolate to a full N=100 fit.
- **PASS:** completes without error and the extrapolated N=100 fit on **both** subjects
  totals ≤ 12 h (overnight-able). Also try a larger `dim_batch` (math-neutral) to see if
  MPS per-pass overhead dominates.
- **FAIL:** MPS errors out (e.g. an op unsupported in the backward graph) or the
  extrapolation blows the budget → stop, surface to Kyle; the ~$1 rented-GPU fallback is
  an owned deviation row (amended bar entry 2), never a silent move.

### Result (2026-07-15) — PASS

Measured by `m0_hour_one_gate.py` on Qwen2.5-0.5B-Instruct, fp32, MPS (torch 2.13.0,
transformers 5.13.1, reference jlens @ `581d398`), first two WikiText prompts per the
D3 convention (both truncate to `seq_len=128`, `n_valid=111`):

| Config | prompt 0 | prompt 1 (warm) |
|---|---|---|
| `dim_batch=8` (112 backwards) | 44.1 s | 42.7 s |
| `dim_batch=32` (28 backwards) | 1192.0 s | 931.0 s |

- Extrapolated N=100 fit: 0.5B **1.19 h** (measured rate) + 1.5B **6.36 h** (FLOPs-ratio
  extrapolation, 229 s/prompt) = **7.55 h ≤ 12 h → PASS.** MPS handles the full backward
  graph without error; no rented-GPU fallback needed.
- Surprise worth keeping: raising `dim_batch` is math-neutral but ~25× *slower* here —
  the 32×-replicated retained graph appears to blow MPS unified-memory locality. The
  knob stays frozen at **`dim_batch=8`** for all fits on this hardware.

## AGREE gate result (2026-07-15) — deliverables 3 + 4

Run by `m0_agree_gate.py` on Qwen2.5-0.5B-Instruct (fp32, MPS; torch 2.13.0,
transformers 5.13.1, reference jlens @ `581d398`, independent fitter `fitter.py`).
Per D1: all three fits (independent, reference A, reference B) on the byte-identical
first-16 WikiText prompts — loader byte-identity asserted over all 21 prompts used
(D3 sanity, PASS). Readout confirmation on the 5 held-out prompts (WikiText records
17–21), all 23 fitted layers, every 4th valid position (28 positions × 23 layers ×
5 prompts = 3220 cells). Fit rate: independent 686 s / 16 prompts, reference
686 s / 16 prompts — identical (42.9 s/prompt).

| Check | Measured | Gate condition | Verdict |
|---|---|---|---|
| Reference run-to-run noise floor (max per-layer rel-Frobenius, refit vs refit) | **exactly 0** — MPS backward is fully deterministic; per-prompt diagnostics of the two refits match line-for-line | calibration input; pre-declared stand-in 1e-4 applies | — |
| Matrix gate (max per-layer rel-Frobenius, independent vs reference) | **0.000e+00** — bitwise-identical `J_l` at every layer | ≤ 10× floor = 1e-3 | **PASS** |
| Readout confirmation (top-1 token agreement, held-out cells) | 3220/3220 | Wilson 95% LB 0.9988 ≥ 0.95 | **PASS** |

**VERDICT: AGREE.** The independent build computes the same mathematical object as
the reference down to the last bit — stronger than the gate required (any nonzero
distance below 1e-3 would also have passed). Two incidental observations worth
keeping: (1) MPS fp32 backward is bitwise reproducible run-to-run, so the 1e-4
stand-in floor did real work exactly as pre-declared; (2) the reference's own
convergence diagnostic (`max_d_mean`, the relative shift in the running mean per
added prompt) was still ≈ 7% at prompt 16 — supporting D3's choice to fit the
production lens at N=100 rather than stopping at 16.

## Deviations table (starter — grows as M0 runs)

| # | Paper / reference | Ours | Why | Status |
|---|---|---|---|---|
| 1 | Subjects: Claude Sonnet/Haiku/Opus 4.5 (+ Qwen3.6-27B demo) | Qwen2.5 0.5B + 1.5B Instruct | The whole thesis: is the workspace readable at laptop scale? | Owned, by design |
| 2 | Fit corpus: 1000 × 128-token seqs, pretraining-like web text | WikiText-103 via reference loader, N=100 (pending D3) | Wall-clock; §9.3 saturation evidence | Pending D3 |
| 3 | Hardware: datacenter accelerators | MPS, fp32 | $0/trial constraint | Owned |
| 4 | Band: derived from CKA + 4 diagnostics per model | Proportional transplant 38–92% + descriptive diagnostics (pending D2) | Forking-paths guard at small N | Pending D2 |
| 5 | Stimuli: used as shipped | Single-token pre-filter under Qwen tokenizer | Rank-based grading needs single-token targets | Owned, mechanical |

## What M0 does NOT decide

- M1–M3 protocol details (each gets its own start-of-stage brief).
- The swap/steering intervention code (M1+; M0 is read-only).
- 3B escalation (only if BOTH subjects null — KICKOFF decision on record).
- Fitted-lens artifact policy (`*.pt` stays gitignored until we see sizes).

## Frozen decisions

*Picked by Kyle, 2026-07-15. These are settled — relitigating any of them is a deviation
row, not a conversation.*

| # | Decision | Choice | Reason |
|---|---|---|---|
| D1 | AGREE-gate metric | Matrix-level Frobenius gate (tolerance = 10× reference run-to-run noise floor) decides; top-1 readout agreement (Wilson LB ≥ 0.95) confirms; lens-eval pass@k reported, never gates | Most sensitive detector of estimator divergence; localizes failure to a layer |
| D2 | Band selection | Hybrid: proportional 38–92% band pre-registered as primary (0.5B: L9–L21; 1.5B: L11–L24); kurtosis + next-token-accuracy curves as descriptive arm | Zero forking-paths freedom on the gate; still learns whether band structure exists at small scale |
| D3 | Fit corpus | WikiText-103 via the reference's `load_wikitext_prompts` convention, N=100; escalate toward 1000 only if hour-one wall-clock allows overnight | Deterministic, byte-identical prompts for both implementations; owned deviation from "generic web text" |
| D4 | Readability gate | Two-arm: (1) READS iff J-lens pass@10 Wilson 95% LB ≥ 0.5 on ≥3/6 distributions, else NULL; (2) J-advantage per distribution via Newcombe CI on the difference vs `J=I` | Separates "workspace exists" from "Jacobian correction adds value" |
| — | Model variants (assumption, unobjected) | Qwen2.5-0.5B-Instruct + Qwen2.5-1.5B-Instruct | M1–M3 stimuli are chat-format; the lens must be fit on the model being read |
