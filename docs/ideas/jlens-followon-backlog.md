# J-lens follow-on project backlog

_Captured 2026-07-27 from a brainstorm session run against dim-stage's closed v1
results. Purpose: preserve every candidate follow-on project — with feasibility
reasoning and sequencing notes — so no future session has to re-run the brainstorm._

**Status: idea #5 (Off-switch cartography) was PICKED on 2026-07-27** as the next
project, to be kicked off as its own repo. Everything else here is live backlog.

## Session parameters (the lens these were scored through)

- **Payoff: portfolio-first.** Reads-strong-to-a-hiring-manager beats novelty risk
  when they conflict; community value is upside, not the bar.
- **Scope: standard chain.** 2–4 weeks, dim-stage-style milestone chain with
  pre-registered gates.
- **Compute: 7B–14B rentals allowed** (~$10–30 of rented GPU), on top of the usual
  local Qwen2.5-0.5B/1.5B/3B on MPS.
- **Standing design constraint:** designs must not depend on readable full-vocab
  readouts at ≤3B (the M0 triple null). Favor probe-based, transport-based, and
  ablation-based designs, and designs built on dim-stage's CI-clean *positives*.

## The six primitives (referenced by number below)

READ — (1) ranked full-vocab readout; (2) single-vector probe ⟨v_t, h⟩ vs a
threshold; (3) sparse decomposition → discrete active-concept inventory.
WRITE — (4) steer h ← h + α·v_t; (5) ablate (negative α, project out v_t, or
remove top-k J-contents); (6) lens-coordinate swap (orthogonal complement untouched).

## The dim-stage results these build on

Details in [`../M0-BRIEF.md`](../M0-BRIEF.md) through
[`../S4-BRIEF.md`](../S4-BRIEF.md) and the [README](../../README.md). Short form:
M0 readability = pre-registered NULL ×3 scales. M1 = report doesn't follow swaps,
but a steered-in thought becomes reportable at 1.5B only (dose–response, exact-zero
control; S1: J-specific, saturating, mid-band L16–20). M2 = two-hop mostly fails;
at 3B works *only* through J-transport (J−I = +.116 CI-clean; raw rows 0/43).
M3 = "does not modulate" at the gate, but a real dose-ordered, scale-growing focus
signal. S2 = J-specific routing at α=1; α=2 overdose *extinguishes* routing.
S3 = selectivity — the only would-gate to HOLD ×3 (J-space ablation kills two-hop
chains; random damage and WikiText survive). S4/S4b = a late-band concept direction
is a hard output off-switch; concept-specific at 1.5B CI-clean (primed 0/22 vs
control 16/22), specificity emerging with scale.

---

## Read-only

### 1. Emergence Atlas — "where does the workspace turn on?"

- **Pitch:** Extend M0's readability battery (plus a probe/decomposition arm) to
  Qwen2.5-7B and 14B, turning the 3-scale null into a 5-scale emergence curve
  bracketing the paper's 27B.
- **Primitives:** (1), (2), (3).
- **Feasibility given the nulls:** High — the nulls ARE the left half of the curve.
  Risk is infra, not science: the 14B lens fit (d_model 5120) would be the priciest
  fit yet attempted; 7B is comfortable. Honest caveat: the modal outcome is "still
  null at 14B" — informative bracketing, but a third null-flavored headline.
- **Effort:** 2.5–3.5 weeks (fits are the long pole).
- **Demonstrates:** Directly answers the paper's named open question; the natural
  sequel a hiring manager can follow straight from dim-stage's README.

### 2. Detectable-before-readable — signal-detection theory on the workspace

- **Pitch:** Full-vocab readout failed, but is the information *present*? Measure
  single-vector probe detection (ROC/AUC per concept × layer × scale) against the
  logit-lens falsification arm — quantify the gap between "information present"
  and "information legible."
- **Primitives:** (2) as the instrument, (1) as the bar it beats or doesn't.
- **Feasibility:** Excellent — detection is a strictly lower bar than top-10
  readout, so the design is built *for* the null. Existing lenses, local, $0.
- **Effort:** 1.5–2.5 weeks.
- **Demonstrates:** Reframes the null as a measured detectability curve; shows
  signal-detection rigor (AUC + Wilson CIs) — a stats-literate artifact.

### 3. The readability predictor — why exactly did M0 fail?

- **Pitch:** M0's texture (surface-adjacent partial, abstract hard-zero, multihop
  non-monotone) suggests readout success is predictable from item features — token
  frequency, concreteness, tokenization, layer band. Fit a predictor and validate
  it on a pre-registered held-out item set.
- **Primitives:** (1), (2).
- **Feasibility:** Very high — mostly analysis over existing M0 data plus cheap new
  item batteries. Local, $0.
- **Effort:** 1.5–2 weeks (lightest item on the list).
- **Demonstrates:** Analytical maturity — turning "0/6" into "here is exactly what
  fails, and here's the model that predicts it."

### 4. Workspace faithfulness — does the stream predict the answer better than the stated reasoning?

- **Pitch:** Probe the mid-band for answer-relevant concepts during multi-step
  tasks and test whether workspace content predicts the final answer better than
  the model's verbalized reasoning — chain-of-thought faithfulness at hobby scale.
- **Primitives:** (2), (3), optional (5) as causal check.
- **Feasibility:** Medium — probes route around the readability null, but M2 showed
  small Qwen's multi-hop is itself shaky; 7B mitigates.
- **Effort:** 3–4 weeks.
- **Demonstrates:** The hottest safety topic on the list (faithfulness), which
  hiring managers recognize instantly.

## Write-only

### 5. Off-switch cartography — systematize the S4b discovery ★ PICKED 2026-07-27

- **Pitch:** S4b found a concept-specific, late-band, hard output off-switch
  (primed 0/22 vs control 16/22, CI-clean at 1.5B). Map it: a 50+ concept battery,
  layer-band localization, dose–response, a full specificity matrix (prime concept
  A, test outputs A…Z), and scale emergence including a cheap 7B point.
- **Primitives:** (4) steer, (5) ablate; all grading deterministic logit-based.
- **Feasibility:** Highest on the list — the phenomenon is already CI-clean at 1.5B
  with deterministic grading, zero dependence on readable readouts, zero new infra
  (7B fit optional).
- **Effort:** 2–3 weeks.
- **Demonstrates:** The strongest "I found something and characterized it properly"
  story available — it's a dim-stage-original effect, not the paper's, and the
  specificity matrix is a killer figure.
- **Why it won (portfolio-first logic):** the only idea that is simultaneously
  built on a CI-clean positive *we* discovered, fully deterministic in grading,
  zero-dependency on new infra, and a story no one else has. After two
  null-headlined projects, a positive-result project is the portfolio move.

### 6. Steering pharmacology — dose–response curves as a public methods artifact

- **Pitch:** dim-stage saw saturation (S1), overdose-extinguishes-routing (S2's α=2
  inversion), and small-model oversteer. Systematize it PK/PD-style: probe readout
  = "plasma concentration," behavioral effect = "clinical response," across
  concepts × layers × α × scale. Deliverable: a practical dosing guide + open dataset.
- **Primitives:** (4), (6) for writes; (2) to measure in-stream concentration vs
  behavioral effect.
- **Feasibility:** Very high — extends the two strongest positives; local, $0.
- **Effort:** 2–3 weeks.
- **Demonstrates:** A citable methods contribution anyone steering small models
  would actually use; extremely legible figures.

### 7. Surgical forgetting — J-space ablation as targeted unlearning-lite

- **Pitch:** S3's selectivity gate (the only one that HELD ×3) says J-space
  ablation kills targeted chains while sparing WikiText. Push it: delete specific
  facts/associations, measure kill-rate vs collateral damage (neighbor concepts,
  perplexity) with random-damage controls.
- **Primitives:** (5) as the scalpel, (3) to choose targets, (2) to audit collateral.
- **Feasibility:** High — built directly on the validated selectivity result;
  deterministic grading.
- **Effort:** 2–3 weeks.
- **Demonstrates:** Safety-adjacent (machine unlearning) causal surgery with
  proper controls.

## Read + write

### 8. Causal-but-illegible — the lens can write what it cannot read

- **Pitch:** M2-at-3B worked *only* through J-transport (raw rows 0/43;
  J−I = +.116 CI-clean) while readout stayed null. Make the dissociation the
  object: a tasks × scales grid measuring causal success (writes) vs legibility
  (reads), J=I falsification throughout.
- **Primitives:** (6), (4) for causality; (1), (2) for legibility.
- **Feasibility:** High — the anchor effect already exists CI-clean at 3B; 7B
  extends the grid.
- **Effort:** 2.5–3.5 weeks.
- **Demonstrates:** The conceptually sharpest claim on the list — "J-space is a
  causal medium before it's a legible one" — the most interp-community-facing idea.

### 9. Concept thermostat — closed-loop steering with probe feedback

- **Pitch:** A controller that holds a concept's probe reading at a setpoint
  during generation, adjusting α per token. Demo: dial "ocean" to 0.3 / 0.6 / 0.9
  and watch the text shift — a GIF in the README.
- **Primitives:** (2) sensor + (4) actuator in a feedback loop.
- **Feasibility:** Medium-high — fully null-proof (single-vector probe only); the
  risk is controller stability given small-model oversteer; the narrow-window
  heuristic is the mitigation.
- **Effort:** 3–4 weeks (most engineering-heavy).
- **Demonstrates:** Interp × systems engineering — the single most demo-able
  artifact for a hiring manager.

### 10. The introspection window — is 1.5B-only real?

- **Pitch:** dim-stage's weirdest result: steered-in thoughts become reportable at
  1.5B (0→30/101, control exactly 0) but not 0.5B or 3B. Run the identical
  protocol at 7B and 14B: is it a scale *window*, noise, or the early edge of
  re-emergence?
- **Primitives:** (4) steer, (1)-style deterministic top-5 report grading.
- **Feasibility:** High — protocol fully built and validated (M1/S1); needs the
  new lens fits (same infra risk as #1).
- **Effort:** 2–2.5 weeks.
- **Demonstrates:** Resolving our own anomaly with a clean non-monotonicity curve —
  good science instincts on display.

### 11. Injection signatures — can probes see a prompt injection coming?

- **Pitch:** Do injected instructions leave a probe-detectable workspace signature
  before the model complies — and does ablating that direction block compliance?
- **Primitives:** (2), (3) to detect; (5) as the causal blocker.
- **Feasibility:** Medium — probe-based so null-compatible, but small-model
  instruction-following is weak and the signature is unproven; highest scientific
  risk in the read+write group.
- **Effort:** 3–4 weeks.
- **Demonstrates:** Security-flavored interp on a live industry problem — big
  hiring-manager resonance if it works.

### 12. Thought transplant — cross-model concept transfer

- **Pitch:** Read a concept's sparse J-space coordinates in 1.5B, write the
  matched lens vectors into 0.5B/3B, measure behavioral transfer. Do lens spaces
  align across scales at all?
- **Primitives:** (3) read, (6)/(4) write.
- **Feasibility:** Speculative — no prior evidence of cross-scale alignment; cheap
  to test, high null risk (though pre-registered nulls are this lineage's brand).
- **Effort:** 2–3 weeks.
- **Demonstrates:** The most memorable creative swing; highest variance on the list.

---

## Combination & sequencing notes

- **#1 + #10 merge naturally.** If the Emergence Atlas ever runs, fold the
  introspection-window protocol in as a second tracked property — the expensive
  part (7B/14B lens fits) is shared, and the atlas then covers a read property
  (readability) and a write property (injected-thought reportability) on the same
  curve. #10 should probably never run standalone if #1 is on the table.
- **#5 + #6 form a two-project arc.** Off-switch cartography builds the steering
  infrastructure (concept batteries, α sweeps, band windows, deterministic
  emission grading) that steering pharmacology needs; running #6 right after #5
  is mostly new measurement on existing rails. Natural "next after the pick."
- **#2 + #3 pair as cheap read-only analysis projects** — both are $0, local,
  and mostly analysis over existing data. Either works as a between-chains
  palate-cleanser or a 1–2-week gap-filler; together they'd make one solid
  "anatomy of the null" project.
- **#7 extends #5's toolkit** — ablation-as-scalpel shares the concept-battery
  and collateral-audit machinery the off-switch specificity matrix builds.
- **#8 is the community-facing pick** whenever the goal flips from portfolio to
  interp-community contribution — it sharpens dim-stage's most interesting
  cross-cutting finding into a single claim.
- **Scale-sweep infra is a one-time cost.** The first project to fit 7B (and
  especially 14B) lenses pays the infra tax for every later idea that wants those
  scales (#1, #4, #8, #10, and the #5 stretch all touch 7B+).

## Ranking at time of capture (portfolio-first lens)

1. **#5 Off-switch cartography** — picked; see "why it won" above.
2. **#1 Emergence Atlas** — strongest sequel narrative, best use of the rental
   budget, but modal outcome is another null headline.
3. **#6 Steering pharmacology** — strongest community-useful artifact with
   near-guaranteed results; cheapest strong option.

Re-rank when parameters change: under a community-first lens #8 rises to the top;
under a $0/local-only constraint #2, #3, and #6 lead.
