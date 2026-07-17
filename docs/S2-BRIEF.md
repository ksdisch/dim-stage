# S2 start-of-stage brief — flexible generalization (stretch)

*2026-07-16 · stage-opening PR (docs only, per-stage rhythm). Kyle picked
generalization over selectivity/wrap as the second stretch stage. Decisions
D19–D22 below are open until Kyle freezes them; the freeze is recorded in
`DECISIONS.md` and lands with the build PR alongside the runner and results.*

## What S2 is, in plain terms

The workspace story says a thought written to the workspace is **broadcast** —
readable by *many* downstream circuits, not just the one that produced it. The
paper's test: take one concept (say *France*) and sixteen different one-line
questions ("The capital of {arg} is the city of", "Most people in {arg}
speak", …). Swap the workspace representation of France for China — the same
swap in every prompt — and ask whether *each* question now answers as if the
prompt said China: Beijing, Chinese, Asia. If one identical edit redirects
sixteen different computations, the swapped vector is functioning as a shared
argument that many circuits consume — the "flexible generalization" property.

Jargon, once each: the **swap** is M1's lens-coordinate exchange (read how much
of the source and target directions the residual stream contains, exchange the
two amounts, leave everything else untouched — `intervention.py`, unchanged
since M1). **Clamped** means the edit is applied at *every token position*, not
just where the argument sits. **α** (alpha) is swap strength: α = 1 is the
exact exchange; α = 2 doubles the amount of source removed and target added.

This is a **reading-and-writing** stage with no new model, no new fit, no new
band, and (almost) no new operator — the one extension is an `alpha` knob on
the existing swap, pinned by algebra below.

## The findings we build on (M2 + S1, verbatim)

- **M2 (two-hop swap):** the same swap operator, aimed at a *latent*
  intermediate, redirected the final answer rarely (.286/.073/.116 on
  0.5B/1.5B/3B vs the paper's .60 Sonnet anchor) — but at 3B it worked **only
  through the Jacobian transport** (identity rows flipped 0; J − I Newcombe
  +.116 [+.011, +.245]).
- **S1 (introspection follow-up):** the 1.5B injected-thought dose–response is
  CI-cleanly a **J-transport** effect, saturates like the paper's Figure 7, and
  localizes to the middle band third.

S2 asks the natural next question: when a swap *does* take, is the swapped
vector consumed as a general-purpose argument by **many different functions**
(4 categories × 4 functions), and does the paper's own predictor — how strongly
the concept was loaded in the workspace to begin with — explain where it fails?
Framing stays **descriptive** throughout (triple readability NULL; the
pre-registered re-scope holds).

## Design extraction (verbatim, source-cited)

**Sources:** paper §"The J-space supports flexible generalization" (Figures 18,
19, 68; transformer-circuits.pub/2026/workspace — no arXiv ID); reference repo
`data/experiments/flexible-generalization.json` + its `README.md` entry
("flexible-generalization"). The reference repo ships the **complete item
set**, so extraction is verbatim — no reconstruction needed. Per the lossy-wall
rule the runner reads the *data file* from `refs/` at runtime (M2's
`EXPERIMENT_PATH` precedent); reference *code* stays unimported.

**Item set (shipped):** 4 categories — countries (France/Canada/China/Egypt),
months (February/April/July/October), animals (lion/eagle/shark/spider),
numbers (three/five/seven/nine) — × 4 function templates each = **16
templates**, each with a per-argument answer map. Within a category every
ordered source→target pair is a trial: 12 pairs × 16 templates = **192
trials**. The diagonal (source = target) is the unmodified **baseline**: the
greedy next token must match the source's own answer (64 cells).

**Protocol (paper + repo README, quoted conventions):**

- Swap the J-lens vector of the source argument for the target's, "at every
  token position across a band of intermediate layers, applying the identical
  swap regardless of which prompt we are in" — i.e. clamped at every prompt
  position, every band layer. That is exactly `swap_edits(..., positions=None)`.
- Grade the model's next-token output distribution at the last prompt
  position: **success = the target-appropriate answer is top-1**; when it is
  not, the appendix records its rank (Figure 68's grey numbers). Deterministic
  logit-rank oracle — our standing convention unchanged.
- **α = 2** is the "double strength" swap, "doubling the strength with which we
  subtract the source lens vector and add in the target."
- Prompts are raw completions (no chat template), greedy readout — the M2
  convention.

**α, pinned by algebra.** Our swap adds the correction
`Δ = (c_t − c_s)(v_s − v_t)` (with `c` the lens coordinates: in a source
prompt `c_s ≫ c_t`, so Δ *subtracts* source direction and *adds* target — the
paper's wording exactly). The swap is an involution (applying it twice restores
the original), so "double strength" cannot mean "apply twice"; the only
consistent reading is scaling the correction: **`h ← h + α·Δ`**. α = 1 is the
current operator bit-for-bit; α = 0 is an exact no-op. New pre-committed
invariants in `test_intervention.py` cover both plus s = t at every α.

**Anchors (Sonnet-scale, from the paper):**

| Cell | Anchor |
|---|---|
| Pooled swap success, α = 1 | **76/192** (.396) |
| Pooled swap success, α = 2 | **101/192** (.526) |
| Countries, α = 1 | 42/48 off-diagonal at rank 1 |
| Numbers, α = 1 | 0/48 ("never succeed") |
| Category order | countries ≫ months > animals > numbers |
| Failure structure | failures keep the *original* answer top-1, target rises in rank; α = 2 recovers many |
| Predictor (Fig. 19 right) | **workspace loading** — cosine similarity between the residual stream and the concept's lens vector, "averaged over the argument and readout positions in the unmodified forward pass" — predicts swap success; countries load highest, numbers lowest |

**Single-token survival (measured 2026-07-16, tokenizer shared by all three
subjects):** all 16 arguments have single-token forms; **60/64 answers** do.
The four failures are all in animals — savanna, arachnid, convocation, shiver —
killing the 3 swaps *into* each of those cells: **180/192 trials gradable**
(countries 48, months 48, animals 36, numbers 48), 60/64 baselines. One owned
deviation row; conveniently confined to the category the paper itself reports
as rarely swapping.

## Decisions to freeze (Kyle picks each; recommendations flagged)

### D19 — Item set & filter handling

- **A (recommended).** Reference set verbatim + the standing M0 single-token
  pre-filter: run the 180 gradable trials, count and skip the 12 filtered ones,
  report pooled-180 beside the 192-trial anchor. *Merit:* verbatim extraction,
  zero invented items, the filter is the same one every prior milestone owned.
  *Trade-off:* the pooled cell is 180 vs the anchor's 192 (a ~6% denominator
  dent, all in animals).
- **B.** Substitute single-token answers for the four failures (e.g. re-keying
  habitat/lion to "plains"). *Merit:* restores 192. *Trade-off:* invents items
  the paper never ran — breaks verbatim extraction for a marginal category;
  not recommended.
- **C.** Drop the animals category entirely (uniform 3-category, 144-trial
  design). *Merit:* no partially-filtered category. *Trade-off:* throws away 36
  gradable trials and the paper's animals row for tidiness; not recommended.

### D20 — Arms

- **A (recommended).** Two arms: **J-lens** swap and the standing **J = I**
  falsification arm (same coordinate-exchange, directions built from raw
  unembedding rows), every trial, every α; Newcombe 95% CI on (J − I) per α.
  *Merit:* the project's signature control, like-for-like with M2's Arm 2 —
  and M2 says raw rows flip *nothing* at 3B, so if S2's swaps work only through
  the transport, this arm is what shows it. *Trade-off:* doubles wall-clock
  (still minutes).
- **B.** J-lens only (the paper's published experiment). *Merit:* minimal.
  *Trade-off:* leaves the one falsification the whole project is built around
  off the table; not recommended.

### D21 — α grid

- **A (recommended).** **α ∈ {1, 2, 4, 8}** — the two anchored points plus two
  doublings of headroom, with the S1-D17-style **degeneracy guard** adapted to
  greedy readouts: flag any α × arm cell where one single token is top-1 on
  more than half of all trials across the 16 templates (a swap that shreds the
  stream collapses everything to one output). Beyond-paper cells (4, 8) are
  flagged as our own convention, same footing as S1's grid extension.
  *Merit:* the α = 1 → 2 direction is the paper's own dose check; 4 and 8 tell
  us whether small-scale swaps are merely *underdosed* (the paper's failure
  reading) or qualitatively absent. *Trade-off:* two owned grid points.
- **B.** Anchored {1, 2} only. *Merit:* pure like-for-like. *Trade-off:* if
  α = 2 still under-delivers, we can't tell "not enough dose" from "no effect" —
  the question S1's grid answered for steering.

### D22 — Readout frame

- **A (recommended).** Three readouts, all from the same runs: **(i)
  unconditional** pooled + per-category success (the anchor frame);
  **(ii) baseline-conditioned** — success among trials where the model got
  *both* the source prompt and the target's own diagonal right unswapped (it
  demonstrably knows f(source) and f(target); the honest small-scale frame,
  M2's lesson); **(iii) workspace loading** per argument (band-layer mean
  cosine at argument + readout positions, unmodified pass — free from the
  baseline forwards), reported against per-argument swap success as the
  paper's Fig-19-right shape check. *Merit:* anchor comparability, small-scale
  honesty, and the paper's own predictor, one run. *Trade-off:* three numbers
  to narrate instead of one.
- **B.** Unconditional only. *Merit:* one clean anchor cell. *Trade-off:* at
  our scale many baselines will fail; unconditional alone conflates "can't
  route the vector" with "never knew the fact" — the confound M2 taught us to
  condition away.

### Assumptions on record (cost nothing to change before coding)

- **Subjects:** all three (0.5B / 1.5B / 3B), forward-only, local MPS, serial.
  Bands and lenses are the frozen M0 artifacts (0.5B L9–L21, 1.5B L11–L24,
  3B L14–L32); no refit.
- **Swap rows take the prompt-position token form:** the argument appears *in*
  these prompts (unlike M2's latent intermediates), so the source row is the
  argument's token id as actually tokenized in each prompt (leading-space in
  most templates, bare where sentence-initial), and the target row is the
  matching variant (fall back to its other single-token form if the variant is
  missing; the runner counts fallbacks, expected 0). A deliberate, owned
  departure from M2's bare-form default, for mechanism-faithfulness.
- **Grading** uses `word_rank` over both single-token forms (min), M1/M2
  convention unchanged. Readout is `output_logits` at the last prompt position.
- **Runtime read-back** (M2's checked-swap pattern) on every swap; wrong-arm
  input → `VERDICT: INVALID` exit 2; `--dry-run` validates then stops;
  `--limit N` is smoke, never a result.
- **Per-template cells (N = 12) are pre-declared UNDERPOWERED** texture —
  reported (they're Figure 19-left's shape) but carrying no verdict weight.
  Verdict-bearing cells are pooled (180) and per-category (36–48).

## Pre-committed readouts (would-gate wording — descriptive mode)

- Baseline cell per subject: k/60 correct diagonals, Wilson 95%.
- Primary: pooled swap success k/180 per α per arm, Wilson 95%, beside the
  76/192 (α=1) and 101/192 (α=2) anchors; per-category rows.
- J − I Newcombe 95% at each α (D20-A).
- Shape checks, stated in advance: the paper's category order
  (countries ≫ … ≫ numbers), the α=1→2 improvement direction, and
  loading-tracks-success (D22-A-iii). Each either holds or fails openly.
- Degeneracy guard (D21-A) + INVALID conditions dry-run before any real run,
  wrong-arm case included.
- No property claim under the standing re-scope: the verdict line reads
  "routes / does not route (descriptive)" with the CI table, exactly as M2/M3.

## Deviations table (starter — grows as S2 runs)

| # | Deviation | Owned where |
|---|---|---|
| 1 | Qwen2.5 0.5B/1.5B/3B vs the paper's Sonnet-scale subject | standing (M0) |
| 2 | Single-token filter: 180/192 trials, 60/64 baselines (4 animal answers unencodable) | this brief, measured |
| 3 | Swap rows use prompt-position token forms (M2 used bare) | this brief, assumption |
| 4 | Fit corpus n=100 vs paper-grade ~1000 | standing (M0) |
| 5 | α ∈ {4, 8} beyond the paper's published {1, 2} (if D21-A) | this brief |
| 6 | MPS fp32 vs CUDA | standing (M0) |

## Wall-clock plan (forward passes only, prompts ≤ ~20 tokens)

Per subject: 64 baseline forwards (+ loading readout, free) + 180 × |α| × |arms|
swap forwards ≈ **1,500 forwards** at the full D20-A × D21-A grid. S1 measured
~2/7/12 min per subject for runs of this order with *longer* prompts; estimate
**≲ 5/10/20 min** for 0.5B/1.5B/3B, well under an hour serial, **$0**, no fit,
no rented GPU. Serial-GPU discipline unchanged.

## What S2 does NOT decide

- Selectivity (the last stretch property) — separate scope, still gated on Kyle.
- Any new model, fit, or band; the only operator change is the α parameter on
  the existing swap, invariant-gated.
- Framing stays **descriptive** (triple readability NULL; the pre-registered
  re-scope holds — S2 characterizes routing, it does not reclassify it as a
  reproduction claim).
- Nothing in this PR is harness code; the brief precedes the build
  (per-stage rhythm).
