# M2 start-of-stage brief — two-hop swap

*2026-07-16 · status: **COMPLETE** — D9–D11 frozen (Kyle, all three
recommendations; see "Frozen decisions"), measured on all three subjects the
same day. Results below; headline in `ROADMAP.md`; outcomes in
`DECISIONS.md`.*
*Sources: the paper (`refs/workspace-paper.md`, §Lens-coordinate swaps redirect
internal reasoning / Figures 13–16) and the reference repo's experiment data
(`refs/jacobian-lens/data/experiments/README.md` §probe-swap,
`probe-swap.json`). Neither is committed; refetch commands live in
`readability.py`'s error messages.*

## What M2 is, in plain terms

A **two-hop prompt** forces the model to compute an unspoken middle step: *"The
language spoken in the country where the Amazon River ends is …"* requires
inferring **Brazil** (the **bridge entity** — the intermediate the prompt never
names) before answering *Portuguese*. M1 swapped the model's *answer* as it was
about to speak; M2 swaps the *bridge* — mid-reasoning, one hop upstream — and
asks whether the **final answer follows the redirected chain**: swap
Brazil → Mexico inside the residual stream, and does the model now say
*Spanish*?

This is the paper's sharpest causal claim about the workspace ("J-lens vectors
mediate internal reasoning"), and M2 reuses everything M1 built: the same swap
operator (`intervention.py`, D6-gated, runtime read-back included), the same
frozen bands, the same two-arm falsification structure. No new operator code —
only a runner and the trial conventions below.

## Design extraction (verbatim, source-cited)

**The protocol** (reference README §probe-swap, verbatim):

> 90 two-hop factual prompts. `items[*].prompt` ends just before the answer;
> `intermediate` is the bridge entity, `swap_to` the replacement. Baseline:
> greedy next-token == `answer`. Swap: replace the `intermediate`
> representation (linear-probe direction) with `swap_to` across the band at
> every prompt token position; score next-token at the final position ==
> `swap_answer`. `category` groups items by relation type for the per-category
> breakdown.

**The systematic experiment** (paper, §two-hop, verbatim):

> We evaluate the role of Jacobian lens vectors in multi-step reasoning more
> systematically, using a set of 50 two-hop factual prompts with known
> intermediates … We measure the fraction of trials in which the swap moves
> the target-appropriate answer to the top of the model's output distribution.
> The Jacobian-lens coordinate swap succeeds in 54% of trials on Haiku 4.5,
> 70% on Sonnet 4.5, and 70% on Opus 4.5.

**The n=90 anchor** (paper, Figure 16 + text): the shipped 90-item set is the
Figure 16 experiment; its **raw J-lens token-vector swap** — our protocol —
"flips the model's answer to the swapped-in intermediate on … **60%**" of the
90 trials (Sonnet 4.5, top-1 grading, Wilson 95% CIs in the figure). The
shipped items fix each `swap_to` pairing (the 50-prompt experiment chose
targets at random within category; the data as shipped is deterministic —
used as-is, mechanical).

**Extraction wrinkles, owned:**

- The README's parenthetical "(linear-probe direction)" describes Figure 16's
  *probe* variants; the raw J-lens token-vector swap on the same 90 prompts is
  the paper-text protocol M2 reproduces (its 60% bar sits beside the probes'
  61%/28% in the same figure). The probe apparatus is not shipped (below).
- Figure 13's caption calls the qualitative examples a "**clamped**
  lens-coordinate swap at every position". No clamping procedure is defined
  anywhere in the released material; the only shipped swap definition is the
  §technical-details formula (M1-BRIEF, design extraction) applied across the
  band — which is what `intervention.py` implements and D6 gates. Any
  stronger "hold the coordinate at every layer" variant is unspecified;
  deviations row 3.

### Anchors (Claude models; ours run descriptive against them)

| Anchor | Value | Source |
|---|---|---|
| Two-hop swap success (top-1 flip), 50 prompts | **54% / 70% / 70%** (Haiku / Sonnet / Opus 4.5) | §two-hop; Figure 15 left |
| Raw J-lens token-vector swap, the shipped n=90 | **60%** (Sonnet 4.5, Wilson 95% CIs) | Figure 16 right |
| Probe J-space / non-J-space / clamped variants | 61% / 28% / 6% — **not in M2 scope** | Figure 16 right |

### What the reference does NOT ship (checked 2026-07-16)

The Figure 16 **probe decomposition is not reproducible from shipped data**:
fitting each intermediate's probe needs "a set of prompts that imply the same
intermediate through different surface cues" (unshipped), plus the gradient-
pursuit split against a J-lens dictionary (k=25; unshipped). `probe-swap.json`
carries only the 90 items. Consequence: M2 = the **raw J-lens token-vector
swap**, which is also exactly KICKOFF's core-three scope ("two-hop swap");
the probe comparison was never in v1. No new intervention code is needed —
M1's operator + D6 invariants carry over unchanged.

## Mode: descriptive (settled)

All three subjects are readability-NULL (M0), so M2 verdicts are
characterization, never reproduction claims — same pre-registered re-scope as
M1; gate wording still freezes now (D10) so nothing is chosen after results.

## Inherited frozen conventions (M1 — not re-decided)

Raw-text prompts get M0's treatment; the swap token is the **bare** single-token
form, grading is min over {`w`, `␣w`} forms, items failing the single-token
pre-filter drop and are counted; `v = J_lᵀu` from raw `lm_head.weight` rows,
each band layer its own `J_l`; swaps at **every prompt position** (README:
"every prompt token position"); D6 runtime read-back on every application;
INVALID (exit 2) validation + `--dry-run` before any real run.

## Decisions to freeze (Kyle picks each)

### D9 — Trial set, readout, and baseline conditioning

- **A. Baseline-conditioned primary cell (recommended).** All 90 shipped items;
  raw-text encoding (these are `Fact: …` completions, not chat — M0's
  lens-eval precedent); readout = final prompt token. A trial's **baseline
  passes** iff the unswapped greedy next token is among the `answer`'s
  single-token forms (the README's "Baseline: greedy next-token == answer").
  **The primary swap cell = baseline-passing items only** — a "flip" needs a
  working chain to redirect; a swap "success" on an item the model couldn't
  answer anyway is not evidence of redirected reasoning. Baseline accuracy and
  the unconditioned all-items rate are reported alongside. *Merit:* measures
  the causal question; keeps the anchor's meaning ("flips the model's
  answer"). *Trade-off:* small-model two-hop accuracy may shrink the primary
  cell — possibly below N=20 at 0.5B → pre-declared UNDERPOWERED, which is
  itself a reportable fact.
- **B. Unconditioned n=90 primary.** Grade every item regardless of baseline.
  *Merit:* fixed N, matches the figure's n=90 denominator (Claude's baseline
  is presumably ~all-correct, so for the paper the two readings coincide).
  *Trade-off:* at our scale it conflates "can't do two-hop at all" with "swap
  didn't redirect"; not recommended as primary (it is reported anyway).

### D10 — Verdict conditions (frozen wording; descriptive mode)

- **A. Two-arm, top-1 flip, mirroring D7 (recommended).**
  - *Arm 1:* success = the swapped run's greedy next token is among
    `swap_answer`'s single-token forms — **top-1, the anchor's grading**
    ("moves the target-appropriate answer to the top"). Wilson 95% CI on the
    primary cell; frozen would-gate wording: LB ≥ 0.5 (the project-wide bar,
    sitting just under the 54–70% anchor range). Reported descriptively:
    `swap_answer`'s rank (top-5/top-10), the **displaced-original** rate (the
    old answer no longer top-1 — did the swap at least knock the chain over),
    and the per-category breakdown (the README's `category` field; 29
    multihop + 61 simpler relational items).
  - *Arm 2:* every swap repeated with raw unembedding rows (**J = I**),
    Newcombe 95% CI on the difference — the standing falsification arm,
    unchanged.
- **B. Gate on top-5** (M1's k). *Merit:* consistency with M1's Arm 1.
  *Trade-off:* not this anchor's grading — the two-hop numbers (54–70%, 60%)
  are top-1 flip fractions; a top-5 gate would inflate our rate relative to
  the anchor. Top-5 is reported descriptively either way; not recommended as
  the gate.

### D11 — Answer-swap comparison arm (the paper's smuggling confound)

- **A. Include, descriptive-only (recommended).** The paper's named confound:
  maybe swapping *Brazil→Mexico* works only because the Brazil vector
  "smuggles in" some *Portuguese* (the answer), not because the reasoning
  chain reruns. The paper's full test sweeps layer ranges (out of scope), but
  its comparison object is cheap here: repeat every trial swapping the
  **answer tokens** (`answer → swap_answer`) instead of the intermediates, at
  the same frozen band — 2 extra forwards per item. If the intermediate swap
  only smuggles answers, the two arms should look alike; if the chain reruns,
  they can differ. Never gates; one owned convention (single band, no depth
  sweep — deviations row 4). *Merit:* addresses the confound the paper itself
  raises, at ~180 forwards per subject.
- **B. Skip it.** *Merit:* leanest possible M2. *Trade-off:* the smuggling
  confound goes unexamined at our scale, and the comparison is nearly free.

### Assumption on record

Subjects = all three (0.5B/1.5B/3B, lenses on disk); bands = the frozen
D2/`FROZEN_BANDS` table; swap arms share one baseline forward per item. Costs
nothing to change before coding starts.

## Deviations table (starter)

| # | Paper / reference | Ours | Why | Status |
|---|---|---|---|---|
| 1 | Subjects: Claude 4.5 tiers | Qwen2.5 0.5B/1.5B/3B Instruct | The project's thesis | Owned, standing |
| 2 | 50-prompt experiment draws swap targets randomly within category | The shipped 90-item set's fixed pairings | Data as shipped, deterministic | Owned, mechanical |
| 3 | Figure 13 says "clamped" swap; no procedure defined | The §technical-details swap formula applied at every band layer (M1's operator) | Only shipped definition; clamping variant unspecified | Owned, pre-declared |
| 4 | Smuggling confound tested via layer-range sweep (Fig 15 right) | Single-band answer-swap comparison arm (if D11 = A) | Depth sweep out of v1 scope | Owned, pre-declared |
| 5 | Claude baseline ~all items answered correctly | Baseline accuracy is a measured cell; primary conditioning per D9 | Small-model capability differs | Owned, pre-declared |

## Wall-clock plan (from measured M1 rates; forward passes only)

Per subject: 90 baselines + 90 × 2 swap arms (+ 90 × 2 answer-swap if D11 = A)
≈ 270–450 short raw-text forwards — about a minute at 0.5B, a few minutes at
3B on MPS. Serial GPU discipline unchanged.

## Results — two-hop swap (all subjects, descriptive) — 2026-07-16

Runner: `m2_two_hop.py` (D9 baseline conditioning; D10 top-1 grading, both
arms; D11 answer-swap comparison; D6 runtime read-back on every application).
JSONs in `results/two-hop-*.json`. Anchor: **60%** top-1 flips on these 90
prompts (Sonnet 4.5); 54–70% across Claude tiers. 9 of 90 items dropped by
the four-field single-token pre-filter (shared tokenizer, same 9 everywhere);
no primary cell fell below N = 20.

Primary cells (baseline-correct items only; flips = swapped greedy token ∈
`swap_answer`'s forms):

| Subject | Baseline | Intermediate flips, J-lens (Wilson 95%) | J = I | J − I (Newcombe 95%) | Answer-swap J / I (D11) | Displaced | swap_answer top-5 |
|---|---|---|---|---|---|---|---|
| 0.5B | 28/81 = .346 | 8/28 = **.286** [.153, .471] | 4/28 | +.143 [−.075, +.347] | 16/28 / 14/28 | 14/28 | 15/28 |
| 1.5B | 41/81 = .506 | 3/41 = **.073** [.025, .194] | 0/41 | +.073 [−.025, +.194] | 9/41 / 7/41 | 12/41 | 22/41 |
| 3B | 43/81 = .531 | 5/43 = **.116** [.051, .245] | 0/43 | **+.116 [+.011, +.245]** | 6/43 / 3/43 | 18/43 | 23/43 |

Descriptive reading (no property claims — triple readability NULL):

- **The chain mostly does not follow the bridge swap at these scales.** Flip
  rates sit at .07–.29 against the .60 anchor. The 0.5B "high" comes with its
  own caveat: only 28 chains worked to begin with, and flimsy chains flip
  easily — exactly the conflation D9's conditioning exists to expose.
- **The project's first CI-clean J-transport advantage for writing.** At 3B,
  raw unembedding rows flip **0/43** while J-lens vectors flip 5/43 —
  Newcombe +.116 [+.011, +.245] excludes zero. At 1.5B the identity arm is
  also 0-for-41 (diff not CI-clean). After M0's reading reversal and M1's
  nothing, this is the first cell where the Jacobian transport is the
  difference between some effect and none. Arm 2 never gates; this is a
  descriptive contrast.
- **The smuggling confound comes out clean at 3B** (D11): direct answer swaps
  flip 2× the intermediate rate at 0.5B (16 vs 8), but by 3B they are equal
  (6 vs 5) — the intermediate swap is not just riding a smuggled answer
  component there, and the raw-row zeros corroborate (a smuggled answer
  would survive J = I). Single-band comparison only; the paper's depth sweep
  is out of scope (deviations row 4).
- **The swap disturbs chains far more often than it redirects them.** The
  original answer is displaced on ~40% of primary trials (14/28, 12/41,
  18/43), and the redirected answer reaches top-5 about half the time
  (15/28, 22/41, 23/43) — the intervention lands, but rarely lands *on
  target* at these scales.
- **Two-hop capability itself is the bottleneck at 0.5B** (baseline .346) and
  only reaches ~.53 by 3B — Claude answers essentially all 90; our subjects
  answer half. Both facts are cells in the JSONs.

## What M2 does NOT decide

- M3 (directed modulation) — its own brief; it reuses steering + D6.
- Nothing here is harness code; the brief precedes the build (per-stage
  rhythm). The runner lands with D9–D11 frozen in the same PR.

## Frozen decisions

*Frozen 2026-07-16 (Kyle) — all three recommendations, recorded here and in
`DECISIONS.md`. Relitigating after this is a deviation row, not a
conversation.*

- **D9 = A** — baseline-conditioned primary cell (raw-text encoding,
  final-prompt-token readout, strict {`w`, `␣w`} forms of the shipped
  strings, no synonym table); baseline accuracy + unconditioned rates
  reported alongside.
- **D10 = A** — top-1 flip grading (the anchor's), Wilson on the primary
  cell, frozen would-gate wording LB ≥ 0.5 (descriptive framing applies);
  J = I falsification arm with Newcombe; top-5/top-10, displaced rate,
  per-category breakdown descriptive.
- **D11 = A** — answer-swap comparison arm (both arms, same band),
  descriptive-only.
