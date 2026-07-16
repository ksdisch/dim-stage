# M1 start-of-stage brief — verbal report

*2026-07-15 · status: **D5–D8 frozen 2026-07-16** (Kyle — all four
recommendations; see "Frozen decisions"). Mode resolved **all-descriptive**
by the 3B NULL. Intervention module + D6 gate: `intervention.py` +
`test_intervention.py`; measurement runners next.*
*Sources: the paper (`refs/workspace-paper.md`, §"The J-space supports verbal report",
§"Technical details of J-lens use cases") and the reference repo's experiment data
(`refs/jacobian-lens/data/experiments/README.md`, `verbal-report.json`,
`verbal-introspection.json`). Neither is committed; `readability.py`'s error messages
carry the refetch commands.*

## What M1 is, in plain terms

M0 asked: *can our lens read anything out of the workspace band?* (Answer at
0.5B/1.5B: no, at the frozen bar.) M1 asks the paper's first functional question:
**does the model's verbal report track the workspace — and if you rewrite the
workspace, does the report follow?**

This is the project's first **intervention** milestone. M0 only ever *read*
activations; M1 *writes* them mid-forward-pass and watches what the model says.
New vocabulary, defined once:

- A **J-lens vector** `v_t` is one token's direction in the residual stream at a
  given layer: the direction whose presence makes the lens (and, the paper argues,
  the model's future output) favor token `t`. Concretely `v_t = J_lᵀ u_t`, where
  `u_t` is token `t`'s row of the unembedding matrix — the lens logit for `t` is
  (approximately) the inner product `⟨v_t, h⟩`.
- A **swap** (the paper says *patching in lens coordinates*) exchanges one
  concept for another inside an activation while touching nothing else: read off
  how much of `h` lies along `v_s` (the concept the model had) and `v_t` (the one
  we're implanting), exchange those two coordinates, leave every direction
  orthogonal to both exactly alone.
- **Steering** just adds a scaled concept direction: `h ← h + α·v_t`. Negative α
  (or projecting the component out) is an **ablation**.
- A **pseudoinverse** (`V†`) is the standard least-squares way to read
  coordinates against a set of non-orthogonal directions — here just a 2-column
  solve; nothing exotic.
- A **prefill** (teacher-forcing) means we *write the model's reply for it* up to
  a chosen point and only ask for the next token after that point. It makes the
  readout deterministic — no sampling, ever.
- **Reciprocal rank** is `1/rank`; the **median reciprocal rank (MRR)** over
  trials is the paper's introspection metric (rank 1 → 1.0, rank 10 → 0.1).

M1's deliverables: (1) this brief's decisions frozen; (2) the intervention module
(swap + steer, with pre-committed correctness invariants — see D6); (3) the
verbal-report swap measurement on all subjects; (4) the verbal-introspection
steering measurement (if D8 keeps it in); (5) per-subject verdicts in the frozen
mode (measured or descriptive — see "Mode" below).

## Design extraction (verbatim, source-cited)

### The two intervention operators (paper, §Technical details of J-lens use cases)

**Steering / ablation:**

> The simplest intervention is steering along a J-lens vector: `h ← h + α·v_t`,
> applied at one or more layers and token positions. With negative α, or by
> projecting out the component of `h` along `v_t` entirely, this becomes an
> ablation. … We use positive steering to test introspective detection of an
> injected concept.

**Swap (patching in lens coordinates):**

> Given a source token `s` and target token `t`, we form `V = [v_s v_t]`, read
> the lens coordinates `c = V†h` (where `V†` is the pseudoinverse of `V`), and
> set `h_patched = h + V(σ(c) − c)`, where σ swaps the two entries of `c`
> (optionally scaled by a factor α). The component of `h` orthogonal to
> span{v_s, v_t} is unchanged.

The paper's plain-words version of the same operation (§verbal report): *"we
subtract the projection onto the Soccer lens vector and add an equal-magnitude
projection onto the Rugby lens vector."* The formula is the spec; the sentence is
the intuition.

**The steering direction's scale** (reference README, verbal-introspection):

> its Jacobian-lens steering direction — the unit-normalized transpose row for
> that token, scaled by the layer's mean residual norm times a strength scalar —
> is added to the residual stream at every band layer and every token of the
> user's question turn; strength 0 is the control.

So steering strengths are in units of the layer's typical residual size — a
strength of 1 adds a vector as long as an average residual. *Which corpus defines
"mean residual norm" is unspecified in the released material* — a convention we
must freeze (D5).

### Protocol 1 — verbal report (swap). Reference README, verbatim:

> `candidates` maps 14 category names → 14 words each. The prompt is
> `Think of a {category}. Answer in one word.`; the model's greedy next token at
> the final `:` is taken as the answer and used as the swap-out target. For each
> of the first 10 listed candidates (skipping the answer itself), swap
> answer→candidate across the band at every prompt position. Grading: the
> swapped-in candidate's rank in the output distribution at the final `:`;
> success = rank 1.

Paper details that pin the rest (§The J-space supports verbal report):

- Swap applied "at all token positions", "leaving the rest of the activation
  unchanged, and allow the forward pass to continue."
- The scaled-up experiment swaps in targets "excluding those that were already in
  the top-10 of the model's possible outputs", and Figure 6's grading is
  "restricted to candidates starting at rank ≥ 11."
- "the final `:`" is the colon of Claude's `Assistant:` prompt format — i.e. the
  **final prompt token**, the position whose next-token distribution produces the
  first answer token. (Qwen has no such colon; see D5 and deviations row 2.)

### Protocol 2 — verbal introspection (steering). Reference README, verbatim:

> The model is told a thought may have been injected and asked to identify it
> (`intro_prompt`); one of `prefills` is teacher-forced as the reply, ending in
> an open quote so the next predicted token is the reported word. For each
> `surface` in `concepts`, its Jacobian-lens steering direction … is added to the
> residual stream at every band layer and every token of the user's question
> turn; strength 0 is the control. Score: the rank of `surface` in the next-token
> distribution at the open quote (the last prefill token). The figure reports
> median reciprocal rank vs strength.

Shipped data: `intro_prompt` is a 4-message chat (researcher explains 50% of
trials have an injected thought; assistant acknowledges; "Trial 1: Do you detect
an injected thought?"); two prefill variants, `default` ("…The thought is about
\"") and `word` ("…about the word \""); **101 concepts** (paper's figure says
n=100).

### Anchors (the paper's numbers we compare against — all Claude Sonnet 4.5)

| Anchor | Value | Source |
|---|---|---|
| Verbal-report swap: targets reaching **top-5** | **88%** (Wilson 95% CIs in figure) | Figure 8 (middle); KICKOFF anchor row cites "88% top-5 (N=90), rank-1 grading" |
| Introspection: report rate at best strength | "majority of trials"; **0.54** cited in the broadcast-heads section | §verbal report; §broadcast hub |
| Introspection: MRR vs strength curve | rises then saturates; interquartile band over n=100 concepts | Figure 7 |

The eligible-trial count for the swap anchor isn't stated next to the 88%; our
run pins its own N (≤ 140 = 14 categories × 10 candidates, minus rank-≥-11 and
tokenizer exclusions) and reports it.

### What the reference does NOT ship (checked 2026-07-15)

**No intervention or experiment-protocol code exists in the reference repo.**
`jlens/protocol.py` is a model-interface definition, `jlens/lens.py` is read-only
lens application, `jlens/hooks.py` is an activation recorder. The operators above
exist only as paper formulas + data-README conventions. Consequence: **M1 cannot
have an M0-style AGREE gate** — there is no oracle to diff against. Correctness
must come from pre-committed invariants instead (D6).

## Mode: measured vs descriptive (branch-proof)

Pre-registered in KICKOFF (risk #1) and ROADMAP: a readability-NULL subject gets
**descriptive** M1 verdicts — we run the identical protocol and report the same
Wilson/Newcombe numbers, but the headline claim is *characterization*
("here is what swaps do at a scale where the workspace isn't readable"), never
*reproduction*. The 3B escalation verdict (due 2026-07-16 AM) picks the branch:

- **3B READS** → M1 runs **measured** on 3B (D7 gate applies); 0.5B/1.5B still
  run, descriptive, for the scaling story.
- **3B NULL** → all subjects descriptive; D7's gate text still freezes *now* so
  the descriptive/measured wording isn't chosen after seeing results.

**Resolved 2026-07-16: 3B is NULL (0/6) — all subjects run descriptive.**

## Decisions to freeze (Kyle picks each)

### D5 — Trial-set and readout conventions (the unspecified parts, frozen before any run)

- **A. Reference-faithful package (recommended).**
  (1) Prompts built with the tokenizer's chat template
  (`apply_chat_template`, generation prompt appended) — new in M1; M0's
  lens-evals were raw text. (2) Readout position = **final prompt token**, the
  M0 precedent (`readability.py` scores `seq_len − 1` everywhere but poetry) and
  the exact analog of the paper's `Assistant:` colon. (3) Swap-out `s` = greedy
  next token at that position (single-token by construction). (4) Swap-in
  trials = first 10 listed candidates per category, skipping the spontaneous
  answer (README), graded only where baseline rank ≥ 11 (paper); candidates
  must pass the M0 single-token filter ({`w`, `␣w`} forms; drops counted).
  (5) Mean residual norm (steering scale) = per-layer mean L2 norm over the
  frozen D3 fit corpus (N=100 WikiText prompts) — deterministic, already
  frozen, stored with the run config. *Merit:* every choice is cited or
  inherits an M0-frozen convention; like-for-like with the anchor.
  *Trade-off:* rank-≥-11 + tokenizer exclusions shrink N below 140.
- **B. Maximize N.** All 14 candidates, no rank restriction. *Merit:* more
  trials per cell. *Trade-off:* breaks like-for-like with the 88% anchor (which
  excludes already-likely candidates — swapping in a candidate the model
  already ranked #3 isn't evidence the swap did anything); not recommended.

### D6 — Intervention correctness gate (replaces the impossible AGREE gate)

- **A. Pre-committed invariants as tests (recommended).** Before any real run,
  the intervention module must pass: (1) *rigged-subject oracle* — a hand-built
  tiny model (M0's `RiggedSubject` test pattern) where the swap's effect on the
  output distribution is known analytically; (2) *coordinate check on real
  subjects* — after patching, the lens coordinates read back exactly swapped
  (`c' = σ(c)` to fp tolerance) and the component orthogonal to span{v_s, v_t}
  is unchanged (‖Δh⊥‖ = 0); (3) *null-op checks* — α=0 steering and s=t swaps
  are bitwise no-ops. *Merit:* machine-checkable, pre-committed, catches sign /
  transpose / normalization bugs — the failure modes that would silently fake a
  null. *Trade-off:* invariants can't catch "faithful to the formula but the
  formula was misread"; mitigated by the verbatim extraction above.
- **B. No formal gate** (trust the formulas, eyeball one example). *Merit:*
  fastest. *Trade-off:* violates the pre-commit bar (gates as code, dry-run
  before real runs); not recommended.

### D7 — Verdict conditions (both modes frozen now, wording pre-declared)

- **A. Two-arm, mirroring D4 (recommended).**
  - *Arm 1 — swap success (gates only in measured mode):* per eligible trial,
    hit iff the swapped-in candidate reaches **top-5** at the readout position
    (the anchor's k). Measured-mode gate: Wilson 95% LB ≥ **0.5** pooled over
    eligible trials → *verbal report tracks the workspace* on that subject;
    below → NULL. Rank-1 rate (the README's stricter "success") and top-10
    reported descriptively alongside.
  - *Arm 2 — J-transport advantage (standing falsification arm):* repeat every
    swap with raw unembedding rows (`v = u_t`, i.e. **J = I**) in place of
    J-lens vectors; Newcombe 95% CI on the difference. The M0 continuity: at
    1.5B the J-transport was *worse* than identity on typo — M1 tests whether
    the same inversion holds for *writing*, not just reading.
  - In descriptive mode both arms report the identical numbers with the
    pre-declared descriptive framing; no property claim either direction.
- **B. Gate on rank-1** (README's per-trial "success"). *Merit:* strictest.
  *Trade-off:* not the anchor's grading (Fig 8 gates top-5); a rank-2 landing
  that the paper counts as success would count as failure here — conflates
  "swap works" with "swap works perfectly"; not recommended as the gate.

### D8 — Introspection arm scope

- **A. Include, descriptive-first (recommended).** Run Protocol 2 on every
  subject with strength grid **α ∈ {0, 0.5, 1, 2, 4, 8}** (mean-norm units;
  0 = the README's control), `default` prefill primary, `word` prefill
  reported descriptively. Metrics: MRR vs strength (the paper's figure) +
  report rate (rank-1 at the open quote) at each strength, Wilson CIs.
  Never gates M1's verdict — the paper itself treats it as the softer,
  qualitative-leaning result ("majority of trials"). *Merit:* the steering
  operator is required for M3 (directed modulation) anyway — building and
  validating it here is free leverage; the injected-thought result is the
  paper's most talked-about phenomenon and a natural hobby-scale question.
  *Trade-off:* ~700 forwards per subject of added wall-clock (small); one more
  owned convention (the strength grid — the paper sweeps but doesn't publish
  its grid).
- **B. Defer introspection out of M1.** *Merit:* single-protocol milestone.
  *Trade-off:* M3 pays the steering build cost later without D6's invariants
  already battle-tested; the "(+ verbal-introspection.json)" scope in KICKOFF
  goes unaddressed.

### Assumption on record

Subjects = the M0 set (0.5B, 1.5B) plus 3B iff the escalation verdict lands it;
per-subject bands = the frozen D2/`FROZEN_BANDS` table; swaps and steering are
applied at **every band layer** (both protocols say so) and graded at the single
readout position. Costs nothing to change before coding starts.

## Deviations table (starter — grows as M1 runs)

| # | Paper / reference | Ours | Why | Status |
|---|---|---|---|---|
| 1 | Subjects: Claude Sonnet 4.5 (+Haiku/Opus corroboration) | Qwen2.5 0.5B/1.5B(/3B) Instruct | The project's thesis | Owned, standing |
| 2 | Readout at the `Assistant:` colon (Claude prompt format) | Final prompt token under Qwen's chat template | Qwen has no colon; M0 precedent (`readability.py` scores `seq_len−1`) | Owned, pre-declared |
| 3 | Candidates as shipped | M0 single-token pre-filter ({`w`,`␣w`}) on swap-in candidates | Rank-based grading needs single-token targets | Owned, mechanical |
| 4 | (implicit) intervention code validated in-house | No reference oracle exists → D6 invariants instead of an AGREE gate | Reference ships data only | Owned, pre-declared |
| 5 | Introspection n=100 concepts (figure) | 101 shipped concepts, minus tokenizer drops | Use the data as shipped | Owned, mechanical |
| 6 | Mean-residual-norm corpus unspecified | Frozen D3 fit corpus (N=100 WikiText) | Deterministic, already frozen | Owned, pre-declared |
| 7 | v_t: "row of the unembedding matrix" (norm details of Claude's stack unstated) | Raw `lm_head.weight` row; Qwen's final-RMSNorm scale γ not folded in | Literal formula reading; γ-folding is an uncited variant | Owned, pre-declared |

## Wall-clock plan (from measured M0 rates; forward passes only — no fitting)

Interventions are hooked forward passes; prompts are ~30–60 tokens. Per subject:
verbal report ≤ 14 baselines + ~2 × 140 swap forwards (J arm + J=I arm);
introspection ~101 × 6 strengths + control ≈ ~700 forwards. Even at 3B this is
minutes-to-an-hour per subject on MPS — nothing here needs an overnight slot.
Serial GPU discipline unchanged: no M1 run shares the machine with a fit.

## What M1 does NOT decide

- M2 (probe-swap) / M3 (directed-modulation) protocol details — own briefs;
  both reuse the swap/steer operators and D6 invariants built here.
- Whether 3B joins the subject list — that's the escalation verdict + Kyle.
- Nothing in this PR is harness code; the brief precedes the build
  (per-stage rhythm).

## Frozen decisions

*Frozen 2026-07-16 (Kyle) — all four recommendations, recorded here and in
`DECISIONS.md`. Relitigating after this is a deviation row, not a
conversation.*

- **D5 = A** — reference-faithful package: chat-template prompts;
  final-prompt-token readout; greedy swap-out; first-10 candidates with
  baseline rank ≥ 11 and the M0 single-token filter; steering scale from the
  frozen D3 corpus.
- **D6 = A** — pre-committed invariants, merged before any run
  (`test_intervention.py`): rigged analytic oracle, coordinate read-back,
  null-ops, literal-formula agreement.
- **D7 = A** — two-arm verdict: Arm 1 top-5 (Wilson LB ≥ 0.5 wording frozen;
  descriptive framing applies — triple NULL), Arm 2 J = I falsification
  (Newcombe). Rank-1 / top-10 descriptive.
- **D8 = A** — introspection in, descriptive-first: α ∈ {0, 0.5, 1, 2, 4, 8},
  `default` prefill primary, never gates.

Implementation conventions recorded alongside (DECISIONS.md 2026-07-16):
v_t from the raw `lm_head.weight` row (deviations row 7); swap via the
identity (c_t − c_s)(v_s − v_t) with s = t an exact no-op; closed-form 2×2
coordinate solve, (near-)parallel pairs rejected.
