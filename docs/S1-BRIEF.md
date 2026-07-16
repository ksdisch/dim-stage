# S1 start-of-stage brief — introspection dose–response follow-up (stretch)

*Start of stage · status: **decisions pending Kyle** (D15–D18 below). First stretch
stage, opened after v1 close (2026-07-16, ROADMAP). Per-stage rhythm: this brief +
options land before any code.*
*Sources: the paper (`refs/workspace-paper.md`, §"The J-space is available for
verbal report" / Figure 7 and its specificity control, §"Comparing J-space across
layers"); the reference repo (`refs/jacobian-lens/data/experiments/verbal-introspection.json`).
Neither is committed; the runner's error messages carry the refetch command.*

## What S1 is, in plain terms

M1 produced the one place a small Qwen model actually *did the paper's thing*: at
**1.5B**, a thought we steered into the residual stream became **reportable** — asked
"do you detect an injected thought?", the model named the injected concept. The report
rate rose monotonically with steering strength, **0 → 30 of 101 concepts**, against a
"no steering" control that was **exactly 0/101** everywhere. 0.5B stayed flat zero; 3B
reached only 5/101. It is the project's strongest, most paper-shaped result.

But it is also the project's *least-hardened* result. Two things the paper does around
this exact experiment, we never did:

1. **We never checked the curve is a *J-transport* effect.** Every other milestone
   (M1 verbal-report, M2, M3) carried a standing **falsification arm** — rerun the
   identical operation with `J = I` (the plain logit lens: steer along the raw
   unembedding direction, no Jacobian transport) and take the Newcombe CI on the gap.
   The introspection curve is the *only* headline finding with **no such arm**. So we
   cannot yet say the 1.5B dose–response is a *workspace* phenomenon rather than
   something any token-derived direction would do. The paper makes exactly this
   contrast and says the J-space component is what drives the report (extraction §2).
2. **We never saw it saturate.** The paper's Figure 7 shows median reciprocal rank
   *rising then saturating* with strength. Our grid stopped at α=8 with the 1.5B curve
   still climbing (26→30 from α=4→8). We don't know if it saturates (paper shape),
   keeps climbing, or collapses (over-steering breaks the model).

And one thing the paper's broader analysis invites that would turn a *rate* into an
*internals* result:

3. **We don't know *where* in the band the reportability lives.** We steer **all 14
   band layers at once** (L11–L24 at 1.5B). The paper locates the workspace in the
   model's *middle block* and notes the J-space "stores different concepts in different
   layers" (extraction §3). Steering sub-bands separately would localize where 1.5B's
   reportability actually emerges — a genuine interpretability-internals finding, not
   just a bigger number.

S1 is the focused follow-up that closes these gaps on the flagship finding. It adds
**no new model, no new fit, no rented GPU** — it reuses the frozen 1.5B lens, the
shipped 101-concept protocol, and the M1 steering operator, all already validated.

## The finding we build on (M1, verbatim — `results/introspection-*.json`)

Report rate = rank-1 of the steered concept's word at the open quote, `default` prefill,
all 101 concepts single-token-valid on Qwen's tokenizer (0 dropped):

| α (mean-norm units) | 0.5B | 1.5B | 3B |
|---|---|---|---|
| 0 (control) | 0/101 | 0/101 | 0/101 |
| 0.5 | 0/101 | 7/101 | 2/101 |
| 1 | 0/101 | 16/101 | 2/101 |
| 2 | 0/101 | 23/101 | 4/101 |
| 4 | 0/101 | 26/101 | 4/101 |
| 8 | 0/101 | **30/101 = .297 [.217, .392]** | 5/101 = .050 [.021, .111] |

The 1.5B–3B gap at α=8 is CI-clean. Steering moves *ranks* on every subject (median
rank at 1.5B: control 4430 → α=8 15, ~300×); only 1.5B converts movement into reports.

## Design extraction (verbatim, source-cited)

**§1 — the steering operator (unchanged from M1, reused verbatim).** Reference README,
verbal-introspection: the injected direction is *"its Jacobian-lens steering direction —
the unit-normalized transpose row for that token, scaled by the layer's mean residual
norm times a strength scalar — added to the residual stream at every band layer and
every token of the user's question turn; strength 0 is the control."* Frozen in M1 as
D5/D8; S1 keeps it identically (mean-norm from the D3 corpus, α on top, question-turn
positions, rank-1-at-open-quote scoring, MRR + report-rate metrics).

**§2 — the specificity / falsification claim (paper, Figure 7 region):**

> at each condition's most effective injection strength, the J-space component produces
> a report of the injected concept nearly as often as the pure J-lens vectors do, while
> the non-J-space component produces few reports even at injection strengths several
> times larger.

The paper's contrast is J-space component vs non-J-space component. The project's
consistent, cross-milestone falsification convention is **`J = I`** — steer along the
unit-normalized *raw unembedding row* `u_t` (the logit-lens direction), scaled the same
way, at the same layers. This is the same Arm-2 substitution M1/M2/M3 used; S1 applies
it to the introspection curve for the first time. (That ours is `J=I` rather than the
paper's orthogonal-component decomposition is an owned deviation — deviations row 2.)

**§3 — saturation and localization (paper):**

> *(Figure 7 caption)* The adjacent plot tracks the median reciprocal rank of the
> injected concept against steering strength over n=100 concepts, with an interquartile
> band.

> *(§Comparing J-space across layers)* The resulting matrix has a clear block
> structure: an early block encompassing roughly the first third of the model, a long
> middle block, and a small late block … the workspace-like properties of the J-space
> reside only in the middle block.

> *(appendix, list analyses)* This difference indicates that the J-space's effective
> capacity is increased by the ability to store different concepts in different layers.

Figure 7's shape (rise-then-saturate) anchors the **saturation** axis; the middle-block
and different-concepts-in-different-layers claims anchor the **layer-localization** axis.

**§4 — position control (paper, Figure 7 caption):** readouts are taken *"at the comma
after 'If so', at the open quotation mark where the output is read, and at every other
position in the assistant turn as a position control"*, and the paper hypothesizes the
model attends to the J-space *on the user prompt* when answering. This anchors an
optional **position-localization** axis (which positions of the question turn carry it).

## The follow-up axes (menu — S1 picks a scope bundle below)

| Axis | Question it answers | Paper anchor | Cost (1.5B, MPS) | Value |
|---|---|---|---|---|
| **A. Saturation** | Does the curve rise-then-saturate (paper) or collapse (over-steer)? | Figure 7 shape | ~½× a rerun (extend α grid up) | closes the Figure 7 gap; cheap |
| **B. Falsification arm** | Is the dose–response a *J-transport* effect, or would raw unembedding do it too? | §2 (J-space vs non-J-space) | ~1× a rerun (add `J=I` arm) | **the missing arm on the flagship finding** |
| **C. Layer localization** | *Where* in L11–L24 does 1.5B reportability live? | §3 (middle block; different-concepts-in-different-layers) | ~several× (steer sub-bands separately) | the internals result; deepest learning |
| **D. Position localization** | Is steering the whole question turn needed, or do key positions suffice? | §4 (position control) | ~2–3× | mechanism texture; medium |
| **E. Concept selectivity** | Steering concept A — does *only* A become reportable, or generic words too? | implicit specificity | ~free (cross-score other tokens on saved logits) | precision check; cheap bonus |

A degeneracy guard rides along with A regardless: at high α the model can break and
emit the steered token for *everything*; we check the reply stays coherent (the α=0
control already rules out spontaneous reporting, but collapse needs its own check).

## Decisions to freeze (Kyle picks each; recommendations flagged)

### D15 — Scope bundle

- **Bundle 1 — "harden the flagship" (defensibility floor).** Axes **A + B** on all
  three subjects, focused on 1.5B. Adds the missing `J=I` falsification arm and extends
  the α grid to pin saturation vs collapse. *Merit:* fast (~1–2 h total), closes the two
  paper-anchored holes that most affect whether the finding is defensible. *Trade-off:*
  no new internals result — it strengthens the existing claim rather than extending it.
- **Bundle 2 — "localize it" (recommended).** Bundle 1 **+ axis C** (layer-localization
  sweep on 1.5B). *Merit:* turns "reportable at 1.5B" into "reportable via layers X–Y at
  1.5B" — a real internals finding, the on-thesis mid-layer question, and where the
  interpretability learning actually is. Still $0, still local, one extra afternoon of
  wall-clock. *Trade-off:* the localization sweep is the bulk of the compute; needs one
  granularity convention (D18).
- **Bundle 3 — "full characterization."** Bundle 2 **+ axes D and E**. *Merit:* fullest
  picture. *Trade-off:* diminishing returns per hour; two more owned conventions;
  position/selectivity are texture, not headline. Not recommended as the opener — D and E
  can be added later if C surfaces something worth chasing.

*Recommendation: **Bundle 2.** B is non-negotiable (it's the one hole in the strongest
finding); A is nearly free; C is where the deep-dive payoff and a novel result live. If
wall-clock or scope pressure is real, **Bundle 1** is the honest defensibility-first
floor and can stand alone.*

### D16 — Falsification convention (axis B)

- **A (recommended).** `J = I`: steer along the **unit-normalized raw unembedding row**
  `u_t`, scaled by the same per-layer mean residual norm × α, at the same band layers
  and question-turn positions. Newcombe 95% CI on (J-lens − J=I) report rate at each α.
  *Merit:* identical Arm-2 convention as M1/M2/M3 — like-for-like across the whole
  project; reuses the operator unchanged. *Trade-off:* it's the logit-lens direction,
  not the paper's orthogonal-non-J-space component (owned deviation).
- **B.** Reproduce the paper's exact J-space / non-J-space *component* decomposition of
  the steering direction. *Merit:* verbatim to §2. *Trade-off:* a new operator to build
  and gate, inconsistent with the project's standing `J=I` arm; not recommended.

### D17 — Extended α grid (axes A, B)

- **A (recommended).** Keep the frozen `{0, 0.5, 1, 2, 4, 8}` and **append `{12, 16, 24}`**
  (geometric-ish, ~3× headroom past the current top). Report the full grid; flag the
  first α where any subject's reply degenerates (degeneracy guard). *Merit:* enough
  headroom to see saturation or collapse without unbounded compute. *Trade-off:* the
  grid endpoints past 8 are an owned convention (the paper sweeps but publishes no grid —
  same footing as D8).
- **B.** Push much further (α up to 64) until every subject collapses. *Merit:* maps the
  full failure mode. *Trade-off:* mostly measures garbage-token regimes; low value.

### D18 — Localization granularity (axis C; only if D15 = Bundle 2/3)

- **A — sub-band thirds (recommended).** Steer three contiguous sub-bands of L11–L24
  separately — **early L11–15, middle L16–20, late L21–24** — at the best-reporting α
  (and its `J=I` arm), 101 concepts each. *Merit:* directly tests the "middle block"
  claim; 3 configs × ~2 arms is cheap and legible; a clean bar chart. *Trade-off:*
  coarse — can't resolve a single pivotal layer.
- **B — single-layer sweep.** Steer each of the 14 band layers alone. *Merit:* full
  resolution — pinpoints the layer(s) that carry it. *Trade-off:* 14 configs × arms ×
  101 concepts; more wall-clock and a busier readout. Recommended only if A shows a
  sub-band clearly dominates and the exact layer matters.

### Assumptions on record (cost nothing to change before coding)

- **Subjects:** 1.5B is primary (the finding). 0.5B and 3B get the A+B arms too (cheap,
  completes the scale table); the C localization sweep runs on **1.5B only** — 0.5B has
  no signal to localize and 3B's 5/101 is too thin. Kyle can widen C if desired.
- **Prefill:** `default` primary (M1 precedent); `word` reported alongside as in M1.
- **Band/scale/operator:** all frozen from M0/M1 — S1 introduces no new fit, band, or
  intervention primitive; the D6 runtime read-back self-check still fires on every steer.

## Deviations table (starter — grows as S1 runs)

| # | Paper / reference | Ours | Why | Status |
|---|---|---|---|---|
| 1 | Subjects: Claude (n=100 concepts) | Qwen 1.5B primary (+0.5B/3B), 101 shipped concepts | Project thesis; use data as shipped | Owned, standing |
| 2 | Specificity = J-space vs non-J-space *component* | `J = I` raw-unembedding falsification arm | The project's consistent cross-milestone Arm-2 convention | Owned, pre-declared (D16) |
| 3 | Figure 7 grid unpublished | Frozen `{0,0.5,1,2,4,8}` + appended `{12,16,24}` | Bounded headroom to see saturation/collapse | Owned, pre-declared (D17) |
| 4 | Localization via CKA block structure over all layers | Contiguous sub-band thirds of L11–L24, steered separately | Legible, hobby-scale localization of the *report* (not the geometry) | Owned, pre-declared (D18) |

## Wall-clock plan (from M1's measured introspection rates; forward passes only)

M1's full introspection run was ~700 forwards/subject at "minutes-to-an-hour" on MPS.
S1 adds, per subject: axis B ≈ 1× that (the `J=I` arm), axis A ≈ +3 α × 101 × 2 prefills.
Axis C (1.5B only, Bundle 2) ≈ 3 sub-bands × 2 arms × 101 concepts at the best α ≈ ~600
forwards. Total Bundle 2 is a few hours on MPS, **$0**, no fit, no rented GPU. Serial-GPU
discipline unchanged: no S1 run shares the machine with anything else.

## What S1 does NOT decide

- Generalization or selectivity (the two KICKOFF stretch milestones) — separate scope,
  separate briefs, still gated on Kyle.
- Any new model, fit, band, or intervention operator — S1 reuses M0/M1 artifacts wholesale.
- Framing stays **descriptive** (triple readability NULL; the pre-registered re-scope
  holds — S1 characterizes the flagship finding, it does not reclassify it as a
  reproduction claim).
- Nothing in this PR is harness code; the brief precedes the build (per-stage rhythm).

## Frozen decisions

*Frozen 2026-07-16 (Kyle) — Bundle 2 with all three convention defaults accepted;
recorded here and in `DECISIONS.md`. Relitigating after this is a deviation row, not a
conversation.*

- **D15 = Bundle 2** — "localize it": axes A (saturation) + B (`J=I` falsification arm) +
  C (layer localization). A+B on all three subjects; C on 1.5B only.
- **D16 = A** — `J = I` falsification arm: steer along the unit-normalized raw
  unembedding row `u_t`, same per-layer mean-norm × α scaling, same band layers and
  question-turn positions; Newcombe 95% CI on (J-lens − J=I) at each α. Project-consistent
  Arm-2 convention (owned deviation from the paper's orthogonal-component contrast).
- **D17 = A** — extended α grid: `{0, 0.5, 1, 2, 4, 8}` (frozen D8) + appended
  `{12, 16, 24}`. Degeneracy guard: flag the first α at which a subject's reply
  degenerates (steered token dominating regardless of concept).
- **D18 = A** — localization granularity: three contiguous sub-bands of L11–L24 steered
  separately — **early L11–15, middle L16–20, late L21–24** — at the best-reporting α (and
  its `J=I` arm), 101 concepts each, 1.5B only.
