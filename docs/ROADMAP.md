# ROADMAP — dim-stage

*Milestone status against the approved plan in `KICKOFF.md`. Updated at each
stage close (per-stage rhythm, CLAUDE.md).*

## M0 — Fit pilot: does the lens fit, agree, and read? — **COMPLETE 2026-07-15**

All six deliverables closed in three PRs (#2 hour-one gate, #3 fitter + AGREE,
#4 readability gate + spine):

| Deliverable | Outcome |
|---|---|
| Hour-one gate (MPS backward, wall-clock) | **PASS** — 42.7 s/prompt on 0.5B fp32; both fits 7.55 h extrapolated ≤ 12 h (actual: 0.5B 71 min, 1.5B ~6.1 h) |
| Design extraction (estimator, band, corpus, grading) | In `M0-BRIEF.md`; conventions frozen before results |
| Decisions D1–D4 frozen | See `DECISIONS.md` |
| Independent build → AGREE gate | **AGREE** — bitwise-identical `J_l` at all layers (rel-Frobenius 0.0 vs 1e-3 tolerance); readout 3220/3220, Wilson LB .9988 |
| Readability gate, all three subjects | **NULL / NULL / NULL** — 0/6 distributions at Wilson LB ≥ 0.5 (pass@10) on 0.5B, 1.5B, or 3B; full tables in `M0-BRIEF.md`, JSONs in `results/` |
| Single-token stimulus pre-filter | Built into `readability.py`; 94–100% of intermediates survive; every cell N ≥ 94 |

**Headline:** the pre-declared kill-risk fired and held under escalation. Across
0.5B, 1.5B, and 3B the workspace is not readable at the frozen bar — the paper's
open question, answered for Qwen2.5 0.5B–3B with a clean, three-scale,
pre-registered *no*. Structure: the abstract-content distributions (association,
poetry) are hard zeros at all three scales; surface-adjacent content is partially
readable but sub-bar and non-monotone (multihop peaks at 1.5B 54% then regresses
to 39% at 3B; order-ops is the closest cell anywhere at 3B 46%, Wilson LB .368);
the J-advantage is content-dependent, not a scale trend — typo's J-transport
reversal deepens monotonically (+43pp → −27pp → −32pp) while order-ops *regains* a
CI-clean J-advantage at 3B (+16pp).

## Consequences for M1–M3 (pre-declared in KICKOFF, risk #1)

The triple null **re-scopes M1–M3 to descriptive**: swap/report/modulation
protocols can still be run and measured, but their gates were premised on a
readable workspace; verdicts become descriptive characterization rather than
reproduction claims. Each still opens with its own start-of-stage brief.

## 3B escalation — CLOSED 2026-07-15 (Kyle: option c) — executed, NULL

KICKOFF decision 2's trigger fired (both subjects null) and Kyle chose (c):
fit **Qwen2.5-3B-Instruct** while M1's start-of-stage brief was written in
parallel. Band L14–L32 (proportional rule, 36 layers) frozen and merged
**before** the run. Local MPS proved infeasible for 3B fp32 (the ~25×
working-set cliff at every dim_batch — probe story in `DECISIONS.md`), so
per the pre-declared fallback the fit ran on a **rented RTX 4090** (CUDA
fp32, ~57 min at 34.5 s/prompt, ~$0.83 total; `remote-fit-3b.sh`); the
readability gate ran locally on MPS on the returned lens. **Verdict: NULL
(0/6)** — 3B does not READ either, so M1–M3 stay descriptive on all three
subjects. The escalation is closed: the trigger was honoured, the strongest
finding this project could produce (an emergence point) was searched for at
the one extra scale we can reach, and it is not there. Full 3B table +
three-scale structure in `M0-BRIEF.md`.

## M1 — Verbal report — **COMPLETE 2026-07-16** (descriptive; triple-null re-scope)

All deliverables closed in three PRs (#10 operators + D6 gate + D5–D8 freeze,
#11 verbal-report runner + results, #12 introspection + stage close):

| Deliverable | Outcome |
|---|---|
| D5–D8 frozen | 2026-07-16, Kyle — all four recommendations (`DECISIONS.md`) |
| Intervention module + D6 gate | `intervention.py` + pre-committed invariants (rigged analytic oracle, exact equality; the reference ships **no** intervention code, so invariants replace the AGREE diff); runtime read-back silent across every real swap |
| Verbal report (swap), all subjects | Top-5 **.175 / .124 / .105** (0.5B/1.5B/3B) vs the paper's **.88** Claude anchor — the report does not follow the swap; Arm-2 J − I Newcombe CIs all straddle zero |
| Verbal introspection (steer), all subjects | **The project's first dose–response curve:** 1.5B report rate rises 0 → **30/101 (.297 [.217, .392])** with strength, α = 0 control exactly 0/101 everywhere; 0.5B flat zero; 3B only 5/101 — the 1.5B–3B gap is CI-clean |

**Headline:** writing the workspace mostly fails to move the spoken report at
these scales — except that a *steered-in* thought becomes reportable at 1.5B
specifically, with a clean control and a monotone strength curve. The
J-transport adds no CI-clean value for writing anywhere (M0's Arm-2 story,
repeated). Non-monotone scale structure again: 1.5B, not 3B, is where the
phenomenon lives. All framing descriptive per the pre-registered re-scope.
## M2 — Two-hop swap — not started (descriptive, per the triple null)
## M3 — Directed modulation — not started (descriptive, per the triple null)
## Stretch (gated behind v1 close) — generalization, selectivity
