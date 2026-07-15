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
| Readability gate, both subjects | **NULL / NULL** — 0/6 distributions at Wilson LB ≥ 0.5 (pass@10) on either subject; full tables in `M0-BRIEF.md`, JSONs in `results/` |
| Single-token stimulus pre-filter | Built into `readability.py`; 94–100% of intermediates survive; every cell N ≥ 94 |

**Headline:** the pre-declared kill-risk fired. At 0.5B/1.5B the workspace is
not readable at the frozen bar — the paper's open question, answered for the
smallest scales, with structure: abstract-content distributions (association,
poetry) are hard zeros; surface-adjacent ones sit at 33–54%; the J-advantage
that exists at 0.5B (typo +43pp, multilingual +21pp) vanishes or reverses at
1.5B (typo −27pp, CI-clean) as the plain logit lens catches up.

## Consequences for M1–M3 (pre-declared in KICKOFF, risk #1)

The double null **re-scopes M1–M3 to descriptive**: swap/report/modulation
protocols can still be run and measured, but their gates were premised on a
readable workspace; verdicts become descriptive characterization rather than
reproduction claims. Each still opens with its own start-of-stage brief.

## 3B escalation — DECIDED 2026-07-15 (Kyle: option c) — fit in flight

KICKOFF decision 2's trigger fired (both subjects null) and Kyle chose (c):
fit **Qwen2.5-3B-Instruct** overnight while M1's start-of-stage brief is
written in parallel. Band L14–L32 (proportional rule, 36 layers) frozen and
merged **before** the run; readability gate queued behind the fit (~15.5 h at
measured rates → verdict expected 2026-07-16 AM). If 3B READS, M1–M3 run
measured on 3B; otherwise they stay descriptive on 0.5B/1.5B.

## M1 — Verbal report — start-of-stage brief on file (`M1-BRIEF.md`)

Decisions D5–D8 pending Kyle; the intervention build starts after they freeze.
Measurement mode (measured-on-3B vs all-descriptive) resolves with the 3B
verdict. Design extraction done 2026-07-15: swap + steering operators verbatim
from the paper; **the reference ships no intervention code**, so M1's
correctness gate is pre-committed invariants (D6), not an AGREE diff.
## M2 — Two-hop swap — not started (scope resolves with the 3B verdict)
## M3 — Directed modulation — not started (scope resolves with the 3B verdict)
## Stretch (gated behind v1 close) — generalization, selectivity
