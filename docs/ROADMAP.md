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

## Decision now on the table (Kyle's call — pre-registered trigger met)

**3B escalation.** KICKOFF decision 2: "escalate to 3B only if BOTH subjects
null on readability." Both are. Options at next session start: (a) escalate to
Qwen2.5-3B-Instruct (fit ~12–14 h overnight at measured rates — bracket the
emergence point); (b) proceed descriptive M1 on current subjects; (c) both.

## M1 — Verbal report — not started (re-scoped descriptive pending 3B call)
## M2 — Two-hop swap — not started (re-scoped descriptive pending 3B call)
## M3 — Directed modulation — not started (re-scoped descriptive pending 3B call)
## Stretch (gated behind v1 close) — generalization, selectivity
