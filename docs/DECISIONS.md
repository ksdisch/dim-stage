# DECISIONS — dim-stage

*Append-only log. Kickoff-interview decisions live in `KICKOFF.md` ("Decisions
on record"); this file starts at M0. Each entry: what was decided, by whom,
and why — so any choice can be defended later.*

## 2026-07-15 — M0

**D1 (Kyle) — AGREE-gate metric.** Per-layer relative Frobenius distance
decides (tolerance = 10× the reference's measured run-to-run noise floor);
top-1 readout agreement (Wilson LB ≥ 0.95) confirms; lens-eval pass@k is
reported but never gates. *Why:* most sensitive divergence detector; localizes
failure to a layer. *Outcome:* AGREE — distance exactly 0 (bitwise-identical);
noise floor exactly 0 (deterministic MPS backward), so the pre-declared 1e-4
stand-in did the work.

**D2 (Kyle) — Band selection.** Proportional transplant of the paper's 38–92%
depth band, pre-registered as primary (0.5B: L9–L21; 1.5B: L11–L24);
kurtosis/next-token diagnostics descriptive only. *Why:* zero forking-paths
freedom on the gate.

**D3 (Kyle) — Fit corpus.** WikiText-103 via the reference's
`load_wikitext_prompts` convention, N=100. *Why:* deterministic, byte-identical
prompts for both implementations; owned deviation from "generic web text."

**D4 (Kyle) — Readability gate.** Two arms: (1) READS iff J-lens pass@10
Wilson 95% LB ≥ 0.5 on ≥3/6 distributions, else NULL; (2) J-advantage per
distribution via Newcombe CI against the `J = I` logit-lens arm. *Why:*
separates "workspace exists" from "Jacobian correction adds value."
*Outcome:* NULL on both subjects (0/6 each); J-advantage 2/6 at 0.5B, 0/6 at
1.5B with typo significantly reversed.

**Model variants (assumption, unobjected).** Instruct variants of both
subjects, because M1–M3 stimuli are chat-format and the lens must be fit on
the model being read.

**dim_batch frozen at 8 on MPS (Claude, measured).** dim_batch=32 is
math-neutral but ~25× slower (replicated retained graph blows unified-memory
locality). Recorded in the hour-one gate result.

**Intermediate→token grading convention (Claude, pre-declared before any
result).** The released material ships no eval-grading code, so the convention
had to be ours: rank = min over the single-token encodings of {`w`, `␣w`};
order-ops synonym table fixed in code (numbers → digit + word form;
operations → word + symbol + spoken form); intermediates with no single-token
form are dropped and counted (deviations row 5/6). *Why pre-declared:* every
added synonym monotonically helps pass@k — choosing them after seeing results
would be forking paths.

**Grading device = MPS for both subjects (Claude).** No mixed-device numerics
between subjects; CPU work was measured slowing the concurrent MPS fit ~8%,
so measurements never share the GPU with a fit.

**Serial, not concurrent, GPU jobs (Claude, ratified by Kyle's afternoon
deadline).** The 1.5B fit and all gate runs were queued strictly
back-to-back: MPS interleaves rather than parallelizes, and memory pressure
is the repo's documented 25× failure mode.
