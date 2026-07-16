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

## 2026-07-15 — 3B escalation (post-M0)

**3B escalation (Kyle) — option (c): overnight 3B fit + M1 brief in
parallel.** The pre-registered trigger (KICKOFF decision 2: escalate only if
BOTH subjects null on readability) fired with M0's double NULL. Kyle chose
(c) over (a) fit-only and (b) descriptive-M1-only. *Why:* a third scale point
brackets the emergence question on Arm 1 and extends the J-advantage
inversion trend on Arm 2 — informative even if NULL again; M1's protocol
extraction is needed on every branch (the brief is written branch-proof:
measured-on-3B if it READS, descriptive otherwise); the measured ~8% MPS
slowdown from concurrent docs-only work is free overnight.

**3B band frozen before the run (mechanical application of D2).**
`proportional_band(36)` = L14–L32 added to `FROZEN_BANDS` (with a
table-equals-rule test) and merged to main before the fit produced any
readout — pre-registration, zero new degrees of freedom. Subject is
**Qwen2.5-3B-Instruct** (the standing Instruct-variant assumption); D3
corpus (N=100 WikiText), `dim_batch=8`, fp32-on-MPS, and the D4 two-arm gate
all carry over frozen and unchanged.

**Probe-as-production + sleep guard (Claude).** The machine has 24 GB
unified memory; 3B fp32 is ~12.4 GB of weights plus the retained backward
graph — edge-of-feasible, and the documented failure mode is the ~25×
memory-pressure cliff. So the production fit doubles as the probe: the
first two prompts' wall-clock is watched against the ~560 s/prompt
extrapolation (measured 1.5B rate × params ratio × 256/192 backward-pass
ratio; ≈15.5 h for N=100). Fallback ladder, pre-declared: kill → resume
from checkpoint at `dim_batch=4` (math-neutral; the checkpoint stores no
dim_batch) → if still pressured, stop and surface the ~$1 rented-GPU
fallback (amended bar entry 2) to Kyle before any spend. Overnight runs are
wrapped in `caffeinate -ims` so the Mac cannot idle-sleep mid-fit; laptop
stays plugged in.

**Probe outcome — the ladder ran to its end (Claude, measured 2026-07-15
evening).** Three findings, in order. (1) *HF streaming wedge:* both first
launches stalled in `load_wikitext_prompts` — huggingface `datasets`
streaming sat in rate-limit backoff (three TCP connections, 0 bytes moved in
15 s, while fresh hub requests answered in 0.17 s). Fixed by prefetching the
D3 prompts once via the non-streaming cached download + a `--prompts-file`
flag on the fitter (PR #7; transport only, same bytes). Honesty note: the
first dim_batch=8 "probe fail" reading was partly confounded by this wedge —
though a call-stack sample did catch it genuinely grinding in MPS compute
30+ min into prompt 1, so the kill stands on either reading. (2) *The cliff
is real at both rungs:* with prompts local, prompt 1 exceeded 40 min of pure
MPS compute at dim_batch=8 AND dim_batch=4 (~11 min healthy estimate; ~25×
class; ~70 h/fit extrapolated; stack samples show `MPSStream` copy-syncs;
no OS pressure alarm, no thermal warning, no checkpoint ever completed).
Halving the graph didn't help because the constant term — 12.4 GB of fp32
weights — sits at the Metal working-set edge by itself. Lesson for the
memory file: **24 GB MPS tops out between 1.5B and 3B for fp32
backward-pass work; probe before committing, and don't expect dim_batch to
rescue a weights-bound cliff.** (3) *No local knob remains:* dim_batch=2
shaves ~1.6 GB against a weights-dominated budget; fp16 would change the
frozen fp32 numerics convention — a bigger deviation than changing device.

## 2026-07-15 — 3B rescue

**3B fit moves to a rented CUDA GPU (Kyle).** The pre-declared fallback,
now invoked: fit on a rented single-GPU box (~$1–3 total), procedure
byte-identical (same `fitter.py`, same frozen prompts file, `dim_batch=8`,
fp32) — `remote-fit-3b.sh` is the one-shot script; the fitter now
auto-selects CUDA. The readability gate still runs **locally on MPS**, like
both other subjects, auto-queued on the lens file's arrival. *Planned
deviation rows (into M0-BRIEF with the results):* (a) 3B lens fitted on
CUDA fp32 vs MPS for 0.5B/1.5B — cross-device fp32 noise is ~1e-7 relative
on a corpus-mean matrix, orders below any gate's sensitivity; the box's
torch version is recorded in the fit log; (b) fit corpus delivered via
`--prompts-file` (PR #7) rather than live streaming — same bytes by
construction, and the run log's `seq_len`/`n_valid` signature per prompt
must match the M0-recorded values (prompt 1: `seq_len=128 n_valid=111`;
Qwen2.5 sizes share one tokenizer).

**3B rescue — executed and closed (Claude, 2026-07-15 evening).** Rented an
RTX 4090 (RunPod, CUDA fp32, torch 2.8.0+cu128, transformers 5.14.0). Fit:
~57 min at **34.5 s/prompt** — the same prompt the Mac could not finish in
40 min, a ~70× speedup that makes the Metal cliff a measured fact. Both
planned deviation checks passed: the fit log's per-prompt `seq_len=128
n_valid=111` matched the MPS runs (corpus identity), and the lens transferred
sha256-identical (verified before the atomic move into `lenses/`). Gate ran
locally on MPS → **NULL (0/6)**. Total spend ~$0.83; pod terminated and its
50 GB network volume deleted (the one lingering-charge risk, closed). Both
deviation rows landed as M0-BRIEF rows 7–8.

**Console label fix — J-REVERSAL made visible (Claude).** The gate's Arm-2
console line printed "no clear gap" for a CI-clean *negative* difference,
conflating a genuine no-gap with a J-transport reversal (e.g. 3B typo
−.323 [−.442, −.188]). Added a three-way label (`J-ADVANTAGE` / `J-REVERSAL`
/ `no clear gap`) so the deterministic oracle's own output is faithful. The
`j_advantage` boolean and the stored JSON are unchanged (Arm 2 never gates,
and the hand-authored M0-BRIEF tables already distinguished "reversed"); this
only corrects what the console prints. Console-only; no re-run needed.

**Verdict on the escalation (record).** KICKOFF decision 2 is fully
discharged: trigger met (double null) → escalated → 3B fitted at the frozen
band → **NULL**. Three scales (0.5B, 1.5B, 3B), pre-registered bar,
deterministic oracle, structured null. No emergence point exists in the
reachable range; M1–M3 are descriptive on all three subjects.

## 2026-07-16 — M1

**D5 (Kyle) — Trial-set and readout conventions.** The reference-faithful
package: chat-template prompts (`apply_chat_template`, generation prompt
appended); readout at the final prompt token (M0 precedent, the analog of
the paper's `Assistant:` colon); swap-out = the greedy next token there;
swap-in trials = first 10 listed candidates per category, skipping the
spontaneous answer, graded only where the candidate's baseline rank ≥ 11
(the anchor's exclusion); candidates must pass the M0 single-token filter;
steering scale = per-layer mean L2 residual norm over the frozen D3 fit
corpus. *Why:* every choice is cited or inherits an M0-frozen convention —
like-for-like with the paper's 88% top-5 anchor; the trade-off (N shrinks
below the maximal 140) is owned.

**D6 (Kyle) — Correctness gate = pre-committed invariants.** No reference
intervention code exists (verified 2026-07-15), so no AGREE diff is
possible. The gate, merged before any real run (`test_intervention.py`):
(1) rigged-subject analytic oracle — a tiny model with an exactly-known
Jacobian where every post-patch logit is hand-computed and asserted with
exact equality; (2) coordinate read-back — c′ = σ(c) and the component
orthogonal to span{v_s, v_t} unchanged; (3) null-ops — α=0 steering and
s=t swaps change nothing, exactly; plus a guard that the implementation
equals the paper's literal V(σ(c)−c) form on random tensors. The
measurement runner will repeat the read-back as a runtime self-check on
real subjects. *Why:* machine-checkable and pre-committed; catches the
sign/transpose/normalization bugs that would silently fake a null.

**D7 (Kyle) — Two-arm verdict, wording frozen before any result.** Arm 1
(would gate in measured mode): per eligible trial, hit iff the swapped-in
candidate reaches **top-5** at the readout position; Wilson 95% LB ≥ 0.5
pooled over eligible trials ⇒ "verbal report tracks the workspace". Arm 2
(never gates): every swap repeated with raw unembedding rows (J = I) — the
standing falsification arm; Newcombe 95% CI on the difference, testing
whether M0's J-transport reversal holds for *writing*, not just reading.
Rank-1 and top-10 reported descriptively. All three subjects are NULL on
readability, so M1 runs entirely descriptive: identical numbers, the
pre-declared characterization framing, no property claims either way.
*Why:* mirrors D4; top-5 is the anchor's own grading.

**D8 (Kyle) — Introspection arm in, descriptive-first.** Protocol 2 runs on
every subject at strengths α ∈ {0, 0.5, 1, 2, 4, 8} (mean-residual-norm
units; 0 = the README's control); `default` prefill primary, `word`
descriptive; MRR-vs-strength and report rate with Wilson CIs; never gates
M1's verdict. *Why:* the steering operator is required for M3 anyway —
validating it under D6 now is free leverage; the strength grid is an owned
convention (the paper sweeps but doesn't publish its grid).

**v_t construction (Claude, pre-declared before any run).** `v_t = J_lᵀu_t`
with `u_t` = the raw `lm_head.weight` row — the literal reading of the
paper's "row of the unembedding matrix", consistent with the README's
"transpose row" (a row of U·J_l). Qwen's final RMSNorm has an elementwise
scale γ sitting between the residual and that matrix; it is NOT folded into
u_t — folding it is an uncited variant. M1-BRIEF deviations row 7.

**Swap implemented through the identity (c_t − c_s)(v_s − v_t) (Claude).**
Algebraically equal to the paper's `V(σ(c) − c)` (σ(c) − c = (c_t − c_s)·
[1, −1]) and asserted against the literal pseudoinverse form on random
tensors in the D6 gate; it makes s = t an exact no-op (v_s − v_t is exactly
zero) instead of a degenerate solve. Coordinates come from the closed-form
2×2 normal equations — exact for two columns, and free of `torch.linalg`
kernels MPS doesn't fully cover; (near-)parallel direction pairs are
rejected loudly rather than solved badly.

**Protocol conventions frozen in code before their runs (Claude).** Verbal
report: swap token = the candidate's bare single-token form (the shape an
answer takes right after the chat template's trailing newline); grading =
min over both {`w`, `␣w`} forms (M0's convention); "skipping the answer
itself" = skip a first-10 candidate iff the greedy answer token is among its
forms. Introspection: steering scale = per-layer mean L2 residual norm over
the D3 corpus **at the fit's own valid positions** (skip_first=16, final
excluded); question-turn positions = the third `intro_prompt` message's
whole templated block, located via the template's tokenized-prefix property
(asserted at runtime, INVALID otherwise); α = 0 control computed once per
prefill (steering by zero is a D6-tested exact no-op).

**M1 outcomes (record, 2026-07-16).** Both protocols ran on all three
subjects, descriptive throughout (triple readability NULL). *Verbal report
(swap):* pooled top-5 **.175 [.112, .263] / .124 [.070, .208] /
.105 [.056, .187]** at 0.5B/1.5B/3B vs the paper's .88 Claude anchor — the
report does not follow the swap; Arm-2 J − I Newcombe CIs all straddle zero
(writing mirrors M0's reading: no measurable J-transport advantage; at
rank-1 the raw rows beat the J-lens vectors on every subject). *Verbal
introspection (steer):* report rate at α = 8 (default prefill) **0/101,
30/101 [.217, .392], 5/101 [.021, .111]** — a monotone dose–response at
1.5B with the α = 0 control exactly 0/101 on every subject; the 1.5B–3B gap
is CI-clean, the strongest descriptive contrast in the project; median
steered rank moves 3747→1322 / 4430→15 / 3791→382 (control → α = 8). The D6
runtime read-back stayed silent across every swap application; every
wrong-arm dry-run exited INVALID as designed. M1 closed.
