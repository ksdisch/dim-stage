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

## 2026-07-16 — M2

**D9 (Kyle) — Baseline-conditioned primary cell.** All 90 shipped items,
raw-text encoding (M0's lens-eval treatment), readout at the final prompt
token; baseline passes iff the unswapped greedy token is among the `answer`'s
single-token forms — strict {`w`, `␣w`} of the shipped string, **no synonym
table** (the README's "greedy next-token == answer", verbatim; pre-declared
before any run since every synonym would monotonically raise rates). Primary
swap cell = baseline-passing items; baseline accuracy and unconditioned
rates reported alongside; n < 20 pre-declared UNDERPOWERED. *Why:* a "flip"
needs a working chain to redirect; Claude's near-perfect baseline made the
two readings coincide for the paper — ours diverge.

**D10 (Kyle) — Top-1 flip verdict, two arms, wording frozen.** Success = the
swapped greedy token is among `swap_answer`'s forms (the anchor's own
grading); Wilson 95% on the primary cell, would-gate wording LB ≥ 0.5,
descriptive framing throughout. Descriptive extras: swap_answer top-5/10,
displaced-original rate, per-category breakdown. Arm 2 = raw rows (J = I),
Newcombe — the standing falsification arm. *Why:* like-for-like with the
60%/54–70% anchors, which are top-1 fractions.

**D11 (Kyle) — Answer-swap comparison arm, descriptive-only.** Every trial
repeated swapping `answer → swap_answer` instead of the intermediates, both
arms, same frozen band (~2 extra forwards/item). *Why:* the paper's named
smuggling confound, addressed at our scale without its out-of-scope depth
sweep (M2-BRIEF deviations row 4).

**M2 outcomes (record, 2026-07-16).** 9/90 items dropped by the four-field
single-token pre-filter (same 9 on every subject — shared tokenizer); no
primary cell below N = 20. Baselines 28/81, 41/81, 43/81. Intermediate-swap
flips (primary): **8/28 = .286 [.153, .471] / 3/41 = .073 [.025, .194] /
5/43 = .116 [.051, .245]** vs the .60 anchor. Identity arm: 4/28, **0/41,
0/43** — at 3B the J − I Newcombe CI **excludes zero (+.116 [+.011, +.245])**,
the project's first CI-clean J-transport advantage for writing (Arm 2 never
gates; descriptive contrast). Answer-swap (D11): 16/28, 9/41, 6/43 — double
the intermediate rate at 0.5B, equal at 3B; with the raw-row zeros, no
answer-smuggling signature at 3B. Displaced-original ~40% everywhere (14/28,
12/41, 18/43); swap_answer reaches top-5 on about half of primary trials.
The D6 read-back stayed silent; wrong-arm dry-run exited INVALID. M2 closed
same-day: brief → freeze → runner → three subjects → spine.

## 2026-07-16 — M3

**Correction on the record (Claude).** M1-D8's rationale claimed the steering
operator was "required for M3." Wrong: M3 is a **reading** milestone — the
modulation is the instruction in the prompt; the lens only reads. D8's other
merits stand (KICKOFF scoped verbal-introspection in; it produced the 1.5B
dose–response), but that rationale line is retracted. Owned in M3-BRIEF.

**D12 (Kyle) — Two shipped families.** Category-instance + math-expression
verbatim from `directed-modulation.json`, plus the constructed
no-instruction baseline; the line-width family is out (stimuli unshipped,
different metric, would need an owned RNG — KICKOFF's shipped-stimuli scope).
*Why:* reproduce what ships; own what doesn't as a deviation row, not a
reconstruction.

**D13 (Kyle) — Deterministic full grid + owned prompt frame.** Every
phrasing × every target (1,104 trials), carrier rotated
`carriers[(i + j) mod 20]`; baseline = every target × `carriers[j mod 20]`
(46). User turn = filled phrasing + a space + `Copy this sentence exactly:
"<carrier>"` (the reference ships no frame — owned); assistant teacher-forced
to the carrier verbatim; readout span = the carrier's tokens; hit = a tracked
token is the J-lens **top-1 at any (layer, position)** (Figure 10, verbatim);
tracked tokens = single-token {`w`, `␣w`} forms, zero-form targets dropped
and counted. No RNG anywhere. *Why:* trials are wall-clock-free at this
size; determinism beats sampling conventions.

**D14 (Kyle) — Contrast-first two-arm verdict, wording frozen.** Cells per
family × condition (focus/control/suppress/baseline) × arm (J-lens /
logit-lens J = I), Wilson 95%. Would-gate: "modulates" iff focus − suppress
Newcombe excludes 0 (positive) on BOTH families AND the pooled baseline is
clean — Wilson UB ≤ 0.10, **pooled across families (n = 46)**: a zero-hit
n = 22–24 cell mathematically cannot get its UB under .10, so the per-family
bar would be unreachable; per-family baselines are reported alongside
(implementation note, pre-declared before any run). Also reported:
suppress − baseline (the white-bear check), focus − control (instruction vs
mere mention), the dismissal vs negated-think split (Figure 65), and
J − logit per condition (the standing falsification arm). Descriptive
framing throughout (triple readability NULL). *Why:* the phenomenon is the
contrast; no numeric absolute anchor exists in the paper's text.

**M3 outcomes (record, 2026-07-16).** 1,150 trials per subject, no targets
dropped. The would-gate reads **"does not modulate" on all three subjects**
(math is a hard zero everywhere; the gate needs both families). The
KICKOFF-cited anchor reproduces: pooled no-instruction baseline 0/46
(UB .077 ≤ .10) on every subject, both arms. Category family, J-lens focus:
2/110 → 6/110 → 9/110 with control ≈ suppress ≈ 0 — focus − suppress and
focus − control both CI-clean at 1.5B (+.055) and 3B (+.082), and the
scale growth 0.5B → 3B is CI-clean (+.064 [+.004, +.131]). Logit-lens arm:
3B focus **19/110** (control 4/132, suppress 0/286) → **J − logit −.091
[−.181, −.002], a CI-clean J-transport reversal** — directed modulation
exists at 3B and is read better *without* the transport (M0's typo
reversal, now on instructed content). No mention-priming (control 0–4/132
everywhere, vs the paper's mention ≈ focus) and no measurable white-bear
(suppress ≈ baseline ≈ 0: nothing enters uninstructed, so suppression has
nothing to fail at). M3 closed same-day: brief → freeze → runner → three
subjects → spine. **v1 core-three measurement is complete.**

## 2026-07-16 — S1 (stretch: introspection dose–response follow-up)

*First stretch stage, opened after v1 close (Kyle's pick). Deepens M1's
flagship finding — the 1.5B injected-thought dose–response (0→30/101, clean
α=0 control). Full options + design extraction in `docs/S1-BRIEF.md`. No new
model, fit, band, or intervention operator: reuses M0/M1 artifacts. Framing
stays descriptive (triple readability NULL holds).*

**D15 (Kyle) — Bundle 2 ("localize it").** Scope = axis A (saturation) +
axis B (`J=I` falsification arm) + axis C (layer localization). A+B on all
three subjects; C on 1.5B only (0.5B has no signal to localize, 3B's 5/101
too thin). *Why:* B is the one falsification arm the strongest finding never
got; A is nearly free; C is where a novel internals result and the deep-dive
learning live. Bundle 1 (A+B only) was the honest defensibility floor;
Bundle 3 (+ position + selectivity) deferred as diminishing returns.

**D16 (Kyle) — `J = I` falsification convention.** Steer along the
unit-normalized **raw unembedding row** `u_t` (the logit-lens direction, no
Jacobian transport), scaled by the same per-layer mean residual norm × α, at
the same band layers and question-turn positions; Newcombe 95% CI on
(J-lens − J=I) report rate at each α. *Why:* identical Arm-2 substitution as
M1/M2/M3 — like-for-like across the project; reuses the operator unchanged.
Owned deviation from the paper's orthogonal non-J-space *component* contrast
(deviations row 2).

**D17 (Kyle) — Extended α grid + degeneracy guard.** Keep the frozen D8 grid
`{0, 0.5, 1, 2, 4, 8}` and append `{12, 16, 24}` (~3× headroom past the
current top, where the 1.5B curve was still rising). Degeneracy guard: flag
the first α at which a subject's reply degenerates (the steered token
dominating regardless of concept), so saturation isn't confused with
model collapse. *Why:* enough headroom to see the Figure-7 rise-then-saturate
shape without unbounded compute; the grid past 8 is an owned convention (the
paper sweeps but publishes no grid — same footing as D8).

**D18 (Kyle) — Sub-band-thirds localization.** Steer three contiguous
sub-bands of L11–L24 separately — **early L11–15, middle L16–20, late
L21–24** — at the best-reporting α (and its `J=I` arm), 101 concepts each,
1.5B only. *Why:* directly tests the paper's "middle block" workspace claim
at hobby scale; 3 configs is cheap and legible (a clean bar chart). Full
single-layer resolution deferred unless a sub-band clearly dominates and the
exact layer matters.

**S1 outcomes (record, 2026-07-16).** All 101 concepts single-token-valid (0
dropped); no subject collapsed at any α (degeneracy guard silent — the
extended grid broke no model). Owned deviation: `default` prefill only (M1
showed `word` tracks it; deviations row 5). **B (falsification):** the 1.5B
dose–response is CI-cleanly a J-transport effect — J-lens − J=I CI-clean from
α=1, peak **+.178 [+.067, +.286]** at α=8; J-lens ~2× the raw-unembedding arm
(30–31/101 vs 12–14). The flagship finding's one missing arm is closed in the
favorable direction — the project's first CI-clean J-transport advantage for
*report*, matching the paper's specificity control. **A (saturation):** 1.5B
J-lens plateaus ~30/101 from α=8 (30/30/29/31 across 8→24), MRR still climbing
(.067→.125) — the paper's Figure-7 shape. **C (localization):** best
non-collapsed α=24; the **middle third L16–20 alone recovers 29/101 ≈ the
full band's 31** (full−mid +.020 [−.105, +.144], overlaps 0) — the paper's
mid-layer "middle block"; J-transport advantage CI-clean mid (+.129
[+.014, +.240]) and late (+.129 [+.041, +.219]), overlapping early (+.040).
**Cross-scale bonus:** the extended grid exposed 3B reporting as *purely*
J-transport (identity 0–1/101, J-lens →9/101, J−I CI-clean from α=8:
+.050 [+.003, +.111] up to +.089 [+.034, +.161]); 0.5B null both arms
(J-lens immovable, attractor 1.00). S1 closed same-day: brief → freeze →
runner → three subjects → spine.

## S2 — flexible generalization (stretch): D19–D22 frozen 2026-07-16 (Kyle)

*All four recommendations accepted (S2-BRIEF menu). Scope: the paper's
16-template lens-coordinate swap experiment, verbatim item set from the
reference repo's `flexible-generalization.json`. No new model, fit, or band;
the one operator change is an α parameter on the existing swap, pinned by the
involution argument (S2-BRIEF, "Design extraction") and invariant-gated.
Framing stays descriptive (triple readability NULL holds).*

**D19 (Kyle) — Verbatim item set + standing single-token filter.** Run the
reference set as shipped; skip and count the 12 trials whose target answers
have no single-token form (all in animals — savanna, arachnid, convocation,
shiver; measured 2026-07-16 on the shared Qwen2.5 tokenizer). Pooled cell is
**180 gradable trials** (countries 48, months 48, animals 36, numbers 48),
reported beside the 192-trial anchor. *Why:* zero invented items — the same
M0 pre-filter every prior milestone owned; substitution or dropping animals
would trade verbatim extraction for tidiness.

**D20 (Kyle) — Two arms, J-lens + `J = I`.** Every trial in both arms at
every α: the J-lens coordinate swap and the standing falsification arm (same
exchange with directions = raw unembedding rows); Newcombe 95% CI on (J − I)
per α. *Why:* like-for-like with M2's Arm 2, where raw rows flipped nothing
at 3B — if S2's swaps work only through the transport, this arm shows it.

**D21 (Kyle) — α ∈ {1, 2, 4, 8} + degeneracy guard.** The two anchored
points (76/192, 101/192) plus two doublings of headroom; guard adapted to
greedy readouts: flag any α × arm cell where a single token is top-1 on
≥ half of all trials (S1's `COLLAPSE_SHARE` convention — with 16 templates
and ~60 distinct expected answers, no honest cell is half one token).
*Why:* distinguishes "underdosed" (the paper's own reading of α=1 failures)
from "no effect"; {4, 8} are owned convention rows, same footing as D17.

**D22 (Kyle) — Three readouts from one run.** (i) **Unconditional** pooled +
per-category success (the anchor frame); (ii) **baseline-conditioned** —
trials where the model got both the source prompt and the target's own
diagonal right unswapped (it provably knows f(source) and f(target); M2's
conditioning lesson); (iii) **workspace loading** per argument (band-layer
mean cosine between the unmodified residual and the argument's lens vector
at argument + readout positions) against per-argument swap success — the
paper's own Fig-19-right predictor. *Why:* anchor comparability, small-scale
honesty, and the paper's predictor, at no extra forwards.

**Convention note (pre-committed with the freeze, before any run):** the
would-gate wording reuses M2's frozen floor — "routes" iff the α=2 pooled
J-lens Wilson LB ≥ 0.5 (the anchor's own rate is .526, so this is the
paper-level bar) — descriptive mode applies throughout. Swap rows take the
prompt-position token form (the argument appears *in* these prompts, unlike
M2's latent intermediates; owned departure from M2's bare-form default,
deviations row 3). Per-template cells (N = 12) are pre-declared UNDERPOWERED
texture.

**S2 outcomes (record, 2026-07-16).** Plan exactly as frozen: 180 gradable /
12 filtered; 6 variant fallbacks, all `animals/group` (the template pluralizes
`{arg}s`, so the argument token never appears — falls back to M2's
bare-preferred rows; owned in S2-BRIEF deviations row 3). The generalized D6
read-back silent on every subject at every α; the degeneracy guard silent
everywhere except **3B α=8**, where both arms collapse to a `!` attractor at
share 1.00 (its first real catch — true junk collapse, with logits saturating
into mass rank-1 ties; success grading is greedy-membership and unaffected).
**Would-gate: "does not route" on all three subjects** (α=2 J-lens LB vs the
frozen .5 floor: all ≈ 0). The structure inside: **a CI-clean J-transport
routing signal at α=1 on all three subjects** (0.5B +.078 [+.031, +.131],
1.5B +.061 [+.012, +.114], 3B +.078 [+.029, +.132]), extinguished from α=2
on — the anchor's dose direction (76/192 → 101/192) **inverts** at our scales
(17→1, 16→0, 18→1 of 180): at α=2 the greedy output becomes the swapped-in
argument itself and the target answer falls out of the top ranks (vocab floor
at 1.5B). 1.5B reproduces the paper's category order exactly; workspace
loading reproduces the "numbers load lowest" tail at every scale and its top
end comes into alignment at 3B (countries load highest and route best).
Conditioned α=1 routing: 13/42 / 12/62 / 16/56 (~3–4× unconditional) — the
numbers-category zero is a knowledge confound (0/16 diagonals on every
subject; the models continue "Two times three equals" with " what", or bare
whitespace at 3B).

## S3 — selectivity (stretch): D23–D26 frozen 2026-07-17 (Kyle)

*All recommendations accepted (S3-BRIEF menu, "across the board"). Scope: the
paper's selectivity property — targeted flexible-vs-automatic reading contrasts
plus the J-space ablation. One new operator (projection removal), invariant-gated
per the M1-D6 precedent (the reference ships no ablation code, so invariants
replace the AGREE diff). No new model, fit, or band. Framing stays descriptive
(triple readability NULL holds).*

**D23 (Kyle) — Scope: both bundles.** The two shipped targeted item sets
(`selectivity-language.json`, 8 passages × explicit/automatic;
`selectivity-linecount.json`, 11 passages × 4 conditions) as reading contrasts,
plus the ablation arms. *Why:* the targeted sets are nearly free forward-only
texture that frames the ablation result; the ablation probes causation, where
every real finding of this project has lived.

**D24 (Kyle) — Ablation conventions.** k = 10 and the clean-top-10 exclusion
rule verbatim from the paper (never ablate a token in the clean forward pass's
top-10 next-token predictions at that position — our per-position reading,
owned). Ablation = zero the residual's projection onto the span of the selected
lens directions (the unique minimal edit that leaves every selected direction's
lens coordinate at exactly zero — "zero out the projection onto each" achieved
simultaneously; sequential one-at-a-time zeroing of non-orthogonal directions
would leave residues, S3-BRIEF assumptions). Strength tiers start-anchored on
the frozen band: light = first third of band layers, medium = first two-thirds,
heavy = full band (len//3 and 2·len//3 prefixes), so S1's mid-band finding is
tested by the design rather than baked into it. Random-direction control:
matched count (10 per layer × position), medium tier, frozen seed. Selection is
per band layer × position by lens logit (the layer's own J_l), the operational
reading of "most strongly activated" — owned.

**D25 (Kyle) — Ablation eval sets.** Flexible task: M2's shipped two-hop items
and grading verbatim (greedy top-1 vs the answer's single-token forms; primary
cell = baseline-correct items) — our own unablated baselines exist, so ablation
deltas are like-for-like within-project. Automatic task: fresh WikiText-103
records, the D3 selection rule (first qualifying ≥600-char records, streamed in
order) but taking the records *after* the first 100 — provably disjoint from
every fit corpus. Metric: per-position top-1 match between ablated and clean
next-token predictions (per-position primary, per-prompt as robustness view).

**D26 (Kyle) — Readout frame + underpowering.** Targeted item sets run verbatim
and every targeted cell is pre-declared UNDERPOWERED (8 and 11 passages vs the
standing N ≥ 20 rule) — texture, not CI-gated claims. Lens-presence convention
for the targeted arms: a position scores a hit if any target-set token is in the
lens top-10 at any band layer (top-10 matches M0's pass@10 and the ablation's
k; top-5 reported as texture). Only the ablation cells carry CI-gated
statements, per the pre-committed wording: **"selectivity-consistent"** iff, per
subject, (i) heavy J-ablation drops two-hop accuracy on the primary cell
CI-cleanly below the unablated baseline (Newcombe excludes 0), (ii) wikitext
top-1 match under heavy J-ablation stays CI-cleanly above the two-hop retention
rate, and (iii) the random control disrupts two-hop CI-cleanly less than
J-ablation at the medium tier. Failing (iii) is the headline *against*
("any-direction damage, not workspace selectivity"). Degeneracy guard (S1/S2's
COLLAPSE_SHARE = 0.5) on every ablated two-hop cell; D6-style runtime read-back
(post-ablation lens coordinates on the selected directions must be ~0) on every
applied edit, INVALID on failure.

**S3 outcomes (record, 2026-07-17).** Two-hop plan exactly as frozen (81
gradable); baselines reproduce M2 bit-for-bit (28/81, 41/81, 43/81).
The D6-style read-back fired twice during the build — a least-squares
projection blowing up on ill-conditioned real direction sets, then LAPACK SVD
non-convergence — and the operator shipped as modified Gram-Schmidt; it also
surfaced a silent MPS `.to("cpu", float64)` value-corruption bug (fixed:
move-then-cast; project memory). On real runs: read-back silent everywhere,
degeneracy guard silent in every cell (worst share .20). **Would-gate:
selectivity-consistent on all three subjects** — leg (i) clean − heavy: +.964/+.878/+.930;
leg (ii) match − retention: +.187/+.244/+.358; leg (iii) random −
J-medium: +.536/+.488/+.395. Structure: 0.5B all-tier cliff, 1.5B
graded curve (21→13→5 of 41), 3B sharper still (27→17→3 of 43). Wikitext degradation is
graded and monotone with tier at every scale but far from the paper's "mostly
intact" (heavy match .223/.366/.428) — the owned framing is
*relative* selectivity (flexible hit CI-cleanly harder than automatic, and
targeted CI-cleanly harder than random), never surgical precision. Targeted
texture (UNDERPOWERED as pre-declared): language label enters the lens on
demand (explicit 7/8, 8/8, 8/8 vs automatic 0/8, 2/8,
1/8); linecount presence peaks under the direct question and
trails under automatic linewrap at every scale. Qwen's digit-by-digit number
tokenization emptied the README's two-digit target half — number-words only,
owned in the brief's deviations table.

## S4 — naming vs avoiding (stretch): D27–D30 frozen 2026-07-17 (Kyle)

*All recommendations accepted (S4-BRIEF menu; D27 = option c). Scope: the
paper's Figure-69 inclusion/exclusion experiment — single-concept (k = 1)
J-lens ablation at sub-band depths, naming vs avoidance instructions. No new
model, fit, band, or operator (S3's `ablate` unchanged). Framing descriptive.*

**D27 (Kyle) — Constrained item construction.** The reference ships no item
set (checked 2026-07-17), so items are constructed — the project's first —
under frozen rules: concepts only from measured vocabularies (S2 args:
countries/months/animals, numbers excluded per the S2 knowledge confound; M3
topic_categories: planets, instruments, gemstones, metals, farm animals,
insects), controls from the same shipped list, 20 concepts × 3 clues = 60
items, no clue contains its concept/control word or a derivative (test-
guarded). Frozen in `s4-avoidance-items.json` before any run. *Why:* the
paper's own items were constructed too; its competence gate, not authorship,
is what filters items honestly — and measured vocabularies mean we already
know what each subject knows.

**D28 (Kyle) — Ablation depths + control.** k = 1 (the implied concept's own
J-lens vector, M1 direction convention, bare-form token), applied at every
prompt position across the S1 sub-band thirds of the frozen band — early /
middle / late (paper contrasts early vs late; middle is free texture).
Specificity control: the same-category alternative's vector through the
identical operator, early tier. S3's runtime read-back and degeneracy guard
carry over unchanged.

**D29 (Kyle) — Readout + competence gate.** One user turn through the chat
template (M1-D5). Primary binary per trial: the concept's single-token form
is the greedy next token (produced / not). Concept softmax probability mass
recorded as paper-comparable texture. Competence gate: **greedy-based** —
unablated naming produces the concept AND unablated avoidance does not — with
the paper's verbatim P ≥ .85 / P ≤ .15 pass-rate reported alongside. Gated
set = primary cell; n < 20 ⇒ pre-declared UNDERPOWERED. Items whose concept
or control lacks a single-token form are dropped and counted (standing M0
filter).

**D30 (Kyle) — Would-gate wording (pre-committed).**
**"Avoidance-dissociation-consistent"** iff, per subject, on the gated cell:
(1) avoidance failure under early primed ablation CI-cleanly above unablated
(Newcombe excludes 0); (2) naming success under early primed ablation shows
NO CI-clean drop vs unablated — a pre-declared null leg; (3) early primed
ablation raises avoidance failure CI-cleanly more than the early control
ablation. Late-layer suppression is descriptive texture, not gated. INVALID
on wrong-arm inputs; `--dry-run` validates and stops; `--limit` is smoke.

**S4 outcomes (record, 2026-07-17).** Items 60/60 gradable; read-back and
degeneracy guard silent. **Would-gate: NOT shown on all three subjects** —
leg 1 (early breaks avoidance) fails everywhere (0.5B +.400 [−.118, +.769]
n=5 UNDERPOWERED; 1.5B +.000 [−.149, +.149] n=22; 3B +.000 n=8 UNDERPOWERED);
leg 2 (early spares naming, the pre-declared null leg) holds everywhere; leg
3 (primed > control) fails everywhere. The competence gate is itself the
first finding: clean naming 20/28/21 of 60, clean avoidance blurts 17/6/13 —
only 1.5B reliably does exclusion (the M1/S1 subject again). The late half of
Figure 69 reproduces as a hard switch: late-third k=1 ablation → naming 0/n
and concept mass ≈ .000 on every subject (no matched late control — D28
scoped the control to early; owned texture, not a claim). Sub-threshold mass
rise under early ablation at 0.5B/3B (texture). Greedy-first-token naming
grading undercounts prefixed answers ("The …") — owned readout caveat on the
competence numbers.
