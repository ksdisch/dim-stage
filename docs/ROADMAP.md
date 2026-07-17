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
## M2 — Two-hop swap — **COMPLETE 2026-07-16** (descriptive; triple-null re-scope)

Closed in two PRs (#13 brief, #14 D9–D11 freeze + runner + results + close):

| Deliverable | Outcome |
|---|---|
| D9–D11 frozen | 2026-07-16, Kyle — all three recommendations (`DECISIONS.md`) |
| Design extraction | Shipped 90-item set = the Figure 16 experiment; M2 = the raw J-lens token-vector swap (anchors: 60% n=90 Sonnet; 54–70% tiers); probe decomposition unshipped + out of scope; Fig 13's "clamped" wording owned as a deviation |
| Two-hop swap, all subjects | Primary flips **.286 / .073 / .116** (0.5B/1.5B/3B, baseline-conditioned) vs the .60 anchor; baselines .346/.506/.531 of 81 gradable items |
| Arm 2 (J = I) | Identity rows flip **0/41 and 0/43** at 1.5B/3B; at 3B the J − I difference is **CI-clean (+.116 [+.011, +.245])** — the project's first clean J-transport advantage for *writing* |
| Answer-swap arm (D11) | 2× the intermediate rate at 0.5B, equal by 3B (6 vs 5) — no answer-smuggling signature at 3B |

**Headline:** redirecting the unspoken bridge mostly fails to redirect the
answer at these scales — but where it works at all (3B), it works *only
through the Jacobian transport*: raw unembedding rows flip nothing. The swap
disturbs chains (~40% displaced) far more often than it redirects them.
All framing descriptive per the pre-registered re-scope.
## M3 — Directed modulation — **COMPLETE 2026-07-16** (descriptive; triple-null re-scope)

Closed in two PRs (#15 brief, #16 D12–D14 freeze + runner + results + close).
A **reading** milestone (the modulation is the instruction; M1-D8's "steering
required for M3" rationale was wrong — owned in the brief and DECISIONS):

| Deliverable | Outcome |
|---|---|
| D12–D14 frozen | 2026-07-16, Kyle — all three recommendations (`DECISIONS.md`) |
| Anchor check | The KICKOFF-cited anchor **reproduces**: no-instruction baseline ≈ 0 on every subject, both arms (pooled 0/46, UB .077 < the pre-declared .10) |
| Would-gate wording | **"Does not modulate" on all three subjects** — the math family is a hard zero everywhere and the frozen gate needs both families |
| The structure inside | Category-family focus signal is real, **ordered as the paper describes** (focus ≫ control ≈ suppress ≈ baseline ≈ 0), CI-clean at 1.5B and 3B, and **grows CI-cleanly with scale** (J-lens focus 3B vs 0.5B +.064 [+.004, +.131]) |
| Falsification arm | **CI-clean J-transport reversal at 3B**: the plain logit lens reads the focused concept ~2× the J-lens (19/110 vs 9/110; J − logit −.091 [−.181, −.002]), itself contrast-clean |

**Headline:** small Qwen models show a genuine, dose-ordered, scale-growing
trace of top-down workspace control on concrete category content — but an
order of magnitude below Claude, absent entirely for computed math content,
with no mention-priming and no white-bear effect (nothing enters the
workspace uninstructed to suppress) — and at 3B the phenomenon is read
*better without* the Jacobian transport. All framing descriptive.

## v1 — CLOSED 2026-07-16

**All three properties measured on all three subjects (2026-07-16)**, and the
final KICKOFF "v1 done" item — the README honesty contract — shipped as the
top-level `README.md` (the synthesis of all four milestone verdicts under the
standing frame; Kyle signed off on framing before merge). Every "v1 done means"
line in `KICKOFF.md` is now discharged.

## S1 — Introspection dose–response follow-up (stretch) — **COMPLETE 2026-07-16**

First stretch stage (Kyle's post-v1 pick: deepen the strongest finding).
Bundle 2 — harden + localize the 1.5B injected-thought dose–response. Closed in
two PRs (#18 brief, build PR D15–D18 freeze + runner + results + close). No new
model, fit, band, or intervention operator — reuses M0/M1 artifacts. Descriptive
framing holds (triple readability NULL).

| Deliverable | Outcome |
|---|---|
| D15–D18 frozen | 2026-07-16, Kyle — Bundle 2 + all three convention defaults (`DECISIONS.md`) |
| B — `J = I` falsification arm (the arm the flagship finding never had) | **J-transport advantage CI-clean from α=1**, peak +.178 [+.067, +.286] at α=8; J-lens ~2× the raw-unembedding arm (30–31 vs 12–14). The 1.5B dose–response is a genuine workspace effect, matching the paper's specificity control — the project's first CI-clean J-transport advantage for *report* |
| A — saturation (extended α grid) | Plateaus ~30/101 from α=8 (30/30/29/31 across 8→24), MRR still tightening (.067→.125) — the paper's Figure-7 rise-then-saturate. **No subject collapsed** at any α (degeneracy guard silent) |
| C — layer localization (1.5B) | **The middle third L16–20 alone recovers 29/101 ≈ the full band's 31** (full−mid +.020 [−.105, +.144], overlaps 0) — the paper's mid-layer "middle block." J-transport advantage CI-clean mid (+.129) and late (+.129) |
| Cross-scale bonus | The extended grid exposed **3B reporting as *purely* J-transport**: identity arm dead (0–1/101), J-lens →9/101, J−I CI-clean from α=8. 0.5B null both arms |

**Headline:** the project's strongest finding, hardened on every front — the 1.5B
injected-thought dose–response is CI-cleanly a **J-transport** effect (not
any-direction steering), it **saturates** as the paper does without breaking the
model, and it **localizes to the middle of the workspace band**. A stretch stage
that turned one curve into a characterized, falsification-survived, layer-localized
phenomenon. All descriptive per the standing re-scope.

## Remaining stretch (gated, un-scheduled) — generalization, selectivity

The introspection follow-up is done. Generalization (needs N≈200/arm — free but
slow) and selectivity remain **new scope decisions for Kyle, not defaults**.
Nothing is scheduled until he picks.

## S2 — flexible generalization (stretch) — **COMPLETE 2026-07-16** (descriptive)

Second stretch stage (Kyle's pick over selectivity and wrap-up). The paper's
**broadcast** test: sixteen function templates, one identical lens-coordinate
swap clamped at every prompt position, graded on whether each function's greedy
answer becomes the swapped-in argument's answer. Verbatim item set from the
reference repo (192 trials; 180 gradable under the standing single-token
filter). Closed in two PRs (#20 brief, #21 D19–D22 freeze + α operator +
runner + results + close). No new model, fit, or band; one operator extension
(α on the swap), invariant-gated.

| Deliverable | Outcome |
|---|---|
| D19–D22 frozen | 2026-07-16, Kyle — all four recommendations (`DECISIONS.md`) |
| Anchor cells (would-gate) | **"Does not route" on all three subjects** — α=1 J-lens **17/16/18 of 180** vs the paper's 76/192; α=2 **1/0/1** vs 101/192 |
| The α=1 routing signal | **J−I CI-clean at α=1 on all three subjects**: 0.5B **+.078 [+.031, +.131]**, 1.5B **+.061 [+.012, +.114]**, 3B **+.078 [+.029, +.132]** — a transport-specific routing effect exists, an order of magnitude below anchor (identity rows: 3–5/180) |
| The α cliff (vs the paper's dose direction) | The paper's α=2 *rescue* inverts: hits collapse to ~0 from α=2 on every subject; at α=2 the greedy output becomes the swapped-in **argument itself** and the target answer falls out of the top ranks (to the vocab floor at 1.5B, median ~151,844/151,936) — overdose turns a routed argument into a verbalization impulse. D6 read-back silent throughout; the degeneracy guard silent through the cliff (blurted outputs are real words) and fired exactly once — 3B α=8, both arms, `!` attractor at share 1.00 (true junk collapse, a separate regime) |
| Category structure + loading | 1.5B matches the paper's order **exactly** (countries 10/48 > months 4/48 > animals 2/36 > numbers 0/48; 3B: countries 14/48 dominate, numbers 0); workspace loading puts **numbers lowest at every scale** (the paper's own worst-category prediction), and the predictor sharpens with scale — at 3B the top end aligns too (countries load highest and route best) |
| Conditioned frame (D22) | Where both facts are provably known, α=1 routing is **13/42 (0.5B), 12/62 (1.5B), 16/56 (3B)** — ~3–4× the unconditional rate — and the numbers zero is a knowledge/pragmatics confound (0/16 baselines everywhere; models continue "Two times three equals" with " what", or bare whitespace at 3B) |

**Headline:** the broadcast property has a real, falsification-surviving trace
at small scale — a J-transport-specific routing signal at α=1, with the paper's
own category ordering and loading prediction — but it is narrow: an order of
magnitude below the anchor, extinguished (not amplified) by the paper's own
double-strength dose, and gated by whether the model knew the facts at all.
All framing descriptive per the standing re-scope.

## S3 — selectivity (stretch) — **COMPLETE 2026-07-17** (descriptive)

Third stretch stage (Kyle's pick). The paper's **selectivity** test, both
halves: the two shipped targeted reading contrasts (language, linecount —
verbatim item sets, pre-declared UNDERPOWERED texture) and the **J-space
ablation** — a new projection-removal operator (top-10 lens directions per
band layer × position, clean-top-10 output exclusion, span projected out),
invariant-gated in place of AGREE (the reference ships no ablation code).
Closed in two PRs (#23 brief, build PR D23–D26 freeze + operator + runner +
results + close). No new model, fit, or band.

| Deliverable | Outcome |
|---|---|
| D23–D26 frozen | 2026-07-17, Kyle — all recommendations (`DECISIONS.md`) |
| Ablation operator + invariant gates | Modified Gram-Schmidt projection after the runtime read-back caught a least-squares blow-up and an SVD non-convergence on real direction sets — plus a **silent MPS `.to("cpu", float64)` corruption** found and fixed (project memory). 16 pre-committed gates; read-back silent on every real edit |
| Would-gate, all three subjects | **Selectivity-consistent on all three subjects — the project's first would-gate to HOLD everywhere** — all three frozen legs (heavy kills two-hop; wikitext survives above retention; random control far gentler) |
| The structure inside | 0.5B: cliff (1/28 at every tier). 1.5B: the paper's graded dose curve (21 → 13 → 5 of 41). 3B: sharper still (27 → 17 → 3 of 43). Random control retains 16/28 / 33/41 / 34/43 |
| Targeted texture | Language label present-on-demand (explicit 7/8, 8/8, 8/8 passages vs automatic 0/8, 2/8, 1/8); linecount count-presence peaks at the direct question, trails under automatic linewrap, every scale |

**Headline:** the paper's flexible-vs-automatic dissociation is present and CI-clean at all three scales — targeted removal of the top-10 lens directions kills two-hop chains (retention .04/.12/.07 at heavy) while matched random damage and ordinary WikiText prediction survive CI-cleanly above it — the only property in this project to clear its full pre-committed gate on every subject. Owned framing: *relative* selectivity (heavy ablation still changes 57–78% of ordinary predictions — not the paper's “mostly intact”). Texture: presence-on-demand in both targeted arms, and the paper's letter > direct ordering emerging at 3B (.028, 11/11 passages).

## S4 — naming vs avoiding (stretch) — **COMPLETE 2026-07-17** (descriptive)

Fourth stretch stage (Kyle's pick): the paper's Figure-69 inclusion/exclusion
experiment — k = 1 ablation of an implied concept's lens direction at
sub-band thirds, naming vs avoidance instructions. First **constructed** item
set of the project (the reference ships none): 20 concepts × 3 clues from
measured vocabularies, frozen pre-run, leakage test-guarded. Closed in two
PRs (#25 brief, build PR D27–D30 freeze + items + runner + results + close).

| Deliverable | Outcome |
|---|---|
| D27–D30 frozen | 2026-07-17, Kyle — all recommendations (`DECISIONS.md`) |
| Competence gate (the first finding) | Only 1.5B reliably does exclusion: gates 5 / **22** / 8 of 60 (0.5B and 3B blurt the forbidden concept unablated on 17 and 13 items) — the M1/S1 subject again |
| Would-gate | **NOT shown on all three subjects** — early primed ablation raises avoidance failure nowhere (0/22 at 1.5B; CIs straddle 0 everywhere); the pre-declared null leg (naming spared) holds everywhere; primed > control fails everywhere |
| Late-band texture | The paper's "output intention" half reproduces as a **hard switch at every scale**: late-third k=1 ablation → naming 0/n, concept mass ≈ .000 (no matched late control — owned as texture, not a claim) |

**Headline:** at these scales the concept's late-band lens direction is a
clean output off-switch, but the paper's early-band *suppression copy* does
not appear — where small models can do the exclusion task at all, they do it
without that machinery. A pre-registered NOT-shown, reported as such.

**S4b — late-tier specificity control — COMPLETE 2026-07-17.** D31 (Kyle)
added the matched same-category control at the middle + late tiers; the full
re-run reproduced every shared cell bit-for-bit. **The off-switch is
concept-specific on the powered subject** (1.5B: primed 0/22 vs control
16/22, +.727 [+.471, +.868] CI-clean; 3B: 0/8 vs 8/8, CI-clean but
UNDERPOWERED-tagged) — and the specificity emerges with scale (0.5B's late
tier breaks under any single-direction removal; 3B's control is untouched,
mass .924 ≈ clean). Middle-tier cells benign everywhere. Closed in PR #27
(brief) + the S4b build PR.

## Remaining stretch — none scheduled

All in-scope paper properties are measured: readability (M0), report (M1),
two-hop (M2), modulation (M3), introspection dose–response (S1), broadcast
(S2), selectivity (S3), naming-vs-avoiding (S4). What remains is Kyle's wrap
decision (/seed-hunt or a targeted follow-up such as a late-tier specificity
control for S4's switch texture). Nothing is scheduled until he picks.
