# Is the Global Workspace Readable in Small Language Models? A Pre-Registered Three-Scale Null, and the Structure Inside It

*dim-stage — a hobby-scale reproduction and measurement study. All numbers in this
paper are lifted verbatim from the repository's committed result files
(`results/*.json`) and stage records (`docs/*-BRIEF.md`, `docs/DECISIONS.md`,
`docs/ROADMAP.md`); nothing was re-run or re-derived for the write-up. The six
figures are rendered deterministically from the committed result JSONs by
`docs/paper/figures.py` (`uv run --with matplotlib docs/paper/figures.py`); every
plotted value is a count or rate already recorded in those files.*

## Abstract

Anthropic's paper "Verbalizable Representations Form a Global Workspace in Language
Models" (2026) introduces the Jacobian lens — a per-layer linear map that transports a
mid-layer residual-stream activation into the final layer's basis before decoding it
with the model's own unembedding — and reports that what it reads out behaves like a
global workspace: reportable, swappable, steerable content. The method is demonstrated
at 27B parameters and above; the paper names readability below that scale an open
question. We answer it for Qwen2.5-0.5B, 1.5B, and 3B-Instruct. We re-implemented the
lens independently from the paper's spec, validated it against the reference
implementation (bitwise-identical Jacobians at every layer; 3220/3220 readout
agreement), and measured readability plus seven workspace properties under
pre-committed statistical gates (Wilson 95% intervals on cells, Newcombe 95% intervals
on differences, N ≥ 20 or pre-declared UNDERPOWERED). The headline is a pre-registered
null: 0 of 6 evaluation distributions reach the frozen readability bar at any of the
three scales. Downstream properties, measured descriptively under a pre-registered
re-scope, land far below the paper's anchors (verbal report .105–.175 vs .88; two-hop
flips .073–.286 vs .60) — but with real structure inside: a clean injected-thought
dose–response at 1.5B (0 → 30/101) that is CI-cleanly transport-specific and localizes
to the middle of the workspace band; a CI-clean Jacobian-transport advantage for
two-hop writing at 3B; and selectivity — flexible reasoning dies under targeted
workspace ablation while random damage and ordinary prediction survive — as the only
property to clear its full pre-committed gate on all three subjects.

## 1. Introduction

Anthropic's workspace paper makes a concrete, testable set of claims — a mid-layer
band of the residual stream, read through a Jacobian lens, holds content the model
can verbally report, that mediates multi-step reasoning, and that the model can
modulate on instruction — and demonstrates them on Claude-family models and a 27B
open model. The paper explicitly leaves open whether any of this is readable below
that scale.

This project's contribution is deliberately narrow and honestly framed: **we
reproduced and measured a published method — here is the narrow, measured slice.**
Nothing here is invented. We built the lens-fitting procedure independently from the
paper's specification, proved it computes the same object as the reference
implementation, fitted lenses to three small instruction-tuned Qwen models on
consumer hardware (one fit required $0.83 of rented GPU), and ran the paper's own
shipped experiments under gates whose wording — including the verdict sentences — was
frozen as code before any real run.

The result is a null with structure. At the pre-registered bar, the workspace is not
readable at 0.5B, 1.5B, or 3B: the pre-declared kill-risk fired and held under a
pre-registered escalation to the third scale. Everything downstream of readability was
therefore re-scoped — also pre-registered — from reproduction claims to *descriptive
characterization*: the identical protocols and statistics, with verdicts worded as
"what swaps/steering/ablation do at scales where the workspace is not readable."
Within that frame, small models are not a scaled-down copy of the paper's subjects
but a different regime with recognizable traces of the same phenomena: one subject
(1.5B) reports injected thoughts along a clean dose–response curve; the Jacobian
transport is sometimes the entire effect and sometimes worse than doing nothing; and
the band the paper calls the workspace is causally load-bearing for flexible
reasoning at every scale we can reach, even though its contents are not readable at
the frozen bar.

A pre-committed null is a reportable result. This paper reports it, its escalation,
and the property measurements around it, with every gate verdict as frozen —
including the ones that came back "not shown."

## 2. Background and method

### 2.1 The Jacobian lens

The **residual stream** is the running vector each transformer layer reads and adds
to, one vector per token position. The **unembedding** turns the final layer's
residual vector into one score (**logit**) per vocabulary token. The classic **logit
lens** applies the unembedding directly to a *middle* layer's residual vector; it
assumes middle layers share the final layer's coordinate system, which they mostly do
not. The **Jacobian lens** repairs this by first multiplying the middle-layer vector
by `J_l`, the *average Jacobian* of the final layer's residual with respect to layer
`l` — a matrix of sensitivities estimated by backpropagation over a fit corpus — and
then unembedding: `lens_l(h) = unembed(J_l @ h)`.

Following the reference implementation's spec, `J_l` is estimated by injecting one-hot
cotangents at every valid target position at once and backpropagating (a
vector–Jacobian product per output dimension), summing over current-and-future target
positions, averaging over source positions, and running-averaging over prompts. The
paper locates the lens-readable "workspace" in a contiguous band of middle layers —
roughly 38%–92% of network depth, on layers reindexed to percentages.

### 2.2 What we measured

Eight properties, each anchored to the paper's own experiments and, wherever the
reference repository ships stimuli, run on those stimuli verbatim:

- **M0 Readability** — can the lens recover known workspace content (six shipped
  evaluation distributions) at rank ≤ 10?
- **M1 Verbal report** — swap the model's ready answer in lens coordinates; does the
  spoken report follow? Plus **verbal introspection**: steer a concept in and ask the
  model to name it.
- **M2 Two-hop swap** — swap the unspoken bridge entity mid-reasoning; does the final
  answer follow the redirected chain?
- **M3 Directed modulation** — instructed to think about (or ignore) a concept while
  copying unrelated text, does the concept surface in the lens?
- **S1** — harden M1's dose–response: falsification arm, saturation, layer
  localization.
- **S2 Flexible generalization (broadcast)** — one identical swap across sixteen
  function templates; do many circuits consume the swapped argument?
- **S3 Selectivity** — ablate the workspace (project out the top-10 lens directions
  per layer × position); flexible reasoning should die, automatic prediction survive.
- **S4/S4b Naming vs avoiding** — delete one implied concept's lens direction early
  or late in the band, under naming vs avoidance instructions.

### 2.3 Measurement discipline

Every verdict in this project was decided by machinery frozen before the first real
run of its stage:

- **Deterministic oracles only.** Every outcome is a logit ranking read from
  tensors — rank-1 hits over the workspace band, greedy next-token identity, swap
  grading by the rank of the swapped-in candidate. No LLM judges, no text parsing.
- **Wilson 95% confidence intervals on cells; Newcombe 95% intervals on
  differences** — like-for-like with the paper, which reports Wilson intervals
  natively. A difference whose interval includes zero is stated as a null or small
  effect, never a win.
- **N ≥ 20 per cell or the verdict is pre-declared UNDERPOWERED.**
- **Gates as code, dry-run first.** Each runner validates its inputs and exits
  `VERDICT: INVALID` on wrong-arm input; the wrong-arm dry-run was executed before
  every official run. Verdict wording — including "does not modulate" and
  "NOT shown" — was frozen in advance.
- **A standing falsification arm.** Nearly every experiment repeats the identical
  operation with `J = I` (raw unembedding rows; the plain logit lens, no transport),
  with a Newcombe interval on the difference. Any claim about the *Jacobian* has to
  beat doing nothing.
- **Deviations are owned.** Every departure from the paper is a row in a per-stage
  deviations table; the headline rows appear in §6.
- **Runtime self-checks.** The intervention operators re-verify their defining
  algebra on every real application (coordinate read-back); silent across every
  recorded run, and the catcher of two genuine numerical failures during the S3
  build before any result was recorded.

### 2.4 Correctness gates

The lens fitter was validated by an **AGREE gate** against the reference
implementation on an identical model and byte-identical prompt list: maximum
per-layer relative Frobenius distance **0.000e+00** (bitwise-identical `J_l` at every
layer, against a tolerance of 1e-3) and top-1 readout agreement **3220/3220** held-out
cells (Wilson lower bound .9988 ≥ .95). The reference package is pinned as a
dev-dependency used by the cross-check only; no measurement code imports it.

The reference ships no intervention or ablation code, so the swap, steering, and
projection-removal operators are gated instead by **pre-committed invariants**: a
rigged analytic subject whose Jacobian is known exactly (every post-patch logit
asserted with exact equality), exact null-op checks (α = 0 steering and
source = target swaps change nothing), agreement with the paper's literal
pseudoinverse formula on random tensors, and the runtime read-back above.

## 3. Experimental setup

**Subjects.** Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, and (escalation)
Qwen2.5-3B-Instruct. Instruct variants because the shipped stimuli are chat-format.
The kickoff pre-registered the escalation trigger: fit 3B only if *both* smaller
subjects came back null on readability. They did, and it was.

**Bands.** The paper's 38–92% depth band, transplanted proportionally and
pre-registered per subject: 0.5B layers 9–21, 1.5B layers 11–24, 3B layers 14–32. The
3B band was frozen and merged before its fit produced any readout.

**Lens fits.** Fit corpus: the first 100 WikiText-103 records with ≥ 600 characters,
streamed in order via the reference's own loader convention — deterministic and
byte-identical across implementations (an owned deviation from the paper's ~1000
generic web-text sequences; the paper's own §9.3 reports quality saturating from far
fewer prompts). Fits ran in fp32 on Apple-silicon MPS (0.5B: 71 min; 1.5B: ~6.1 h,
against a measured hour-one rate of 42.7 s/prompt). The 3B fp32 backward pass
exceeded the 24 GB unified-memory working set (a measured ~25× slowdown cliff at
every batching setting), so the 3B fit ran on a rented RTX 4090 (CUDA fp32, ~57 min
at 34.5 s/prompt, total spend ~$0.83) with byte-identical procedure and prompts; the
returned lens was checksum-verified and all grading ran locally like the other two
subjects. An incidental measured fact: raising the fitter's `dim_batch` from 8 to 32
is math-neutral but ~25× *slower* on MPS (44.1/42.7 s vs 1192.0/931.0 s on the first
two prompts).

**Stimuli.** The paper's own experiment data as shipped in the reference repository
(`verbal-report.json`, `verbal-introspection.json`, `probe-swap.json`,
`directed-modulation.json`, `flexible-generalization.json`, the two selectivity item
sets, and the six `lens-eval-*` distributions). A mechanical single-token pre-filter
under Qwen's tokenizer drops stimuli whose graded token has no single-token form;
every drop is counted (94–100% of intermediates survive per M0 distribution; 9/90
two-hop items; 12/192 generalization trials).

**Constructed components are labeled as constructed.** The S4 item set is the
project's first constructed stimulus set (the reference ships none for that
experiment): 20 concepts × 3 clues drawn only from vocabularies already measured in
earlier stages, frozen in-repo before any run, with the paper's own competence-gate
pattern doing the filtering. Three smaller owned constructions: M3's copy-instruction
prompt frame and its no-instruction baseline condition (the paper specifies neither),
and the steering/swap strength grids (the paper sweeps strength but publishes no
grid). Each is pre-declared in its stage brief.

**Modes.** After M0's triple null, every downstream verdict is **descriptive** — the
pre-registered re-scope. Gate wording was still frozen per stage before its runs, so
"what the gate would have said" is reported without post-hoc choice; we write those
as *would-gate* verdicts below.

## 4. Results

All intervals are Wilson 95% (cells) or Newcombe 95% (differences). "J − I" is the
J-lens arm minus the identity (`J = I`, raw-unembedding / logit-lens) arm. Counts and
intervals are quoted from the committed per-run JSONs and stage records; sub-tables
list the source file of each block.

### 4.1 M0 — Readability: NULL at all three scales

Gate (frozen): a subject READS iff J-lens pass@10 has Wilson lower bound ≥ 0.5 on
≥ 3 of 6 distributions. **Verdict: NULL on 0.5B, 1.5B, and 3B — 0/6 distributions
pass on every subject.** The pre-declared kill-risk fired, and held under the
pre-registered 3B escalation: no emergence point exists in the reachable range.

| Distribution | 0.5B J-lens pass@10 | 1.5B | 3B | Closest arm-2 story |
|---|---|---|---|---|
| association | 0/99 | 0/99 | 0/99 | no gap at any scale |
| poetry | 0/98 | 1/98 | 0/98 | no gap at any scale |
| multihop | 35/94 (37.2%) | 51/94 (54.3%) | 37/94 (39.4%) | no gap |
| multilingual | 149/414 (36.0%) | 155/414 (37.4%) | 148/414 (35.7%) | J-advantage at 0.5B only (+.208 [+.149, +.265]) |
| order-ops | 36/109 (33.0%) | 39/109 (35.8%) | 50/109 (45.9%) | J-advantage at 3B (+.165 [+.037, +.286]) |
| typo | 47/96 (49.0%) | 17/96 (17.7%) | 20/96 (20.8%) | 0.5B +.427 [+.309, +.531]; 1.5B **−.271 [−.389, −.141]**; 3B **−.323 [−.442, −.188]** |

*Source: `results/readability-qwen2.5-{0.5b,1.5b,3b}-instruct.json`; tables in
`docs/M0-BRIEF.md`.*

![M0 readability: J-lens pass@10 per distribution and subject vs the frozen READS bar](figures/fig-m0-readability.png)

*Figure 1 — The headline null: J-lens pass@10 per evaluation distribution with the
recorded Wilson 95% intervals (N per cell 94–414), three subjects. The dashed line
marks the frozen READS criterion, which applies to the Wilson **lower** bound —
1.5B multihop's point estimate (54.3%) crosses the line but its lower bound (.442)
does not. Plotted values and intervals are read verbatim from
`results/readability-qwen2.5-{0.5b,1.5b,3b}-instruct.json`.*

Structure inside the null: the two abstract-content distributions (association — an
evoked, unnamed concept; poetry — a planned rhyme word) are hard zeros at all three
scales — precisely the most workspace-like content. Surface-adjacent content is
partially readable but sub-bar and non-monotone in scale: multihop peaks at 1.5B
(54.3%) then regresses at 3B (39.4%), and the closest any cell comes to the bar is
order-ops at 3B (45.9%, Wilson lower bound .368 < .5). The transport's value is
content-dependent, not a scale trend: on typo it *reverses* CI-cleanly at 1.5B and
3B (the plain logit lens reads surface-form content better), while order-ops regains
a CI-clean J-advantage at 3B.

### 4.2 M1 — Verbal report: the report does not follow the swap

Protocol: swap the model's greedy answer for a candidate (rank ≥ 11 at baseline) in
lens coordinates across the band; success = the swapped-in candidate reaches top-5 at
the readout position. Paper anchor: .88 (Claude Sonnet 4.5).

| Subject | n eligible | J-lens top-5 | J = I top-5 | J − I (Newcombe 95%) |
|---|---|---|---|---|
| 0.5B | 97 | 17/97 = .175 [.112, .263] | 13/97 = .134 | +.041 [−.062, +.144] |
| 1.5B | 89 | 11/89 = .124 [.070, .208] | 8/89 = .090 | +.034 [−.060, +.129] |
| 3B | 86 | 9/86 = .105 [.056, .187] | 10/86 = .116 | −.012 [−.109, +.086] |

*Source: `results/verbal-report-qwen2.5-*.json`.*

Every Wilson upper bound sits below .27 — a fifth of the anchor. No J − I interval
excludes zero, and at rank-1 the raw unembedding rows beat the J-lens vectors on
every subject (7/5/9 vs 3/4/6 hits). The workspace-write does not reach the spoken
report at these scales, with or without the transport.

### 4.3 M1 — Verbal introspection: a dose–response at 1.5B only

Protocol: steer a concept's direction into the band over the question turn of an
"do you detect an injected thought?" chat; report rate = the concept is rank-1 at the
open quote of a teacher-forced reply. α = 0 is the control. (The paper cites a best-
strength report rate of 0.54 for Claude; it publishes no strength grid — ours is an
owned convention.)

| α (mean-residual-norm units) | 0.5B | 1.5B | 3B |
|---|---|---|---|
| 0 (control) | 0/101 | 0/101 | 0/101 |
| 0.5 | 0/101 | 7/101 | 2/101 |
| 1 | 0/101 | 16/101 | 2/101 |
| 2 | 0/101 | 23/101 | 4/101 |
| 4 | 0/101 | 26/101 | 4/101 |
| 8 | 0/101 | **30/101 = .297 [.217, .392]** | 5/101 = .050 [.021, .111] |

*Source: `results/introspection-qwen2.5-*.json`; the median steered ranks quoted
below are recomputed from the per-concept ranks in `results/derived-contrasts.json`
(original record: `docs/DECISIONS.md` / `docs/M1-BRIEF.md`).*

The control is exactly 0/101 on every subject — no steered word is ever the model's
unprompted answer, so the 1.5B curve cannot be the prompt begging for concept words.
The 1.5B–3B gap at α = 8 is CI-clean ([.217, .392] vs [.021, .111]) — the strongest
descriptive contrast in the project, and non-monotone in scale. Steering moves ranks
everywhere (median steered rank, control → α = 8: 3747 → 1322 at 0.5B, 4430 → 15 at
1.5B, 3791 → 382 at 3B); only 1.5B converts movement into reports.

### 4.4 M2 — Two-hop swap: mostly fails; where it works, only through the transport

Protocol: on the paper's 90 shipped two-hop prompts (9 dropped by the single-token
filter), swap the unspoken bridge entity (e.g. Brazil → Mexico) across the band;
success = the greedy answer becomes the redirected chain's answer (top-1, the
anchor's grading). Primary cell = items the subject answered correctly unswapped.
Anchor: .60 on these 90 prompts (Sonnet 4.5); 54–70% across Claude tiers.

| Subject | Baseline | J-lens flips | J = I flips | J − I (Newcombe 95%) | Answer-swap J / I |
|---|---|---|---|---|---|
| 0.5B | 28/81 = .346 | 8/28 = .286 [.153, .471] | 4/28 | +.143 [−.075, +.347] | 16/28 / 14/28 |
| 1.5B | 41/81 = .506 | 3/41 = .073 [.025, .194] | 0/41 | +.073 [−.025, +.194] | 9/41 / 7/41 |
| 3B | 43/81 = .531 | 5/43 = .116 [.051, .245] | 0/43 | **+.116 [+.011, +.245]** | 6/43 / 3/43 |

*Source: `results/two-hop-qwen2.5-*.json`.*

Flip rates sit at .073–.286 against the .60 anchor (and the 0.5B "high" rides on only
28 working chains). The 3B cell is the project's first CI-clean Jacobian-transport
advantage for *writing*: raw unembedding rows flip 0/43 while J-lens vectors flip
5/43, and the interval excludes zero. The answer-swap comparison arm (the paper's
smuggling confound, run at a single band) is 2× the intermediate rate at 0.5B but
equal by 3B (6 vs 5) — with the raw-row zeros, no answer-smuggling signature at 3B.
The swap disturbs chains more than it redirects them: the original answer is
displaced on 14/28, 12/41, and 18/43 of primary trials.

### 4.5 M3 — Directed modulation: gate says no; the structure inside is the paper's

Protocol (reading only): instruction ("think about X" / mention / "ignore X" /
no instruction) + teacher-forced copying of an unrelated carrier sentence; hit = a
tracked token is the lens top-1 at any (layer, position) over the carrier span.
Would-gate (frozen): "modulates" iff focus − suppress excludes zero on *both* shipped
families and the pooled no-instruction baseline is clean (Wilson UB ≤ .10).
**Verdict: "does not modulate" on all three subjects** — the math family is a hard
zero everywhere (focus 0/120 at every scale, both arms) and the gate needs both
families.

| Subject (category family, J-lens) | focus | control | suppress | baseline (pooled) | focus − suppress | focus − control |
|---|---|---|---|---|---|---|
| 0.5B | 2/110 | 0/132 | 1/286 | 0/46 (UB .077) | +.015 [−.006, +.060] | +.018 [−.013, +.064] |
| 1.5B | 6/110 | 0/132 | 0/286 | 0/46 | **+.055 [+.022, +.114]** | **+.055 [+.014, +.114]** |
| 3B | 9/110 | 0/132 | 0/286 | 0/46 | **+.082 [+.041, +.148]** | **+.082 [+.034, +.148]** |

*Source: `results/directed-modulation-qwen2.5-*.json`; the cross-scale focus
contrast quoted below is recomputed in `results/derived-contrasts.json` (original
record: `docs/DECISIONS.md`, M3 outcomes).*

The paper's cited anchor reproduces: the no-instruction baseline is ≈ 0 on every
subject, both arms (pooled 0/46, upper bound .077 ≤ the pre-declared .10). On
concrete category content the signal is real, ordered exactly as the paper describes
(focus ≫ control ≈ suppress ≈ baseline ≈ 0), CI-clean at 1.5B and 3B, and its growth
across scale is CI-clean (J-lens focus, 3B − 0.5B: +.064 [+.004, +.131]). Two
divergences from the paper: mention does not prime at these scales (control 0–4 hits
of 132 everywhere, vs the paper's mention ≈ focus), and no white-bear effect is
measurable — nothing enters the workspace uninstructed (suppress ≈ baseline ≈ 0), so
suppression has nothing to visibly fail at; a non-test rather than a null. And at 3B
the phenomenon is read better *without* the transport: the plain logit lens reads the
focused concept 19/110 vs the J-lens's 9/110 (J − logit −.091 [−.181, −.002],
CI-clean, with the logit-lens signal itself contrast-clean: control 4/132, suppress
0/286).

### 4.6 S1 — The 1.5B dose–response, hardened on three fronts

S1 added the one arm M1's flagship finding never had, extended the strength grid
(appending α ∈ {12, 16, 24}; owned convention), and steered sub-band thirds
separately. No subject collapsed at any α (the degeneracy guard never fired).

| α | 1.5B J-lens | 1.5B J = I | J − I (Newcombe 95%) |
|---|---|---|---|
| 1 | 16/101 | 5/101 | +.109 [+.024, +.197] |
| 2 | 23/101 | 10/101 | +.129 [+.026, +.230] |
| 4 | 26/101 | 12/101 | +.139 [+.031, +.244] |
| 8 | 30/101 | 12/101 | **+.178 [+.067, +.286]** |
| 12 | 30/101 | 14/101 | +.158 [+.045, +.268] |
| 16 | 29/101 | 14/101 | +.149 [+.035, +.258] |
| 24 | 31/101 | 14/101 | +.168 [+.054, +.278] |

![Introspection report rate vs steering strength, three subjects, J-lens vs J = I](figures/fig-s1-dose-response.png)

*Figure 2 — Introspection dose–response: rank-1 report rate (hits / 101) vs steering
strength α over the full 9-point grid {0, 0.5, 1, 2, 4, 8, 12, 16, 24}, J-lens vs
J = I arms, n = 101 concepts per cell. Plotted values are the recorded
`report_hits`/n counts, no smoothing; α ticks are equally spaced for legibility.
Source: `results/s1-introspection-qwen2.5-{0.5b,1.5b,3b}-instruct.json`.*

| Sub-band (1.5B, α = 24) | J-lens | J = I | J − I |
|---|---|---|---|
| L11–15 (early) | 17/101 | 13/101 | +.040 [−.060, +.139] |
| **L16–20 (middle)** | **29/101** | 16/101 | **+.129 [+.014, +.240]** |
| L21–24 (late) | 18/101 | 5/101 | **+.129 [+.041, +.219]** |
| full band L11–24 | 31/101 | 14/101 | +.168 [+.054, +.278] |

*Source: `results/s1-introspection-qwen2.5-*.json`; sub-band difference intervals as
recorded in `docs/S1-BRIEF.md` / `docs/DECISIONS.md` and recomputed in
`results/derived-contrasts.json`.*

![Sub-band localization at 1.5B: report hits per steered sub-band, J-lens vs J = I](figures/fig-s1-localization.png)

*Figure 3 — S1 localization: report hits (of n = 101) when steering only L11–15,
L16–20, L21–24, or the full band L11–24, at α = 24 on 1.5B, J-lens vs J = I arms.
Plotted values are the recorded `report_hits` counts. Source:
`results/s1-introspection-qwen2.5-1.5b-instruct.json`, `localization` block.*

Three findings. **(1) The curve is a transport effect:** J-lens beats the raw-
unembedding arm CI-cleanly from α = 1, roughly doubling the report rate at the
plateau (30–31 vs 12–14) — matching the paper's specificity control, and the
project's first CI-clean J-transport advantage for *report*. **(2) It saturates, the
paper's Figure-7 shape:** the rank-1 rate plateaus at ~30/101 from α = 8
(30/30/29/31 across 8→24) while median reciprocal rank keeps tightening
(.067 → .125); genuine saturation, not collapse. **(3) It localizes to the middle of
the band:** steering only L16–20 recovers 29/101 ≈ the full band's 31/101
(full − mid +.020 [−.105, +.144], overlapping zero) — the paper's mid-layer "middle
block" at hobby scale. A cross-scale bonus from the extended grid: 3B's small
reporting signal (up to 9/101 at α = 16–24) is *purely* transport-specific — its
identity arm is essentially dead (0–1/101; J − I CI-clean from α = 8, +.050
[+.003, +.111], up to +.089 [+.034, +.161]) — while 0.5B is null on both arms.

### 4.7 S2 — Broadcast: a narrow, transport-specific routing signal that overdose destroys

Protocol: the paper's sixteen function templates over four argument categories; one
identical lens-coordinate swap clamped at every prompt position; success = the greedy
answer becomes the swapped-in argument's answer. Verbatim item set; 180/192 trials
gradable under the standing filter (the 12 drops all in animals). Anchors: 76/192 at
α = 1, rescued to 101/192 at α = 2 (Sonnet-scale). **Would-gate: "does not route" on
all three subjects** (α = 2 pooled Wilson lower bounds ≈ 0 against the frozen .5
floor).

| Cell (of 180) | 0.5B | 1.5B | 3B |
|---|---|---|---|
| α=1 J-lens | 17 | 16 | 18 |
| α=2 J-lens | 1 | 0 | 1 |
| α=1 J = I | 3 | 5 | 4 |
| J − I at α=1 | **+.078 [+.031, +.131]** | **+.061 [+.012, +.114]** | **+.078 [+.029, +.132]** |
| conditioned α=1 J-lens | 13/42 | 12/62 | 16/56 |

*Source: `results/s2-generalization-qwen2.5-*.json`.*

![S2 alpha cliff: pooled swap successes vs alpha, three subjects, J-lens vs J = I](figures/fig-s2-alpha-cliff.png)

*Figure 4 — The S2 α-cliff: pooled swap successes (of n = 180 gradable trials) at
α ∈ {1, 2, 4, 8}, J-lens vs J = I arms, per subject. Plotted values are the recorded
pooled `successes` counts. Source:
`results/s2-generalization-qwen2.5-{0.5b,1.5b,3b}-instruct.json`.*

A routing signal exists and is CI-cleanly transport-specific at α = 1 on all three
subjects — an order of magnitude below the anchor. The paper's dose direction
*inverts*: at α = 2 the greedy output becomes the swapped-in argument itself
(" France", " China") and the target answer falls out of the top ranks (at 1.5B to
the vocabulary floor — median rank 151,844.5 of 151,936 over the α = 1 hits;
recomputed in `results/derived-contrasts.json`). Overdose
converts an argument-to-compute-with into an output-to-say. The degeneracy guard
separates this behavioral blurting (real words; guard silent) from its one true
catch: 3B α = 8, where both arms collapse to a `!` attractor at share 1.00. Category
structure tracks the paper where signal exists: 1.5B matches the paper's order
exactly (countries 10/48 > months 4/48 > animals 2/36 > numbers 0/48), and the
paper's own predictor — workspace loading — puts numbers lowest at every scale
(.095/.075/.044) with the top end aligning at 3B (countries load highest, .125, and
route best). The conditioned frame (both facts provably known unswapped) shows
routing at 3–4× the unconditional rate and exposes the numbers zero as a knowledge/
pragmatics confound: 0/16 numbers diagonals on every subject (the models continue
"Two times three equals" with " what", or bare whitespace at 3B).

### 4.8 S3 — Selectivity: the only would-gate to hold on all three subjects

Protocol: at every band-layer × position, project out the residual stream's component
along the top-10 lens directions (never ablating tokens in the clean top-10
next-token predictions); tiers grow the layer range (light = first third of the band,
medium = two-thirds, heavy = full band); matched-count random-direction control at
medium. Flexible task: M2's two-hop items (baselines reproduce M2 bit-for-bit:
28/81, 41/81, 43/81). Automatic task: fresh WikiText records, provably disjoint from
the fit corpus; metric = per-position top-1 match with the clean model (~11,058
positions per cell). **Would-gate: selectivity-consistent on all three subjects** —
all three frozen legs hold everywhere.

| Leg (Newcombe 95%) | 0.5B | 1.5B | 3B |
|---|---|---|---|
| (i) clean − heavy two-hop | +.964 [+.778, +.994] | +.878 [+.719, +.947] | +.930 [+.788, +.976] |
| (ii) wikitext match − two-hop retention @ heavy | +.187 [+.045, +.218] | +.244 [+.111, +.314] | +.358 [+.241, +.404] |
| (iii) random − J-lens retention @ medium | +.536 [+.306, +.702] | +.488 [+.277, +.641] | +.395 [+.189, +.558] |

| Two-hop retention (primary cell) | 0.5B (n=28) | 1.5B (n=41) | 3B (n=43) |
|---|---|---|---|
| J-lens light / medium / heavy | 1 / 1 / 1 | 21 / 13 / 5 | 27 / 17 / 3 |
| random @ medium | 16 | 33 | 34 |

| WikiText top-1 match | 0.5B | 1.5B | 3B |
|---|---|---|---|
| J-lens light / medium / heavy | .405 / .315 / .223 | .612 / .491 / .366 | .699 / .595 / .428 |
| random @ medium | .743 | .827 | .816 |

*Source: `results/s3-selectivity-qwen2.5-*.json`.*

![S3 two-hop retention vs ablation tier, per subject, with random control and unablated baseline](figures/fig-s3-retention.png)

*Figure 5 — S3 selectivity: two-hop retention (hits / n) vs ablation tier per
subject, over the primary cell (chains answered correctly unablated; n = 28 / 41 /
43 — so the unablated baseline is 1.0 by construction, dotted line). Open diamonds:
the matched random-direction control at the medium tier (16/28, 33/41, 34/43).
Plotted values are the recorded `hits`/n counts. Source:
`results/s3-selectivity-qwen2.5-{0.5b,1.5b,3b}-instruct.json`, `two_hop.primary`.*

The paper's flexible-vs-automatic dissociation is present and CI-clean at every
scale: targeted removal kills two-hop chains while matched random damage and ordinary
prediction survive CI-cleanly above them. 0.5B is a cliff (even light ablation kills
the chain); 1.5B shows the paper's graded dose curve (21 → 13 → 5); 3B is sharper
still (27 → 17 → 3). The owned qualifier: this is **relative** selectivity — heavy
ablation still changes 78%/63%/57% of ordinary predictions (nowhere near the paper's
"mostly intact"); the claim is that the flexible task is hit CI-cleanly harder, never
surgical precision. The two shipped targeted item sets add pre-declared UNDERPOWERED
texture: the passage's language enters the lens on demand (explicit 7/8, 8/8, 8/8
passages vs automatic 0/8, 2/8, 1/8; pooled contrasts +.071 [+.027, +.118] /
+.097 [+.037, +.153] / +.139 [+.079, +.198]), and line-count presence peaks under the
direct question while trailing under automatic linewrap at every scale.

### 4.9 S4/S4b — Naming vs avoiding: NOT shown; the off-switch is concept-specific

Protocol (constructed item set — the project's first, frozen pre-run; 60 items, all
gradable): a clue implies a concept; instructions either name it or avoid it; k = 1
ablation of the concept's lens direction at early/middle/late band thirds, plus a
same-category control direction. Competence gate (greedy form of the paper's
P ≥ .85 / P ≤ .15): unablated naming succeeds AND unablated avoidance does not blurt.
**Would-gate: NOT shown on all three subjects.**

The competence gate is itself the first finding: gated cells are 5 / **22** / 8 of 60
(0.5B / 1.5B / 3B; UNDERPOWERED at 0.5B and 3B) — clean naming succeeds on only
20/28/21 of 60 items, and clean avoidance blurts the forbidden concept on 17/6/13.
Only 1.5B reliably does the exclusion task — the same subject where the introspection
phenomenon lives. On the gated cells, early primed ablation raises avoidance failure
nowhere (leg 1: 0.5B +.400 [−.118, +.769]; 1.5B +.000 [−.149, +.149]; 3B +.000
[−.324, +.324]); the pre-declared null leg (naming spared under early ablation) holds
everywhere; primed-beats-control fails everywhere. The paper's early-band suppression
copy does not appear at these scales.

The late half of the paper's dissociation reproduces as a hard switch, and the S4b
follow-up (full re-run; every shared cell reproduced bit-for-bit) added the matched
late-tier control the original design lacked:

| Naming success (gated cell) | 0.5B (n=5, UNDERPOWERED) | 1.5B (n=22) | 3B (n=8, UNDERPOWERED) |
|---|---|---|---|
| primed_late | 0 | 0 | 0 |
| control_late | 1 | 16 | 8 |
| D31 gate (control − primed) | +.200 [−.264, +.624] — not shown | **+.727 [+.471, +.868] — concept-specific** | +1.000 [+.541, +1.000] — concept-specific (UNDERPOWERED) |
| concept mass, control vs primed | .204 vs .000 | .658 vs .003 | .924 vs .000 |

*Source: `results/s4-avoidance-qwen2.5-*.json` (S4b cells superseded in place).*

Removing one direction at the late band third erases the model's ability to say that
word (naming 0/n, concept probability mass ≈ .000, every subject). On the powered
subject the switch is concept-specific and CI-clean; the specificity emerges with
scale — at 0.5B any late-tier single-direction removal wrecks output (control
survives 1/5, mass .204 vs clean .661), at 3B the control is untouched (8/8, mass
.924 ≈ clean .886). Middle-tier removals are benign for both vectors everywhere.

![S4b naming success on the gated cell: clean vs primed-late vs control-late](figures/fig-s4b-late-switch.png)

*Figure 6 — The late-band off-switch and its matched control: naming success on the
competence-gated cell with the recorded Wilson 95% intervals (n = 5/22/8; the 0.5B
and 3B cells carry their pre-declared UNDERPOWERED tags). Ablating the concept's own
late direction silences naming on every subject (primed late, 0/n); the same-category
control separates a per-concept switch (1.5B, 3B) from any-direction output damage
(0.5B). Plotted values and intervals are read verbatim from
`results/s4-avoidance-qwen2.5-{0.5b,1.5b,3b}-instruct.json`, `naming_success_gated`.*

### 4.10 Three findings that cut across stages

1. **The Jacobian transport earns its keep only sometimes.** For writing, it is the
   entire effect where writing works at all (M2's 3B +.116; S1's doubling; S2's α = 1
   routing; 3B introspection with a dead identity arm). For reading, it is
   content-dependent: a CI-clean advantage on order-ops at 3B (+.165) and
   multilingual at 0.5B (+.208), a CI-clean *reversal* on typo that deepens with
   scale (+.427 → −.271 → −.323) and on instructed category content at 3B (−.091).
2. **1.5B, not 3B, is where the strongest phenomena live.** The injected-thought
   dose–response (0 → 30/101, control exactly 0), the paper-exact S2 category order,
   the S3 graded ablation curve, and the S4 competence gate (22 vs 5 and 8) all peak
   at the middle subject; multihop readability does too (54.3% at 1.5B vs 39.4% at
   3B). Below 27B, scale is not a monotone knob.
3. **Where the paper's anchors apply, everything lands well below them** — roughly an
   order of magnitude on verbal report and broadcast, 2–8× on two-hop flips — while
   the *shape* of the phenomena (dose ordering, focus ≫ suppress with baseline ≈ 0,
   rise-then-saturate, mid-band localization, category ordering) tracks the paper
   wherever any signal exists at all.

## 5. Discussion

The thesis question — is the workspace readable at laptop scale? — has a clean
answer: not at the pre-registered bar, at any of three scales, with the most
workspace-like content (abstract, not-yet-verbalized) the hardest zero. The
escalation discipline matters here: the triple null closes the question for the
reachable range rather than leaving an "if only we'd tried one more scale" residue.

What makes the null worth reporting is that the band it concerns is demonstrably not
inert. The same lenses that cannot *read* the workspace at the bar still locate
directions whose removal selectively kills flexible reasoning (S3, the one full-gate
pass), whose injection produces dose-ordered, saturating, mid-band-localized verbal
reports at 1.5B (M1/S1), and whose late-band copies act as concept-specific output
switches (S4b). The honest synthesis is not "small models have no workspace" but
"small models show causal traces of the paper's workspace machinery an order of
magnitude below its anchors, while the readable, reportable surface the paper builds
on is absent at the frozen bar."

The standing falsification arm turned out to be the project's most productive
instrument. It killed claims (M0 typo; M3's 3B reversal), certified claims (M2's 3B
cell; S1's doubling; S2's α = 1 routing), and showed that "the Jacobian transport
helps" is not one fact but a per-content, per-direction family of facts. A
small-scale reproduction running only the J-lens arm would have gotten several of
these stories wrong, in both directions.

Finally, the re-scope discipline — pre-registering that a readability null converts
downstream milestones to descriptive characterization — is what lets this paper
report a dose–response and a full gate pass without quietly upgrading either to a
reproduction claim. The would-gate verdicts stand as frozen: does not modulate, does
not route, NOT shown. The structure inside them is reported beside, never instead
of, those sentences.

## 6. Threats to validity and limitations

- **The band could be misplaced at small scale.** The 38–92% proportional transplant
  was pre-registered to eliminate forking paths, at the owned cost that a workspace
  living elsewhere in a small model would read as a false null. This is the central
  un-eliminated alternative to "no readable workspace."
- **Fit corpus.** N = 100 WikiText prompts vs the paper's ~1000 generic web-text
  sequences (the paper's own saturation evidence motivated the cut; still an owned
  deviation). A richer corpus could conceivably lift readability.
- **Grading conventions the release does not ship are ours.** The intermediate→token
  mapping (single-token {word, ␣word} forms, the order-ops synonym table), the M3
  prompt frame, and the strength grids were all constructed here — pre-declared
  before results existed, but not the paper's own artifacts.
- **The falsification arm is `J = I`, not the paper's decomposition.** The paper
  contrasts J-space vs non-J-space *components*; our standing arm substitutes raw
  unembedding rows. Like-for-like within the project, an owned deviation from the
  paper's exact control.
- **Single-token filter.** Rank-based grading drops multi-token stimuli (9/90 two-hop
  items, 12/192 generalization trials, 0–6% of M0 intermediates); every drop is
  counted, but the measured sets are slightly smaller than shipped.
- **UNDERPOWERED cells stay that way.** S4/S4b gated cells at 0.5B (n = 5) and 3B
  (n = 8), all S3 targeted-arm cells (8 and 11 passages), and S2 per-template cells
  (n = 12) carry pre-declared tags and no gated claims.
- **Descriptive mode is not a loophole.** All downstream "findings" are
  characterizations of intervention behavior at scales where readability is null;
  none is a reproduction claim about the paper's workspace.
- **Hardware/numerics.** fp32 on MPS for everything except the 3B fit (rented CUDA;
  cross-device fp32 noise on a corpus-mean matrix is orders below any gate's
  sensitivity, and the fit log's per-prompt signature matched).
- **Selectivity is relative.** Heavy ablation changes 57–78% of ordinary
  predictions; the gate compares disruption, it does not certify surgical precision.

## 7. Reproducibility

Everything is local and deterministic. The repository ships the independent fitter
(`fitter.py`), the AGREE gate, all milestone runners with `--dry-run` validation
(wrong-arm inputs exit `VERDICT: INVALID`), the pre-committed invariant tests, and
the per-trial results JSONs this paper quotes. Fitted lenses are gitignored — refit
to reproduce (0.5B ≈ 71 min on M-series MPS; 1.5B ≈ 6 h; 3B needs a CUDA GPU,
`remote-fit-3b.sh`, ~$0.83 rented in our run — the project's only outside spend).
Stimuli come from cloning the reference repository into `refs/`; models pull from
HuggingFace on first use; no API keys.

```bash
git clone https://github.com/ksdisch/dim-stage && cd dim-stage
git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens
uv run pytest                          # analytic gate tests; no models needed
uv run fitter.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --out lenses/qwen2.5-0.5b-instruct-n100.pt
uv run m0_readability_gate.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --lens lenses/qwen2.5-0.5b-instruct-n100.pt --dry-run   # then without --dry-run
```

## References

1. Anthropic. *Verbalizable Representations Form a Global Workspace in Language
   Models.* Transformer Circuits, 2026-07-06.
   https://transformer-circuits.pub/2026/workspace/index.html — no arXiv ID exists;
   the canonical URL is the citation.
2. Anthropic. *jacobian-lens* (reference implementation and shipped experiment
   data). https://github.com/anthropics/jacobian-lens, Apache-2.0, marked "not
   maintained". Pinned as a dev-dependency for the cross-check only.
3. Subjects: Qwen2.5-0.5B/1.5B/3B-Instruct (Hugging Face model IDs
   `Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct`); fit corpus: WikiText-103 via the
   reference repository's loader convention. Wilson score intervals and Newcombe
   difference intervals are standard methods and are used here as the lineage's
   stats ruler (`stats.py`); no further sources are recorded in this repository.
