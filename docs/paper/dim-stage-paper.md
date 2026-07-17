# Is the Global Workspace Readable at Laptop Scale? A Pre-Registered Reproduction and Measurement of Jacobian-Lens Workspace Properties in Qwen2.5 0.5B–3B

*dim-stage project paper · 2026-07-17 · repository: `ksdisch/dim-stage`*

## Abstract

Anthropic's workspace paper reports that a **Jacobian lens** — a per-layer linear map that transports a transformer's mid-layer residual-stream activations into the final layer's basis before decoding them with the model's own unembedding — reads out a sparse, mid-layer "global workspace" that the model verbally reports, computes with, and steers on instruction. The method is demonstrated at 27B scale and above; whether any of it survives at laptop scale is named an open question. We independently implemented the lens from the published specification, validated it against the reference implementation to bitwise identity (per-layer relative Frobenius distance exactly 0; 3,220/3,220 held-out readout agreement), and measured workspace readability plus seven downstream properties on Qwen2.5-0.5B/1.5B/3B-Instruct under pre-registered, code-frozen gates: Wilson 95% intervals per cell, Newcombe 95% intervals on differences, N ≥ 20 per cell or the verdict is pre-declared underpowered. The headline is a pre-registered null: 0 of 6 evaluation distributions reach the readability bar at any of the three scales, so every downstream verdict is descriptive characterization, not a reproduction claim. Within that frame: workspace swaps do not move the spoken report (top-5 rates .105–.175 against the paper's .88 anchor); an injected thought becomes reportable at 1.5B specifically (0 → 30/101 with dose; control exactly 0/101), transport-specifically (+.178 [+.067, +.286] over the no-transport arm), saturating and localizing to the band's middle third; two-hop answer flips at 3B occur only through the Jacobian transport (+.116 [+.011, +.245]); and the paper's flexible-versus-automatic selectivity dissociation clears its full pre-committed gate on all three subjects — the only property to do so. Every null is reported at its gate; every deviation from the paper is owned.

## 1 Introduction

Large-model interpretability results rarely come with an answer to the question a hobbyist or an independent researcher most needs: *does any of this survive on hardware I can afford?* The workspace paper ("Verbalizable Representations Form a Global Workspace in Language Models", Anthropic, 2026) makes a set of striking claims — that a linear lens reads out reportable, swappable, steerable content from a contiguous mid-layer band — but demonstrates them on Claude-scale models and a 27B open-weights model. Below 27B is explicitly unexplored territory, and the paper names small-scale readability as an open question.

This project's contribution is deliberately narrow and, we hope, honest: **we reproduced and measured a published finding at scales the paper does not cover — here is the narrow, measured slice.** Nothing here is invented. Concretely:

1. **An independent implementation** of the Jacobian-lens fitting procedure, built from the paper and the reference package's docstring specification, and validated against the reference implementation on identical model and prompts — the two codebases produce bitwise-identical lens matrices at every layer.
2. **A pre-registered readability verdict** at three scales (Qwen2.5 0.5B, 1.5B, 3B Instruct), with the gate's wording and thresholds frozen as code before any measurement ran. The verdict is NULL at all three scales. Under the project's pre-declared risk plan, this null was always a reportable headline, not a failure mode — and it re-scopes everything downstream from "reproduction" to "descriptive characterization."
3. **Descriptive measurements of seven downstream workspace properties** — verbal report, injected-thought introspection (with a dedicated hardening stage), two-hop swap, directed modulation, flexible generalization ("broadcast"), selectivity, and the naming-versus-avoiding dissociation — each against the paper's own anchors, each with a standing falsification arm, each gated by confidence intervals frozen before the run.

The methodological stance matters as much as the numbers. Every verdict gate in this project was committed as code, dry-run against wrong-arm inputs (which must exit INVALID), and only then run for real. A result whose interval overlaps its comparison cell is not claimed. A cell below N = 20 is pre-declared UNDERPOWERED and reported as texture. Where the paper's released materials ship no procedure (grading conventions, intervention code, ablation code, one item set), our substitutes are declared, frozen before first use, and listed as owned deviations. The intended failure mode of this paper is to under-claim.

## 2 The measured object

Some vocabulary, defined once. The **residual stream** is the running vector each transformer layer reads and adds to — one vector per token position. The **unembedding** is the model's final matrix, mapping the last-layer residual vector to one score (**logit**) per vocabulary token. The **logit lens** is the established trick of applying the unembedding directly to a *middle* layer's residual vector; it assumes mid-layers share the final layer's coordinate system, which they mostly do not. The **Jacobian lens** repairs this by first multiplying the mid-layer vector by `J_l` — the *average* Jacobian of the final layer's activations with respect to layer *l* — which transports it into the final layer's basis:

```
lens_l(h) = unembed( J_l · h ),    J_l = E[ ∂h_final / ∂h_l ]
```

`J_l` is estimated by backpropagation: inject a one-hot cotangent at every valid target position at once, backprop, and accumulate row by row; the expectation runs over prompts, source positions, and current-and-future target positions (the reduction the reference package documents as the paper's). The paper reports that what this lens reads out in a **workspace band** — roughly 38–92% of network depth — behaves like a global workspace: verbally reportable, swappable mid-computation, and modulated by instruction.

The paper's claims come with quantitative anchors recorded in this project's kickoff brief: 88% top-5 verbal report of a swapped-in concept (N=90); 54–70% two-hop answer flips across Claude tiers (60% on the shipped 90-prompt set for Sonnet 4.5); a no-instruction modulation baseline ≈ 0; and generalization swap success of 76/192 at unit strength rising to 101/192 at double strength.

## 3 Methods

### 3.1 Independent build and the AGREE gate

The lens fitter was written independently from the specification, then cross-checked against the reference implementation (`anthropics/jacobian-lens`, pinned as a dev-dependency used by the cross-check *only* — no measurement code imports it). The pre-frozen gate: per-layer relative Frobenius distance ≤ 10× the reference's measured run-to-run noise floor, with top-1 readout agreement (Wilson lower bound ≥ 0.95) as confirmation. Measured outcome: the noise floor is exactly 0 (the fp32 backward pass on Apple's MPS backend is bitwise deterministic on this machine, so the pre-declared 1e-4 stand-in carried the tolerance), and the independent build's matrices are **bitwise identical** to the reference at every layer — relative Frobenius distance 0.0 against a 1e-3 tolerance, readout agreement 3,220/3,220 held-out cells (Wilson LB .9988). Two independently written codebases hitting the same bits is what a correctly extracted estimator specification looks like.

### 3.2 Deterministic oracles

Every outcome in this project is a logit ranking read from tensors: rank-1 hits over the workspace band, greedy next-token identity, swap grading as the rank of the swapped-in candidate — the conventions in the reference repository's data READMEs. There are no LLM judges and no text parsing anywhere in the harness.

### 3.3 The statistical ruler

Cells receive **Wilson 95% score intervals**; between-cell differences receive **Newcombe 95% intervals** (the standard method for a difference of two proportions built from the two Wilson intervals). The paper reports Wilson intervals natively, so anchor comparisons are like-for-like. A difference whose Newcombe interval includes zero is reported as no measurable gap — never rounded up to a win. Cells below N = 20 are pre-declared UNDERPOWERED and can only be texture. Trials cost ≈ $0, so N is bounded by wall-clock, not money.

### 3.4 Pre-registration discipline

Every gate — including its verdict *wording* — was frozen in code before its first real run, and dry-run against deliberately wrong inputs, which must exit INVALID. Analysis conventions with any freedom in them (band placement, synonym tables, steering-strength grids, item construction rules) were frozen before results existed — each is a forking path that could otherwise be tuned until something passed. The sharpest example: the paper derives its band per-model from diagnostics that don't ship in the reference library, so we pre-registered a **proportional transplant** of its 38–92% depth band (0.5B: L9–L21; 1.5B: L11–L24; 3B: L14–L32, frozen and merged before its lens existed).

### 3.5 The standing falsification arm

Every experiment carries a second arm with the Jacobian transport removed — `J = I`, the plain logit lens, using raw unembedding rows as concept directions. Any claim that the *workspace* (as opposed to "any token-derived direction") does something must beat this arm with a Newcombe interval excluding zero. As Sections 5–6 show, this arm cuts both ways: it falsified several would-be claims and certified three real ones.

### 3.6 Runtime correctness gates

The reference ships no intervention or ablation code, so no AGREE-style diff was possible for the operators used in M1–S4. In place of it: **pre-committed invariant tests** — a rigged analytic subject whose Jacobian is known exactly, with every post-patch logit asserted to the bit; exact null-ops (zero-strength steering, self-swaps); and a coordinate read-back re-verified at runtime on *every real application*, with the run declared INVALID on any failure. A degeneracy guard flags dose levels where the model collapses to a fixed output (attractor share ≥ 0.5 with a token different from the control's), so saturation is never confused with breakage. These guards did real work: during the S3 build the runtime read-back caught a silent least-squares blow-up on ill-conditioned real direction sets, an SVD non-convergence, and a silent MPS device-transfer corruption bug — before any result was recorded.

## 4 Experimental setup

**Subjects.** Qwen2.5-0.5B-Instruct, -1.5B-Instruct, and -3B-Instruct. The two smaller subjects were the planned scope; the 3B escalation had a pre-registered trigger (both smaller subjects null on readability), which fired. Instruct variants because the stimuli are chat-format and the lens must be fit on the model being read.

**Hardware.** A 24 GB unified-memory Apple-silicon laptop, fp32 on MPS, ≈ $0 per trial. The one exception is owned: 3B fp32 backward exceeds the machine's measured working-set ceiling (a ~25× slowdown cliff at every batching setting), so the 3B lens was fitted on a rented RTX 4090 (CUDA fp32, ~57 minutes at 34.5 s/prompt, total spend **$0.83**), transferred back checksum-verified, and graded locally like the other two. Cross-device fp32 noise on a corpus-averaged matrix is orders below any gate's sensitivity.

**Fit corpus.** WikiText-103 via the reference's own deterministic loader convention, N = 100 prompts of ≤ 128 tokens (the paper uses ~1,000 pretraining-like sequences; its §9.3 reports lens quality saturating quickly with corpus size). An owned deviation.

**Stimuli.** All item sets are the paper's own, shipped in the reference repository — with one disclosed exception: S4's naming-versus-avoiding items do not ship, so a 60-item set (20 concepts × 3 clues) was **constructed** — the project's first — under frozen rules (concepts only from vocabularies already measured in earlier stages; no clue contains its concept or a derivative, test-guarded), committed before any run. A mechanical single-token pre-filter under Qwen's tokenizer drops stimuli whose targets have no single-token form; every drop is counted (94–100% of intermediates survive in M0; 9/90 items in M2; 12/192 in S2; 0 in M1-introspection, M3, S4).

**Figures.** The repository records per-trial JSON files rather than rendered figures; all results below are tabulated directly from those files and the stage briefs that hand-transcribe them.

## 5 Results

The verdict summary. Every gate's wording was frozen before its run; "descriptive" means the pre-registered re-scope after M0's null applies — identical protocols and numbers, no property claims either way.

| Stage | Property | Frozen-gate verdict | The structure inside |
|---|---|---|---|
| M0 | Readability | **NULL × 3** (0/6 distributions each) | abstract content hard-zero everywhere; J-advantage content-dependent |
| M1 | Verbal report (swap) | far below anchor (descriptive) | .175/.124/.105 vs .88; J−I straddles 0 everywhere |
| M1 | Introspection (steer) | descriptive | dose–response at 1.5B only: 0 → 30/101, control exactly 0/101 |
| M2 | Two-hop swap | far below anchor (descriptive) | at 3B works *only* through the transport (+.116 CI-clean) |
| M3 | Directed modulation | **"does not modulate" × 3** | real, ordered, scale-growing category signal; 3B reads it better *without* the transport |
| S1 | Dose–response hardening | descriptive | transport-specific, saturating, mid-band-localized |
| S2 | Generalization / broadcast | **"does not route" × 3** | CI-clean transport-specific routing at α=1; the paper's α=2 rescue *inverts* |
| S3 | Selectivity | **selectivity-consistent × 3** — the only full-gate HOLD | relative, not surgical: heavy ablation still changes 57–78% of ordinary predictions |
| S4 | Naming vs avoiding | **NOT shown × 3** | early suppression machinery absent; late off-switch present |
| S4b | Late-switch specificity | concept-**specific** on the powered subject (1.5B) | specificity emerges with scale |

### 5.1 M0 — readability: a three-scale, pre-registered null

The gate (frozen as D4): a subject READS iff the J-lens pass@10 Wilson 95% lower bound is ≥ 0.5 on at least 3 of the 6 shipped evaluation distributions. Chance under the ~150k-token vocabulary is ≈ 0.007%, so the bar is generous but pre-declared. Outcome: **0 of 6 distributions pass on 0.5B, on 1.5B, and on 3B.** The pre-declared kill-risk fired, the pre-registered escalation to 3B was triggered by the double null, and the escalation returned null as well: no emergence point exists in the reachable range.

J-lens pass@10 per distribution (hits/N):

| Distribution | 0.5B | 1.5B | 3B |
|---|---|---|---|
| association | 0/99 | 0/99 | 0/99 |
| poetry | 0/98 | 1/98 | 0/98 |
| multihop | 35/94 (37.2%) | 51/94 (54.3%) | 37/94 (39.4%) |
| multilingual | 149/414 (36.0%) | 155/414 (37.4%) | 148/414 (35.7%) |
| order-ops | 36/109 (33.0%) | 39/109 (35.8%) | 50/109 (45.9%) |
| typo | 47/96 (49.0%) | 17/96 (17.7%) | 20/96 (20.8%) |

The null has structure. The two abstract-content distributions — association (an evoked, unnamed concept) and poetry (a planned rhyme word) — are hard zeros at every scale; these are precisely the most "workspace-like" contents. Surface-adjacent content is partially readable but sub-bar and *non-monotone in scale*: multihop peaks at 1.5B (54.3%) and regresses at 3B (39.4%); the closest any cell comes to the bar is order-ops at 3B, whose Wilson interval [.368, .552] crosses 0.5 only at its upper bound — it does not READ.

The second arm (J-lens minus logit lens, Newcombe 95%) shows that the Jacobian correction's value is **content-dependent, not a scale trend**. On typo it inverts with scale: +.427 [+.309, +.531] at 0.5B, −.271 [−.389, −.141] at 1.5B, −.323 [−.442, −.188] at 3B — the transport increasingly moves readable surface content *out* of the readable directions while the plain logit lens climbs from 6.3% to 53.1%. On order-ops the opposite: a CI-clean J-advantage *returns* at 3B (+.165 [+.037, +.286]). Multilingual's 0.5B advantage (+.208 [+.149, +.265]) vanishes by 1.5B.

**Consequence, pre-declared:** the downstream gates were premised on a readable workspace, so all downstream verdicts become descriptive characterization. This re-scope was written into the kickoff's risk plan before M0 ran.

### 5.2 M1 — verbal report and introspection

**Verbal report (swap).** The paper's protocol: swap the concept the model is about to say, mid-forward, and grade whether the swapped-in candidate reaches the top-5 of the spoken report. Anchor: .88 (Claude Sonnet 4.5, N=90). Our pooled rates: **.175 [.112, .263] (17/97), .124 [.070, .208] (11/89), .105 [.056, .187] (9/86)** at 0.5B/1.5B/3B. The report does not follow the swap at these scales. The falsification arm agrees: J − I differences are +.041 [−.062, +.144], +.034 [−.060, +.129], −.012 [−.109, +.086] — all straddling zero, so no transport advantage is claimed for writing here.

**Verbal introspection (steer).** Steer a concept's lens vector into the band while the model is asked what it is thinking of, and grade whether the concept becomes the rank-1 report at the open quote. Report rate by steering strength α (in mean-residual-norm units):

| α | 0.5B | 1.5B | 3B |
|---|---|---|---|
| 0 (control) | 0/101 | 0/101 | 0/101 |
| 0.5 | 0/101 | 7/101 | 2/101 |
| 1 | 0/101 | 16/101 | 2/101 |
| 2 | 0/101 | 23/101 | 4/101 |
| 4 | 0/101 | 26/101 | 4/101 |
| 8 | 0/101 | **30/101 = .297 [.217, .392]** | 5/101 = .050 [.021, .111] |

This is the project's first dose–response curve, and it lives at 1.5B specifically. The α = 0 control is exactly 0/101 on every subject — the prompt never begs the concept into the answer — and the 1.5B–3B gap at α = 8 is CI-clean ([.217, .392] vs [.021, .111]). Steering moves ranks on every subject (median steered rank, control → α=8: 3747 → 1322 at 0.5B; 4430 → 15 at 1.5B; 3791 → 382 at 3B); only 1.5B converts movement into actual reports.

### 5.3 M2 — two-hop swap

Swap the latent bridge entity of a two-hop chain ("the country where the Amazon ends" → *Brazil* → Portuguese; swap Brazil → Mexico) and grade top-1 answer flips. The primary cell conditions on chains the model answered correctly unswapped (baselines: 28/81, 41/81, 43/81 — Claude's is near-perfect, ours is not, and an unconditioned rate would conflate "no redirection" with "no chain to redirect").

| Subject | Baseline | Flips, J-lens (Wilson 95%) | Flips, J = I | J − I (Newcombe 95%) |
|---|---|---|---|---|
| 0.5B | 28/81 = .346 | 8/28 = .286 [.153, .471] | 4/28 | +.143 [−.075, +.347] |
| 1.5B | 41/81 = .506 | 3/41 = .073 [.025, .194] | 0/41 | +.073 [−.025, +.194] |
| 3B | 43/81 = .531 | 5/43 = .116 [.051, .245] | 0/43 | **+.116 [+.011, +.245]** |

Against the .60 anchor, redirection mostly fails. But the 3B cell is the project's first CI-clean Jacobian-transport advantage for *writing*: raw unembedding rows flip nothing (0/43) while J-lens vectors flip 5/43, and the difference excludes zero. The paper's named confound — answer-smuggling — was probed with a comparison arm swapping the answers directly: 2× the intermediate rate at 0.5B (16 vs 8) but equal at 3B (6 vs 5), and a smuggled answer would survive J = I, which flips nothing there. The swap disturbs chains far more than it redirects them: the original answer is displaced on ~40% of primary trials (14/28, 12/41, 18/43).

### 5.4 M3 — directed modulation

A *reading* experiment: instructed to think about (or avoid thinking about) a category while copying an unrelated sentence, does the target surface in the lens over the carrier text? The frozen gate required a CI-clean focus-minus-suppress contrast on **both** shipped task families. The math family is a hard zero everywhere (focus 0/120, both arms, every scale), so the gate reads **"does not modulate" on all three subjects** — and that verdict stands unedited.

Inside it, the category family shows the paper's exact qualitative ordering — focus ≫ control ≈ suppress ≈ baseline ≈ 0:

| Subject | focus (J-lens) | control | suppress | focus − suppress | focus − control |
|---|---|---|---|---|---|
| 0.5B | 2/110 | 0/132 | 1/286 | +.015 [−.006, +.060] | +.018 [−.013, +.064] |
| 1.5B | 6/110 | 0/132 | 0/286 | **+.055 [+.022, +.114]** | **+.055 [+.014, +.114]** |
| 3B | 9/110 | 0/132 | 0/286 | **+.082 [+.041, +.148]** | **+.082 [+.034, +.148]** |

The kickoff-cited anchor reproduces: the no-instruction baseline is 0/46 pooled (Wilson UB .077, under the pre-declared .10) on every subject, both arms. The signal grows CI-cleanly with scale (3B − 0.5B: +.064 [+.004, +.131]). Two divergences from the paper: bare mention does not prime at these scales (control 0–4/132 everywhere, vs the paper's mention ≈ focus), and no white-bear effect is *testable* — nothing enters the workspace uninstructed, so suppression has nothing to visibly fail at; that is a non-test, not a null. Most strikingly, at 3B the plain logit lens reads the focused concept nearly twice as well as the J-lens (19/110 vs 9/110; J − logit **−.091 [−.181, −.002]**, itself contrast-clean against its own control cells) — a CI-clean transport *reversal* on instructed content.

### 5.5 S1 — hardening the dose–response

The 1.5B introspection curve was the project's only headline finding without a falsification arm. S1 gave it three tests. **(B) Transport specificity:** re-running the whole curve with J = I (steering along the raw unembedding row), the J-lens beats it CI-cleanly from α = 1 (+.109 [+.024, +.197]), peaking at α = 8 (**+.178 [+.067, +.286]**); at the plateau the transport roughly doubles reporting (30–31/101 vs 12–14/101). The curve is a J-transport effect, matching the paper's own specificity control. **(A) Saturation:** extending the α grid to {12, 16, 24}, the rank-1 rate plateaus at ~30/101 (30/30/29/31 across α = 8→24) while mean reciprocal rank keeps tightening (.067 → .125) — the paper's Figure-7 rise-then-saturate shape, with the degeneracy guard silent throughout (no model broke). **(C) Localization:** steering sub-band thirds separately, the middle five layers **L16–20 alone recover 29/101** against the full 14-layer band's 31/101 (full − mid +.020 [−.105, +.144], overlapping zero) — the paper's mid-layer "middle block", found at hobby scale. The extended grid also revealed that 3B's small reporting signal (→ 9/101) is *purely* transport-specific: its identity arm is essentially dead (0–1/101), J − I CI-clean from α = 8 (+.050 [+.003, +.111] to +.089 [+.034, +.161]). 0.5B is null on both arms.

### 5.6 S2 — flexible generalization ("broadcast")

The paper's boldest property: one identical lens-coordinate swap of a function's argument, clamped at every position, consumed correctly by sixteen different downstream computations (anchor: 76/192 at α = 1, rising to 101/192 at α = 2). On the verbatim item set (180 of 192 trials gradable under the standing single-token filter):

| Cell | 0.5B | 1.5B | 3B | Paper |
|---|---|---|---|---|
| α=1, J-lens | 17/180 | 16/180 | 18/180 | 76/192 (.396) |
| α=2, J-lens | 1/180 | 0/180 | 1/180 | 101/192 (.526) |
| α=1, J = I | 3/180 | 5/180 | 4/180 | — |
| J − I at α=1 | **+.078 [+.031, +.131]** | **+.061 [+.012, +.114]** | **+.078 [+.029, +.132]** | — |

The frozen would-gate ("routes" iff the α=2 pooled Wilson LB ≥ .5) reads **"does not route" on all three subjects.** But a routing signal exists and is CI-cleanly transport-specific at α = 1 on every subject — and then the paper's dose direction *inverts*. At α = 2, hits collapse (17→1, 16→0, 18→1): the greedy output becomes the swapped-in **argument itself** (" France", " China") rather than the function's answer, and the true target answer is anti-ranked toward the vocabulary floor (median rank over the 1.5B α=1 hits: 151,844.5 of 151,936). Overdose converts an argument-to-compute-with into an output-to-say. Category structure reproduces where it can: 1.5B matches the paper's order exactly (countries 10/48 > months 4/48 > animals 2/36 > numbers 0/48), and the paper's own predictor — workspace loading — puts numbers lowest at every scale and aligns at the top by 3B. A conditioned frame (both facts provably known unswapped) lifts routing to 13/42, 12/62, 16/56 — roughly 3–4× the unconditional rate — and exposes the numbers zero as a knowledge/pragmatics confound (0/16 unswapped diagonals on every subject).

### 5.7 S3 — selectivity: the gate that held

The converse property: what runs fine *without* the workspace? At every position, ablate the residual stream's projection onto the ten most active lens directions per band layer (a projection removal, with a clean-top-10 exclusion rule verbatim from the paper), and compare a flexible task (M2's two-hop chains) against an automatic one (ordinary WikiText next-token prediction), with a matched-count random-direction control. The pre-committed three-leg gate holds on **all three subjects** — the only property in the project to clear its full gate everywhere:

| Leg (Newcombe 95%) | 0.5B | 1.5B | 3B |
|---|---|---|---|
| (i) heavy ablation kills two-hop (clean − heavy) | +.964 [+.778, +.994] | +.878 [+.719, +.947] | +.930 [+.788, +.976] |
| (ii) WikiText survives above two-hop retention | +.187 [+.045, +.218] | +.244 [+.111, +.314] | +.358 [+.241, +.404] |
| (iii) random hurts less than targeted (medium tier) | +.536 [+.306, +.702] | +.488 [+.277, +.641] | +.395 [+.189, +.558] |

Two-hop retention traces the paper's graded dose curve at 1.5B (21 → 13 → 5 of 41 across light/medium/heavy) and sharper at 3B (27 → 17 → 3 of 43); 0.5B is a cliff (1/28 at every tier); random control retains 16/28, 33/41, 34/43. The owned qualifier: this is **relative** selectivity. Heavy ablation still changes 78%/63%/57% of ordinary next-token predictions (WikiText top-1 match .223/.366/.428) — nowhere near the paper's "mostly intact" — but the flexible task is hit CI-cleanly harder at every scale, which is the property under test. The two shipped targeted reading contrasts (8 and 11 passages, pre-declared UNDERPOWERED texture) add a small-scale signature: the passage's language enters the lens almost only when the task demands it (explicit 7/8, 8/8, 8/8 passages vs automatic 0/8, 2/8, 1/8) — presence itself is task-gated, where at Claude scale presence is constant and only causal role differs.

### 5.8 S4/S4b — naming versus avoiding

The paper's Figure-69 paradigm: a clue implies a concept; the model must name it, or name something else. The claim has two prongs — ablating the concept's single lens direction *late* in the band should break naming (an output intention), and ablating it *early* should break *avoiding* (a suppression copy the model consults to not say the word). Items were constructed (the reference ships none) and frozen pre-run; a competence gate filters to items each subject can demonstrably do both ways.

The competence gate is itself the first finding: only 1.5B reliably does the exclusion task (gated items 5 / **22** / 8 of 60; 0.5B and 3B blurt the forbidden concept unablated on 17 and 13 items). On the gated cells, the frozen would-gate reads **NOT shown on all three subjects**: early primed ablation raises avoidance failure nowhere (1.5B: +.000 [−.149, +.149]; the underpowered 0.5B and 3B cells also fail their legs), the pre-declared null leg (naming spared under early ablation) holds everywhere, and primed-beats-control fails everywhere. The early suppression machinery — the paper's actual claim — does not appear at these scales.

The late prong, by contrast, reproduces as a **hard switch**: late-third ablation of the one concept direction drives naming to 0/5, 0/22, 0/8 with concept probability mass ≈ .000/.003/.000. S4 owned this as uncontrolled texture (no matched late control existed); S4b added the matched same-category control at the middle and late tiers, with a pre-committed specificity gate, and re-ran all three subjects — every shared cell reproduced bit-for-bit. Result: on the powered subject the switch is **concept-specific, CI-clean** (1.5B: control-late naming 16/22 vs primed-late 0/22; +.727 [+.471, +.868]); 3B is concept-specific too (8/8 vs 0/8, +1.000 [+.541, +1.000]) but carries its UNDERPOWERED tag (n=8); 0.5B is NOT shown (+.200 [−.264, +.624], n=5) — at that scale *any* single late-tier removal wrecks output (control survives 1/5). Specificity emerges with scale: 3B's control cell is untouched (concept mass .924 vs clean .886).

## 6 Discussion

**The headline is the null, and the null is a result.** The readability gate was this project's pre-declared kill-risk, its bar and wording frozen before any lens existed, and its "no" was always defined as a reportable headline — the paper's own open question, answered for Qwen2.5 0.5B–3B with a clean, three-scale, pre-registered *no*. A null that survives a pre-registered escalation is stronger than the null that triggered it: if an emergence point exists, it is above 3B, outside the range a laptop-scale study can reach.

**The Jacobian transport earns its keep only sometimes.** The standing J = I arm produced three CI-clean certifications (M2's 3B flips exist only through the transport; S1's dose–response is transport-specific and roughly doubles the no-transport arm; S2's α=1 routing is transport-specific on all three subjects) and two CI-clean reversals (M0's typo reading, deepening with scale; M3's instructed content at 3B, read nearly twice as well by the plain logit lens). "Does the transport add value" is content-dependent, not a property of the method.

**Scale is not a monotone knob below 27B.** The strongest phenomenon in the project — injected-thought reportability — lives at 1.5B, with 3B CI-cleanly below it; multihop readability also peaks at 1.5B; only 1.5B reliably does S4's exclusion task. Other structures sharpen with scale instead (M3's focus signal, S2's loading predictor, S3's dose curve, S4b's switch specificity). A reproduction that assumed "bigger is closer to the paper" would have mis-designed itself.

**Where the paper's anchors apply, everything lands well below them — while the shapes track.** Verbal report sits at a fifth to an eighth of the anchor (.105–.175 vs .88); two-hop flips 2–8× below (.073–.286 vs .60); broadcast routing an order of magnitude under; modulation roughly an order below Claude's. Yet the *shapes* the paper predicts appear wherever any signal exists: dose-ordering, focus ≫ suppress ≫ baseline, rise-then-saturate, mid-band localization, category ordering, loading-predicts-routing. The one full-gate positive is S3's selectivity dissociation — flexible cognition dies under targeted workspace removal while random damage and automatic prediction survive CI-cleanly above it, at every scale — with the "relative, not surgical" qualifier owned.

**Small models can fail a phenomenon's precondition rather than the phenomenon.** No white-bear effect is measurable because nothing intrudes uninstructed; S4's suppression copy cannot be found in models that mostly cannot do the exclusion task; S2's numbers category cannot route facts the model never knew. Distinguishing a null from a non-test is, at these scales, a large fraction of the interpretive work.

## 7 Limitations and owned deviations

Every deviation from the paper is a recorded row in a stage brief; the load-bearing ones:

| Deviation | Ours vs paper | Status |
|---|---|---|
| Model scale | Qwen2.5 0.5B/1.5B/3B vs ≥27B / Claude | The thesis itself, by design |
| Fit corpus | WikiText-103, N=100 vs ~1,000 pretraining-like | Owned; paper's §9.3 reports fast saturation |
| Band selection | proportional 38–92% transplant vs per-model diagnostics | Pre-registered forking-paths guard; a misplaced band could depress readability |
| Single-token pre-filter | drops stimuli without single-token targets (counted everywhere) | Owned, mechanical |
| Hardware | fp32 MPS; 3B lens fit on rented CUDA ($0.83), checksum-verified | Owned; cross-device noise orders below gate sensitivity |
| Grading conventions the release doesn't ship | synonym table, prompt frames, α grids — frozen pre-run | Owned, pre-declared |
| No reference intervention/ablation code | invariant gates + runtime read-back instead of an AGREE diff | Owned; catches sign/transpose/normalization bugs, cannot catch a shared misreading of the spec |
| S4 item set | constructed (frozen pre-run, competence-gated) vs unshipped | Owned; the project's first constructed items |

Beyond the table: all downstream results are descriptive by pre-declaration, so nothing in Sections 5.2–5.8 is a reproduction claim in either direction; several S4/S4b cells are UNDERPOWERED and are labeled as such; and the S3 targeted contrasts are texture by pre-declaration (8 and 11 passages).

## 8 Conclusion

At the scales a laptop can reach, the global workspace of Anthropic's workspace paper is **not readable at the paper's own kind of bar** — a three-scale, pre-registered null on the paper's stated open question. What exists instead is a fainter, narrower, but repeatedly falsification-surviving trace of the same machinery: an injected thought that becomes reportable at one scale, through the transport specifically, in the middle of the band; a two-hop redirection that works at 3B only through the transport; a broadcast signal an order of magnitude under anchor that the paper's own dose prescription extinguishes; a selectivity dissociation that clears its full pre-committed gate on every subject; and a concept-specific output off-switch in the band's late third. The project's framing was fixed before its first number and survives its last: *reproduced and measured a published finding — here is the narrow, measured slice.*

## References

1. Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models," Transformer Circuits Thread, 2026-07-06. https://transformer-circuits.pub/2026/workspace/index.html (no arXiv ID exists; the canonical citation is the URL).
2. Anthropic, `jacobian-lens` reference implementation (Apache-2.0). https://github.com/anthropics/jacobian-lens
3. Statistical methods: Wilson score intervals for single proportions and Newcombe's score-interval method for differences of proportions are standard techniques; this project's implementations live in `stats.py` (ported from a prior project in the same series) and are pytest-covered.

*All numbers in this paper are transcribed from the repository's recorded results — the per-trial JSONs in `results/` and the stage briefs in `docs/` that transcribe them. A claim-by-claim provenance table appears in the companion presenter pack (`docs/paper/dim-stage-presenter-pack.md`).*
