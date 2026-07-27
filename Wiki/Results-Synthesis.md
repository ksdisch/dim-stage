# Results Synthesis

## Purpose
Assembles the full evidentiary picture across M0, M1, M2, M3, S1, S2, S3, and S4/S4b — stage by stage, with gate verdicts and CI outcomes — in one place. No single brief holds the cross-stage story; the ROADMAP has one-liner headlines; the DECISIONS log embeds numbers in prose. This page is the one place a reader can see all eight stages together: what the gate said, what structure lived inside the verdict, and how the J-advantage story evolved.

Everything here is **descriptive characterization under the pre-registered re-scope**: M0 returned a triple NULL, so nothing downstream is a reproduction claim. See [docs/KICKOFF.md](../docs/KICKOFF.md) honesty contract.

## Key understanding

### The pre-registered readability verdict (M0)

**Fact** (from [docs/M0-BRIEF.md](../docs/M0-BRIEF.md) readability tables): the would-gate was Wilson 95% LB ≥ 0.5 on ≥ 3/6 lens-eval distributions. Results across all three subjects:

| Subject | J-lens pass@10 best cell | Verdict |
|---|---|---|
| Qwen2.5-0.5B-Instruct | typo 49.0%, LB .392 | **NULL (0/6)** |
| Qwen2.5-1.5B-Instruct | multihop 54.3%, LB .442 | **NULL (0/6)** |
| Qwen2.5-3B-Instruct | order-ops 45.9%, LB .368 | **NULL (0/6)** |

**Fact** (D4, [docs/DECISIONS.md](../docs/DECISIONS.md)): the highest Wilson lower bound across all 18 subject×distribution cells is 0.442 — closer to the bar than any other, but still below it. No distribution crosses the pre-registered floor at any scale.

**Inference**: the triple NULL is the headline that shaped everything downstream. It answers the paper's stated open question ("is the workspace readable at small scale?") for Qwen2.5 0.5B–3B with a clean, pre-registered *no*, and re-scopes M1–M3 to descriptive.

### Structure inside the M0 null (three scales)

**Fact** (from [docs/M0-BRIEF.md](../docs/M0-BRIEF.md), "What the structure of the null says"):

- **Abstract-content distributions (association, poetry) are hard zeros at all three scales** — the most workspace-like content (evoked concept, planned rhyme word) is unreadable as a rank-≤10 token under either lens at any scale.
- **Surface-adjacent content is sub-bar and non-monotone**: multihop peaks at 1.5B (54.3%) then regresses to 39.4% at 3B. order-ops is the one distribution that improves monotonically into 3B (33% → 36% → 46%).
- **J-advantage is content-dependent, not a scale trend**: typo's J-transport reversal deepens monotonically (0.5B +42.7pp → 1.5B −27.1pp → 3B −32.3pp; CI-clean at 1.5B and 3B); order-ops *regains* a CI-clean J-advantage at 3B (+16.5pp). These two trends run in opposite directions simultaneously.

### M1 — verbal report and introspection

**Fact** (from [docs/M1-BRIEF.md](../docs/M1-BRIEF.md) results tables; D7 gate): Arm-1 swap top-5 rates of .175 / .124 / .105 against the paper's .88 Claude anchor; all Newcombe J−I CIs straddle zero. Gate verdict: NULL (descriptive mode, triple readability re-scope applies).

**Fact** (introspection table, M1-BRIEF): report rate at α=8 (`default` prefill): 0/101 at 0.5B, **30/101 [.217, .392]** at 1.5B, 5/101 at 3B. The α=0 control is exactly 0/101 on every subject. The 1.5B–3B gap is CI-clean — the **project's first dose–response curve** and its strongest result entering the stretch season.

**Inference**: two protocols using the same band and same J-lens vectors produced opposite-sign findings on usefulness: swapping doesn't move the report; steering a thought in makes it reportable at 1.5B. "Interventions fail at small scale" is false; the truth is protocol-shaped.

### M2 — two-hop bridge swap

**Fact** (from [docs/DECISIONS.md](../docs/DECISIONS.md) D9–D11 outcomes): baseline accuracies 28/81, 41/81, 43/81 (0.5B/1.5B/3B). Baseline-conditioned primary-cell flip rates: **8/28 = .286 [.153, .471] / 3/41 = .073 [.025, .194] / 5/43 = .116 [.051, .245]** vs paper's .60 anchor. Gate: NULL (descriptive).

**Fact** (Arm-2, D10): identity rows flip **0/41 and 0/43** at 1.5B/3B. At 3B, J−I Newcombe **+.116 [+.011, +.245] excluding zero** — the **project's first CI-clean J-transport advantage for writing**.

**Fact** (answer-swap arm, D11): at 3B the answer-swap and intermediate-swap rates are equal (6 vs 5 of 43), with identity rows flipping nothing — no answer-smuggling signature.

### M3 — directed modulation

**Fact** (from [docs/DECISIONS.md](../docs/DECISIONS.md) D14 outcomes): pooled no-instruction baseline 0/46 on every subject, both arms (UB .077 ≤ .10 — the KICKOFF anchor reproduces). Math family is zero everywhere. Gate verdict: **"does not modulate" on all three subjects** (both families needed; math is a hard floor).

**Fact** (category family, J-lens focus): 2/110 → 6/110 → 9/110 (0.5B/1.5B/3B); focus − suppress CI-clean at 1.5B (+.055) and 3B (+.082); scale growth 0.5B→3B CI-clean (+.064 [+.004, +.131]).

**Fact** (logit-lens arm, 3B): focus rate 19/110 vs J-lens 9/110; J−logit CI-clean **−.091 [−.181, −.002]** — a CI-clean J-transport reversal: directed modulation at 3B is read better *without* the transport. This echoes M0's typo reversal, now on instructed content.

**Inference**: a genuine ordered signal exists inside the null verdict (focus ≫ control ≈ suppress ≈ baseline at 1.5B and 3B), qualitatively paper-shaped, but an order of magnitude below anchor level. Pre-registration is what keeps the interesting cell from silently becoming the gate.

### S1 — introspection dose–response hardened

**Fact** (from [docs/S1-BRIEF.md](../docs/S1-BRIEF.md) results; D15–D18): J-lens − J=I at α=8: **+.178 [+.067, +.286]**, CI-clean from α=1. J-lens ~2× raw-unembedding arm at the plateau (30–31 vs 12–14/101). **The project's first CI-clean J-transport advantage for reading/report.**

**Fact** (saturation, D17): J-lens rank-1 rate plateaus ~30/101 from α=8 across {8,12,16,24}; MRR keeps climbing (.067→.125). No subject collapsed (degeneracy guard silent).

**Fact** (layer localization, D18): middle third L16–20 alone recovers **29/101** (full band 31/101; full−mid CI +.020 [−.105, +.144], overlaps 0). J-transport advantage CI-clean at mid (+.129 [+.014, +.240]) and late (+.129 [+.041, +.219]) sub-bands.

**Fact** (3B cross-scale bonus): J-lens rises to 9/101 at α=16–24; identity arm 0–1/101; J−I CI-clean from α=8 (+.050 [+.003, +.111]). The 3B reporting signal, small as it is, is entirely transport-specific.

### S2 — flexible generalization

**Fact** (from [docs/DECISIONS.md](../docs/DECISIONS.md) D19–D22 outcomes): 180 gradable trials (192 − 12 filtered). Would-gate "does not route" on all three subjects: α=2 J-lens **1/0/1** of 180 vs paper's 101/192.

**Fact** (α=1 routing signal): J−I Newcombe CI-clean at α=1 on all three subjects: 0.5B **+.078 [+.031, +.131]**, 1.5B **+.061 [+.012, +.114]**, 3B **+.078 [+.029, +.132]** — a transport-specific routing signal that is ~10× below anchor but present at all scales.

**Fact** (α cliff): the paper's α=2 rescue inverts at small scale — hits collapse to ~0 from α=2; at 1.5B the target answer falls to median rank ~151,844 of 151,936 (vocab floor) while the swapped-in argument becomes the greedy output. Three distinct regimes: routed (α=1), blurted (α=2, guard silent), junk (3B α=8, guard fires once).

**Fact** (workspace loading predictor): 1.5B matches the paper's category order exactly (countries > months > animals > numbers); numbers loads lowest at every scale; at 3B the top end aligns (countries load highest and route best, .125 vs .04–.11 — "present ⇒ consumed" sharpens with scale while "absent ⇒ can't route" is scale-stable).

### S3 — selectivity

**Fact** (from [docs/DECISIONS.md](../docs/DECISIONS.md) D23–D26 outcomes): **selectivity-consistent on all three subjects — the only would-gate to HOLD everywhere** in this project.

| Subject | Leg i (heavy ablation vs baseline) | Leg ii (wikitext vs two-hop retention) | Leg iii (random vs J-ablation at medium) |
|---|---|---|---|
| 0.5B | +.964 (CI-clean) | +.187 (CI-clean) | +.536 (CI-clean) |
| 1.5B | +.878 (CI-clean) | +.244 (CI-clean) | +.488 (CI-clean) |
| 3B | +.930 (CI-clean) | +.358 (CI-clean) | +.395 (CI-clean) |

**Fact**: two-hop retention under heavy J-ablation: 1/28, 5/41, 3/43 (0.5B/1.5B/3B). Random control retains 16/28, 33/41, 34/43. Wikitext heavy-ablation top-1 match: .223/.366/.428 — far from the paper's "mostly intact" but CI-cleanly above the two-hop wreckage.

**Inference** (owned framing, from ROADMAP): the honest claim is *relative* selectivity — the flexible task is hit CI-cleanly harder than the automatic one, and targeted directions hit harder than random. The absolute damage is severe (heavy ablation changes 57–78% of ordinary predictions), not the paper's surgical precision.

### S4 / S4b — naming vs avoiding

**Fact** (from [docs/DECISIONS.md](../docs/DECISIONS.md) D27–D31 outcomes): competence gate passed by only 1.5B (22/60 items); 0.5B and 3B blurt the forbidden concept unablated on 17 and 13 of 60 items. Would-gate: **NOT shown on all three subjects** — early primed ablation raises avoidance failure nowhere (0/22 at 1.5B; all CIs straddle 0).

**Fact** (late-band switch): late-third k=1 ablation → naming 0/n and concept mass ≈ .000 at every scale — a hard output off-switch that reproduces across subjects without a gate claim (D28 scoped the control to early; late was uncontrolled texture until S4b).

**Fact** (S4b specificity, D31): same-category control at the late tier — at 1.5B: primed 0/22 vs control 16/22, **+.727 [+.471, +.868] CI-clean** — the off-switch is concept-specific on the powered subject; at 3B: 0/8 vs 8/8, CI-clean but UNDERPOWERED-tagged. Specificity emerges with scale (0.5B: any single-direction removal at the late tier breaks output; 3B: control untouched, mass .924 ≈ clean).

### The J-advantage arc across all stages

**Inference** (assembled from M0–S4b results):

| Stage | J-transport advantage direction |
|---|---|
| M0 Arm 2 (reading) | Mixed, content-dependent: +42.7pp on typo at 0.5B; CI-clean reversed at 1.5B/3B typo; CI-clean positive at 3B order-ops |
| M1 Arm 2 (swap) | None — all CIs straddle zero; at rank-1 raw rows beat J-lens on every subject |
| M2 Arm 2 (two-hop) | First CI-clean J-advantage for writing at 3B (+.116 [+.011, +.245]) |
| M3 Arm 2 (modulation reading) | CI-clean reversal at 3B (J-lens worse: −.091 [−.181, −.002]) |
| S1 Arm 2 (introspection report) | CI-clean J-advantage at 1.5B from α=1 (peak +.178 [+.067, +.286]); 3B purely transport at α>8 |
| S2 Arm 2 (generalization swap) | CI-clean J-advantage at α=1 on all three subjects (+.061–.078) |
| S3 Arm 2 (ablation, implicit) | Causal: J-ablation kills two-hop; random control does not |

**Inference**: no single direction. The transport helps introspective reporting (S1) and routing (S2 at α=1) but hurts or is neutral for read-lens typo (M0, M1), directed modulation reading at 3B (M3), and verbal-report swaps. Whether the transport adds value is content-type- and operation-type-dependent at these scales.

## Sources
- [docs/M0-BRIEF.md](../docs/M0-BRIEF.md) — readability tables, deviations table, structure-of-null section
- [docs/M1-BRIEF.md](../docs/M1-BRIEF.md) — verbal-report and introspection results
- [docs/DECISIONS.md](../docs/DECISIONS.md) — D1–D31: all outcome records for M2, M3, S1–S4b
- [docs/S1-BRIEF.md](../docs/S1-BRIEF.md) — S1 saturation, falsification, and localization tables
- [docs/ROADMAP.md](../docs/ROADMAP.md) — gate verdicts and one-line headlines per stage
- [docs/KICKOFF.md](../docs/KICKOFF.md) — quantitative paper anchors; honesty contract; success criteria

## Uncertainties & contradictions

- **Unresolved**: whether the M1 introspection dose–response's non-monotone scale pattern (peak at 1.5B, not 3B) reflects something about Qwen2.5 instruction-tuning at 1.5B specifically, or a more general phenomenon. The project did not investigate this.
- **Unresolved**: the M0 multihop apparent peak at 1.5B (54.3%) followed by regression to 3B (39.4%) uses the same 94 gradable items at all three scales, so it is not a sample-size artifact — its cause is undiagnosed.
- **Inference risk**: the "J-advantage arc" table above synthesizes results across very different operations (reading vs writing vs ablation vs steering). Comparing them suggests J-transport is content- and operation-type-dependent, but this is an inference from the pattern, not a tested claim.

## Related pages
- [Methodology-Guardrails](Methodology-Guardrails.md) — the statistical and pre-commitment system that produced these measurements
- [History](History.md) — chronological view of when each stage landed

## Relevance to current work
v1 is closed. These results are the project's permanent evidentiary record. Relevant for: portfolio presentation (the triple-NULL headline + S3's one HOLD + S1's J-transport characterization are the three defensible claims); any future follow-up would build on the J-advantage arc's content-dependence finding.

_Last reviewed: 2026-07-26_
