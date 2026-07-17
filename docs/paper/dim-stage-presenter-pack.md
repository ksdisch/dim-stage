# dim-stage presenter pack — defending the paper, claim by claim

*Companion to `dim-stage-paper.md`. Everything here is traceable to a repo file; the provenance table gives the exact path for every number you might be asked to show live.*

## The one-breath version

We rebuilt Anthropic's Jacobian lens from its published spec, proved our build bitwise-identical to the reference, and asked the paper's own open question — is the workspace readable below 27B? — at three laptop scales under gates frozen before any run. **The answer is no, three times, at the pre-registered bar.** Everything downstream is descriptive characterization of what *does* exist: a fainter, narrower, falsification-surviving trace of the same machinery.

## What you may claim — and the sentence to say

- **AGREE.** "My independent implementation computes the same object as Anthropic's reference — bitwise-identical matrices at every layer, 3,220/3,220 held-out readouts." Not "my lens is correct in some absolute sense" — correct *relative to the reference spec*.
- **M0.** "At the frozen bar — pass@10 Wilson lower bound ≥ .5 on ≥ 3 of 6 distributions — all three scales are NULL, and the two abstract-content distributions are hard zeros everywhere." Never soften the null; it is the headline, pre-registered as such.
- **M1 report.** "Swapping the model's ready answer barely moves its spoken report: .105–.175 versus the paper's .88." The J − I CIs all straddle zero, so claim *no* transport advantage here.
- **M1/S1 introspection.** "A steered-in thought becomes reportable at 1.5B specifically — 0 → 30/101 with dose, control exactly 0/101 — and S1 showed it's transport-specific (+.178 CI-clean over the no-transport arm), saturates like the paper's Figure 7, and lives in the band's middle third." This is the project's strongest *descriptive* finding — never call it a reproduction (readability is null; the gate that could have made it a claim was frozen before the data).
- **M2.** "Two-hop redirection mostly fails (.073–.286 vs .60) — but at 3B it works only through the transport: raw rows flip 0/43, J-lens 5/43, difference CI-clean." Arm 2 never gates; this is a descriptive contrast.
- **M3.** "The frozen gate says 'does not modulate' — math is a hard zero and the gate needs both families. Inside it, the category signal has the paper's exact ordering, is CI-clean at 1.5B/3B, grows with scale — and at 3B is read better *without* the transport." State the gate verdict first, always.
- **S2.** "'Does not route' at the frozen gate — but a CI-clean transport-specific routing signal exists at α=1 on all three subjects, and the paper's α=2 rescue *inverts* here: overdose makes the model blurt the argument instead of computing with it."
- **S3.** "The only property to clear its full pre-committed gate on all three subjects: targeted workspace ablation kills two-hop chains while random damage and WikiText prediction survive CI-cleanly above them." Immediately add the owned qualifier: *relative* selectivity — heavy ablation still changes 57–78% of ordinary predictions.
- **S4/S4b.** "NOT shown: the early-band suppression copy the paper describes is absent where the models can do the task at all. What does reproduce is the late-band output off-switch — and S4b's matched control made that concept-specific and CI-clean on the powered subject (0/22 vs 16/22, +.727)."

## Provenance table — claim → number → source file

| Claim | Number | Source |
|---|---|---|
| AGREE bitwise | rel-Frobenius 0.0 (tol 1e-3); 3220/3220, Wilson LB .9988 | `docs/M0-BRIEF.md` (AGREE table) |
| Readability NULL ×3 | 0/6 distributions each | `results/readability-*.json` (`verdict`); tables in `docs/M0-BRIEF.md` |
| Abstract hard zeros | association 0/99 ×3; poetry 0/98, 1/98, 0/98 | `docs/M0-BRIEF.md` tables |
| typo transport reversal | +.427 → −.271 → −.323, all CI-clean | `docs/M0-BRIEF.md`; `results/readability-qwen2.5-3b-instruct.json` |
| Verbal report | .175 [.112,.263] / .124 [.070,.208] / .105 [.056,.187] vs .88 | `results/verbal-report-*.json` (`pooled`); `docs/M1-BRIEF.md` |
| Introspection curve | 1.5B α=8: 30/101 = .297 [.217,.392]; α=0 = 0/101 ×3 | `results/introspection-*.json` (`summary`); `docs/M1-BRIEF.md` |
| S1 transport specificity | J−I +.178 [+.067,+.286] at α=8; plateau 30/30/29/31 | `results/s1-introspection-qwen2.5-1.5b-instruct.json` (`sweep_diff`) |
| S1 localization | L16–20: 29/101 vs full 31/101; full−mid +.020 [−.105,+.144] | same file (`localization`); `docs/S1-BRIEF.md` |
| M2 3B transport advantage | 5/43 vs 0/43; +.116 [+.011,+.245] | `results/two-hop-qwen2.5-3b-instruct.json` (`pooled.newcombe_intermediate_j_minus_identity`) |
| M3 gate + structure | "does not modulate" ×3; 3B focus−suppress +.082 [+.041,+.148]; J−logit −.091 [−.181,−.002] | `results/directed-modulation-*.json` (`would_gate`, `cells`); `docs/M3-BRIEF.md` |
| M3 baseline anchor | pooled 0/46, UB .077 ≤ .10, ×3 | `docs/M3-BRIEF.md` |
| S2 anchor cells | α=1: 17/16/18 of 180 (paper 76/192); α=2: 1/0/1 (paper 101/192) | `results/s2-generalization-*.json` (`pooled`); `docs/S2-BRIEF.md` |
| S2 J−I at α=1 | +.078 [+.031,+.131] / +.061 [+.012,+.114] / +.078 [+.029,+.132] | same files (`newcombe`) |
| S2 vocab floor | 1.5B median target rank 151,844.5 of 151,936 | per-trial records in `results/s2-generalization-qwen2.5-1.5b-instruct.json` |
| S3 gate legs ×3 | e.g. 1.5B: +.878 / +.244 / +.488, all CI-clean | `results/s3-selectivity-*.json` (`would_gate`); `docs/S3-BRIEF.md` |
| S3 relative-selectivity caveat | heavy WikiText match .223/.366/.428 | `docs/S3-BRIEF.md` |
| S4 gates + verdict | gated 5/22/8; leg 1 at 1.5B +.000 [−.149,+.149] | `results/s4-avoidance-*.json` (`would_gate`, `competence`) |
| S4b specificity | 1.5B +.727 [+.471,+.868]; 3B +1.000 [+.541,+1.000] (UNDERPOWERED); 0.5B not shown | same files (`late_switch_specificity`); `docs/S4-BRIEF.md` |
| $0.83 / 3B rescue | RTX 4090, ~57 min, 34.5 s/prompt, sha256-verified | `docs/DECISIONS.md` ("3B rescue") |

Note: the repo records per-trial JSONs, not rendered figures — if asked for a plot, offer to open the JSON and read the cell live; that is the stronger move anyway.

## Anticipated questions

**Q: Why is a null a result and not a failure?**
Because the bar, the wording, and the escalation plan existed before the data. The kickoff names readability the kill-risk and pre-declares "a null is a headline (answers the paper's open question)." A pre-committed null is information: below 3B, at this bar, the workspace is not readable — and the pre-registered 3B escalation makes it a *bracketed* no, not a shrug.

**Q: Why Wilson intervals? And what's Newcombe?**
Wilson score intervals are honest for small-N proportions — they never leave [0,1] and don't collapse at 0 or 1 hits (a hard requirement in a project full of 0/99 cells). Newcombe's method builds an interval for a *difference* of two proportions from the two Wilson intervals. The gates run on the Wilson *lower bound* — "even pessimistically, is the rate ≥ .5?" — and any difference whose Newcombe interval includes zero is reported as no gap. Bonus: the paper reports Wilson CIs natively, so comparisons are like-for-like.

**Q: If readability is null, why measure the downstream properties at all?**
The re-scope was pre-declared in the kickoff's risk plan: protocols and numbers unchanged, framing demoted from "reproduction" to "descriptive characterization." That's why we can report the 1.5B dose–response without contradiction — it's a described phenomenon, not a workspace-reproduction claim. And the descriptive layer is where every interesting structure (transport specificity, the α=2 inversion, the selectivity hold) was found.

**Q: What's the un-validatable residual?**
Three things, all owned. (1) The intervention and ablation operators have no reference code to diff against — the invariant gates catch sign/transpose/normalization bugs exactly, but cannot catch both implementations (ours and our reading of the spec) being wrong the same way. (2) Grading conventions the release doesn't ship — the synonym table, prompt frames, α grids — are ours; frozen pre-run, but ours. (3) S4's item set is constructed; the competence gate and pre-run freeze make it honest, not paper-identical.

**Q: You constructed S4's items — isn't that "inventing"?**
The reference ships no items for that experiment, and the paper's own items were constructed too. The honesty mechanisms are the same as the paper's: frozen construction rules (concepts only from vocabularies we'd already measured, leakage test-guarded, committed pre-run) and a competence gate that filters to items each subject demonstrably can do. That gate delivered the stage's first finding before any ablation ran: mostly only 1.5B can do exclusion at all.

**Q: Doesn't the rented-GPU fit invalidate the 3B comparison?**
It's a pre-declared fallback, not a silent move: byte-identical procedure (same fitter, same frozen prompts file, same dim_batch), fit-log signature matched to the MPS runs, lens transferred sha256-verified, grading done locally on MPS like the other two subjects. Cross-device fp32 noise on a corpus-averaged matrix is ~1e-7 relative — orders below any gate's sensitivity. Cost: $0.83, in the deviations table.

**Q: Why transplant the band proportionally instead of deriving it per-model, like the paper?**
The paper's band-finding diagnostics are qualitative and don't ship in the library. Deriving our own band and then choosing gates would be a forking path — tune the band until something reads. The proportional transplant has zero degrees of freedom; the cost (a misplaced band could depress readability) is owned in the deviations table. A descriptive diagnostics arm was pre-declared precisely so that a clearly different band would have been surfaced as an owned deviation row — no such row was ever recorded.

**Q: Which results would you defend as the project's strongest?**
S3: the only full-gate hold on all three subjects, with a causal operator, a random-direction control, and an automatic-task control — and its caveat pre-owned (relative, not surgical). Then the S1-hardened 1.5B dose–response: falsification-surviving, saturating, localized, with an exact-zero control.

**Q: What next?**
The recorded open items, not new inventions: the emergence question is only answerable above 3B (out of laptop range — rented-GPU territory); S4's sub-threshold early mass rise at 0.5B/3B is texture that a powered design could chase; and the project's standing wrap decision (continue vs close) is deliberately Kyle's, not scheduled.
