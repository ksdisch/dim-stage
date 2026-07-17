# S4 start-of-stage brief — naming vs avoiding (stretch)

*Start-of-stage brief per the per-stage rhythm. Design extraction first,
decisions second, code after Kyle freezes. Standing descriptive re-scope
applies (M0 triple readability NULL): every readout is characterization, not a
reproduction claim.*

## What S4 is, in plain terms

S3 showed the workspace band is load-bearing for flexible tasks. S4 tests the
paper's sharpest *within-task* version of that claim: **not saying something
takes workspace machinery that saying it does not.** In the psychology this is
the inclusion/exclusion paradigm — deliberately avoiding a primed response is
taken as the mark of conscious processing, because giving the primed response
can happen automatically.

The paper's experiment (appendix, Figure 69): a sentence *implies* a concept
without naming it ("Their trip included croissants, the Louvre, and a climb up
the famous iron tower" → France). Ask one of two questions — **naming** ("Which
country is the sentence describing?") or **avoidance** ("Name a country that
the sentence is NOT describing"). Then delete just that one concept's lens
direction (k = 1 — our S3 projection-removal operator, unchanged) at the
**early** third of the workspace band or the **late** third, and watch the
model's tendency to produce the implied concept:

- **Late-layer ablation** suppresses the concept under *both* instructions —
  late workspace layers look like "the intention to say this word."
- **Early-layer ablation** leaves naming intact but makes the model *blurt the
  concept it was told to avoid* (~5× the failure rate) — the early workspace
  copy is what the model uses to actively suppress a concept.

That double dissociation is the result. It is also the cheapest stage this
project has left: k = 1 ablations, short prompts, forward-only — minutes per
subject with S3's machinery.

## Design extraction (verbatim, source-cited)

Paper §"Different J-space requirements for naming a concept and avoiding it"
(appendix; Figure 69):

- Prompts constructed to imply a concept without naming it.
- Naming condition: "Which country is the sentence describing?" Avoidance
  condition: "Name a country that the sentence is NOT describing."
- **Competence gate**: restrict to items the model can do unablated — names
  correctly with probability ≥ 0.85 AND avoids correctly with probability
  ≤ 0.15. n = 63 items survive at Claude scale.
- Intervention: ablate **that concept's** J-lens vector (single direction) at
  the early workspace layers (L38–54) or the late ones (L75–92).
- **Specificity control**: identical ablation of a *different same-category*
  concept's vector — the effect must be concept-specific, not generic damage.
- Readout: the model's probability of producing the implied concept, per
  instruction × depth.
- **The reference repo ships NO item set for this experiment** (checked
  2026-07-17) — the n = 63 items are the paper's own construction and are not
  in `data/experiments/`. Whatever we run, the items are ours to source; this
  is the stage's load-bearing decision (D27).

## Decisions to freeze (Kyle picks; recommendations flagged)

### D27 — Item sourcing (the big one)

- **(a) Derive from M2's two-hop prompts.** Every shipped two-hop prompt
  already implies its bridge entity without naming it, and `swap_to` ships a
  same-category control concept. But the instruction needs a category noun
  ("name a *country*…") and the items don't carry one — we'd hand-annotate a
  90-row category-noun table, and the completion-style prompts sit awkwardly
  inside an instruction. Zero invented facts; clunky items.
- **(b) Construct a fresh item set by the paper's recipe.** Clue sentences
  implying one concept each, written for this stage and frozen in-repo before
  any run. Clean instructions, exactly the paper's design — but the project's
  first invented item set, a precedent step to own loudly.
- **(c) Constrained construction over measured vocabularies (recommended).**
  Option (b), but concepts drawn only from the categories whose vocabularies
  and per-subject knowledge this project has already measured (S2's countries
  / months / animals + M3's category lists), ~20 concepts × 3 clue sentences
  ≈ 60 items, control concept = another word from the same shipped list. The
  facts and vocabularies stay the reference's; only the clue sentences are
  ours; the paper's own competence gate (adapted, D29) does the honest
  filtering, exactly as it did for the paper's authors. *Why:* the paper's
  design needs clean category instructions, and its own items were
  constructed too — the gate, not the authorship, is what makes items honest.

### D28 — Ablation conventions

Depths = the S1 sub-band thirds of the frozen band (early / middle / late;
the paper contrasts early vs late — middle rides along as free texture).
k = 1 (the implied concept's own J-lens vector, M1's direction convention);
specificity control = the same-category alternative's vector through the
identical operator. S3's runtime read-back and degeneracy guard unchanged.
(Recommended as stated; there is no serious alternative that stays verbatim.)

### D29 — Readout + competence gate

- Prompt style: one user turn through the chat template (M1's D5 convention —
  these are instructions, not completions).
- Primary binary per trial: the implied concept's single-token form is the
  greedy next token (produced / not produced) — Wilson/Newcombe machinery
  unchanged. The concept's softmax probability mass is recorded as the
  paper-comparable texture (their .85/.15 thresholds are probabilities).
- Competence gate, two options: **verbatim** (P ≥ .85 naming, P ≤ .15
  avoidance — likely leaves few items at our scales) vs **greedy-based
  (recommended)**: unablated naming greedy-correct AND unablated avoidance
  greedy-avoids, with the verbatim-P pass-rate reported beside it. The gated
  set is the primary cell; n < 20 ⇒ pre-declared UNDERPOWERED (standing rule).

### D30 — Would-gate wording (pre-committed before any run)

**"Avoidance-dissociation-consistent"** iff, per subject, on the gated cell:
1. **Early ablation breaks avoidance:** avoidance-failure rate under early
   primed-concept ablation is CI-cleanly above the unablated avoidance-failure
   rate (Newcombe excludes 0).
2. **Early ablation spares naming:** naming success under early ablation shows
   *no* CI-clean drop vs unablated (a pre-declared null leg — honest only
   because it is stated before the run).
3. **The effect is concept-specific:** early primed-concept ablation raises
   avoidance failure CI-cleanly more than the same-category control ablation.

Late-layer suppression (the paper's "output intention" story) is reported as
descriptive texture, not gated. INVALID on wrong-arm inputs before any trial;
`--dry-run` validates and stops; `--limit` is smoke, never a result.

## Deviations table (starter)

| Deviation | From | Owned reason |
|---|---|---|
| Model scale 0.5B–3B vs Claude | paper | The project's standing frame |
| Constructed item set (no shipped one exists) | lineage precedent | The reference ships nothing; items frozen in-repo pre-run; competence gate filters (D27/D29) |
| Greedy-based competence gate primary | paper (.85/.15 P) | Small-model probabilities rarely reach .85; verbatim-P rates reported alongside |
| Greedy binary readout primary | paper (probability) | Deterministic-oracle rule; probability mass recorded as texture |

## Wall-clock plan

≤ ~60 items × 2 instructions × 5 conditions (clean, early, late, middle,
control-early) × 3 subjects, k = 1, short chat prompts — well under an hour
total, all local.

## What S4 does NOT decide

- No new model, fit, band, or operator; no re-litigation of the re-scope.
- Whether S4 is the last stage: Kyle's call at close (/seed-hunt or wrap).

## Results — S4 (all subjects, descriptive) — 2026-07-17

Full JSONs in `results/s4-avoidance-*.json`; logs `s4-avoidance-*.log`. All 60
items gradable (no single-token drops); read-back and degeneracy guard silent
throughout.

### The would-gate: **NOT shown on all three subjects**

| Leg (Newcombe 95%, gated cell) | 0.5B (n=5, UNDERPOWERED) | 1.5B (n=22) | 3B (n=8, UNDERPOWERED) |
|---|---|---|---|
| (1) early breaks avoidance | +.400 [−.118, +.769] fails | +.000 [−.149, +.149] fails | +.000 fails |
| (2) early spares naming (null leg) | holds (5/5) | holds (18/22; +.182 [−.002, +.385]) | holds (8/8) |
| (3) primed > control | +.200, fails | −.045, fails | +.000, fails |

### What actually happened, cell by cell

- **The competence gate is the first finding.** Clean naming succeeds on only
  20/28/21 of 60 items (greedy-first-token grading; models that answer "The
  …" are counted misses — owned readout caveat), and clean avoidance *fails*
  (concept blurted) on 17/6/13 of 60. Only 1.5B can reliably do the exclusion
  task — the same subject where M1/S1's introspection lives. Gates: 5 / 22 /
  8 items (verbatim-P .85/.15 would keep 1 / 15 / 6).
- **The late half of Figure 69 reproduces as a hard switch.** k = 1 ablation
  of the concept's vector at the late third drives naming to **0/5, 0/22,
  0/8** and mean concept mass to **.000 / .003 / .000** — on every subject,
  removing one direction at the late band third erases the model's ability to
  say that word. (Caveat, owned: D28 scoped the specificity control to the
  early tier, so the late switch has no matched late-control cell; a
  late-control is the obvious follow-up if this texture ever needs to be a
  claim.)
- **The early half — the suppression machinery — is absent.** Avoidance
  failure under early primed ablation: 2/5 (directional, CI straddles 0),
  **0/22**, **0/8**. The models that *can* avoid the concept keep avoiding it
  with the early workspace copy deleted. Naming under early ablation is
  spared everywhere (leg 2 holds), including the mild 18/22 at 1.5B.
- **Mass texture:** avoidance-condition concept mass ticks up under early
  ablation at 0.5B (.061 → .147) and 3B (.025 → .070) without flipping the
  greedy token — a sub-threshold trace of the paper's effect, texture only.

### One-line synthesis

At our scales the concept's late-band lens direction is a clean output
switch — delete it and the word cannot be said — but the paper's deeper
claim, an *early* workspace copy used to deliberately suppress the concept,
does not appear: where small models can do the exclusion task at all, they do
it without that machinery. Descriptive throughout; the gate said NOT shown
and that is the reported verdict.
