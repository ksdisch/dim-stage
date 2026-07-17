# S3 start-of-stage brief — selectivity (stretch)

*Start-of-stage brief per the per-stage rhythm. Design extraction first, decisions
second, code only after Kyle freezes. Standing descriptive re-scope applies (M0
triple readability NULL): every readout below is characterization, not a
reproduction claim.*

## What S3 is, in plain terms

Selectivity is the last of the paper's workspace properties this project hasn't
measured, and it is the *converse* of everything we've done so far. M1–M3, S1, and
S2 all asked: when a task needs the workspace, does the workspace deliver? S3 asks:
**what does the model do *without* the workspace?**

The paper's claim: the workspace is engaged only for *flexible* thinking — reporting
a fact out loud, or handing it to whatever operation the prompt invents — while
*automatic* processing (continuing text fluently, spotting an intrusion, wrapping
lines at the right column) uses the very same information without ever routing it
through the workspace. In the consciousness analogy this is the claim that most of
what a brain does is unconscious: only a thin slice of processing is "in the
workspace" at any moment.

Two kinds of evidence, and we can pursue either or both:

1. **Targeted (reading):** give the model one stimulus and vary only the task. Watch
   whether the intermediate fact (the passage's language; a character count) *loads
   into the lens readout* more when the task needs it flexibly than when it's used
   automatically. This is pure reading — our existing readout machinery, no new
   operator.
2. **Global (ablation):** *ablate* the workspace — surgically delete its contents at
   every position — and show that flexible reasoning breaks while ordinary text
   prediction survives. "Ablation" here means: find the top-k most strongly loaded
   lens directions at each position and zero out the residual stream's **projection**
   onto each (the projection is the part of the activation vector that points along
   that direction — we subtract exactly that component and leave everything
   perpendicular to it untouched). This needs one new operator, invariant-gated like
   the swap and steer operators before it (the D6 pattern).

One correction to the record: ROADMAP has carried the line "no paper anchor" for
selectivity. After design extraction that's only half-true. There are quantitative
paper reference points — the multihop-collapse-vs-pretraining-intact contrast of
Figure 22, and the presence/causation rates of Figures 20–21 — but they are all
Claude-scale numbers, used descriptively as every anchor since M0 has been. What's
genuinely absent is a small-model number and, for the targeted experiments, any
shipped intervention conventions (the repo's shipped metrics are reading-only).

## The findings we build on

- **M0:** the workspace is not *readable* at the frozen bar at any of our scales —
  which makes S3's question sharper, not moot: even an unreadable workspace can be
  *load-bearing* (M2's 3B J-only flips, S1's dose–response, S2's α=1 routing all
  showed causal traces the readability gate missed).
- **M2:** two-hop accuracy baselines exist per subject (.346 / .506 / .531 of 81
  gradable items) — the natural "flexible task" positive control for ablation, with
  our own clean baselines already measured.
- **S1:** at 1.5B the workspace's causal work localizes to the middle third of the
  band (L16–20 alone ≈ the full band). Any layer-banded ablation design should
  expect the middle to matter most.
- **S2:** the degeneracy guard (top-token share) and the D6 read-back invariant are
  proven furniture; both carry over unchanged.

## Design extraction (verbatim, source-cited)

### Paper §"The J-space selectively mediates flexible but not automatic cognition"

**Language experiment (Figure 20).** Eight short prose passages whose language is
evident but never stated. Four task conditions per passage: *continuation* (write
the next line), *anomaly detection* (a spliced foreign sentence — is anything out of
place?), *explicit report* (name the language), *flexible computation* (a fact about
the language not recoverable from the passage: famous author, word for "hello",
pre-Euro currency). Findings at Claude scale: the language label appears in lens
readouts **at comparable rates in all four conditions**, but a lens swap
(true language → alternative) flips the answer **only** in report and flexible
computation; continuation and anomaly detection are unmoved. Presence without
causation on the automatic tasks — that dissociation *is* the selectivity result.

**Line-count experiment (Figure 21).** Eleven passages, wrapped at a fixed column.
Conditions: *linewrap* (continue, preserving the wrap pattern — needs a running
character count), *explicit report* (how many characters in line one?), *flexible*
(first letter of that count spelled out — the count becomes an unspoken
intermediate). Here, unlike the language case, **presence itself varies with task**:
count tokens are absent from the lens under linewrap, moderate under direct report,
highest under first-letter. A 40s→60s clamped swap moves the two question answers,
never the wrap point. Information is "pulled into" the workspace on demand.

**J-space ablation (Figures 22–24).** At each token position, across a band of
layers: identify the k=10 most strongly activated J-lens vectors, zero the residual
stream's projection onto each, let the forward pass continue. **Exclusion rule:**
never ablate a token that appears in the clean forward pass's top-10 next-token
predictions (so the operator targets internal reasoning, not the model's intended
output). Three strengths differ in the **layer range** ablated (the experiential
section identifies L38–54 as "the first third of the workspace range", implying
strengths grow the range toward the full band). **Positive control:** multihop
accuracy, near-ceiling unablated, drops to near zero under heavy ablation.
**Targeting check:** over pretraining-like text, the ablated model's top-1
next-token prediction still matches the unablated model's at most positions.
**Falsification control:** a same-size random-direction ablation at the medium
range. The 14-task battery (Figure 24) is majority LLM-graded — **out of scope
permanently** per CLAUDE.md; only its deterministic members inform our menu.

### Reference repo (`refs/jacobian-lens/data/experiments/`)

**`selectivity-language.json`** — `passages[*]` = `{category, key, text}`, eight
passages, two per language (fr/de/es/it). `task.explicit_q` asks for a famous author
(answer graded against `authors[category]`, 11–12 surnames per language);
`task.automatic_q` is the neutral continuation control. `intermediates[category]`
are the label tokens tracked in the lens over the question tokens. **README metric:
explicit − automatic label-hit rate.** Note the repo ships a *two*-condition
reading contrast (no anomaly/currency conditions, no swap arm) — leaner than the
paper's four-condition causal version.

**`selectivity-linecount.json`** — 11 passages with per-passage `width`; wrap with
`textwrap.fill`; ground truth = character count of the first wrapped line.
`conditions.{none,direct,letter}` each pair a question with a prefill (e.g. `none`:
no question, prefill " The first line has"); the `continue` condition uses
`explicit_q` as instruction, no prefill. **Lens target set: any two-digit or English
number-word token in the top-k at any prompt position in the band. README metric:
that rate, contrasted across conditions.**

**Ablation:** no item set ships, and `jlens/` ships no ablation operator — same
situation as M1's intervention module (the reference is a cross-check oracle with
nothing to cross-check here), so **invariants replace the AGREE diff** again. The
multihop control maps onto M2's shipped two-hop set; a pretraining-like corpus we
must supply ourselves.

## Decisions to freeze (Kyle picks each; recommendations flagged)

### D23 — Scope bundle

- **(a) Targeted only.** Both repo item sets as reading contrasts, three subjects.
  Cheapest possible stage; no new operator; but it measures *presence*, and M0
  taught us presence is weak at our scales — risk of a foregone near-null.
- **(b) Ablation only.** The new projection operator + two-hop positive control +
  pretraining targeting check + random-direction control. The paper's strongest
  selectivity statement, and it probes *causation*, which is where every real
  finding of this project has lived.
- **(c) Both (recommended).** The targeted sets are nearly free once the runner
  skeleton exists (19 passages total, forward-only), and they supply the
  flexible-vs-automatic *framing* that makes the ablation result legible. S1/S2
  precedent: one stage, one property, all its cheap arms.

### D24 — Ablation conventions (if b/c)

- **k and exclusion rule:** k=10 and the clean-top-10 exclusion, both verbatim
  (recommended — zero-cost fidelity; deviations need a reason).
- **Strength tiers:** three options for the layer ranges.
  - *(i) Start-anchored growth (paper-faithful reading):* light = first third of the
    frozen band, medium = first two-thirds, heavy = full band (0.5B L9–L21,
    1.5B L11–L24, 3B L14–L32 — D2's proportional transplant).
  - *(ii) Mid-anchored growth:* light = middle third (S1 says that's where the causal
    work is at 1.5B), then grow outward.
  - **Recommendation: (i)** — replicate the paper's design and let S1's mid-band
    result be *tested* by it (if light start-anchored ablation does nothing while
    medium does, that's the S1 story again from the ablation side), rather than
    baked into it.
- **Random-direction control:** matched count (10), medium tier, fresh directions
  per position, orthogonalized against nothing (verbatim-spirit). Include
  (recommended — it's the paper's own falsification arm).

### D25 — Ablation eval sets (if b/c)

- **Flexible task:** reuse M2's two-hop item set and grading verbatim (recommended —
  our own unablated baselines already exist, so the ablation deltas are
  like-for-like within-project) vs. the paper battery's other deterministic members
  (MMLU etc. — new item sets, no in-project baseline, scope creep).
- **Automatic task:** top-1 match over fresh WikiText text **not used in lens
  fitting** (recommended; reusing `wikitext-n100-prompts.json` would entangle the
  eval with the fit corpus — if sampling fresh rows is awkward, reuse becomes an
  owned deviation row instead).

### D26 — Readout frame + underpowering

- **Targeted cells are tiny by construction** (8 and 11 passages). Options: extend
  the item sets (breaks the verbatim-set precedent of D19) or run them verbatim and
  **pre-declare every targeted cell UNDERPOWERED** per the standing N ≥ 20 rule,
  letting only the ablation cells (N = 81 gradable two-hop items, ~100 wikitext
  prompts × many positions) carry CI-gated statements. **Recommendation: verbatim +
  pre-declared UNDERPOWERED** — the targeted arms are texture, the ablation arm is
  the result.
- **Per-position vs per-prompt units** for the top-1 match rate: per-position
  (recommended; it's what the paper plots and N is large) with per-prompt as a
  robustness view.

## Pre-committed readouts (would-gate wording — descriptive mode)

Frozen before any real run; dry-run with rigged inputs must exit INVALID first.

- **"Selectivity-consistent"** requires all three, per subject: (i) heavy ablation
  drops two-hop primary-flip/accuracy CI-cleanly below the unablated baseline
  (Newcombe on the difference excludes 0); (ii) wikitext top-1 match under the same
  heavy ablation stays CI-cleanly **above** the two-hop retention rate (the
  disruption is targeted, not general); (iii) the random-direction control shows
  CI-cleanly less two-hop disruption than the J-space ablation at the matched tier.
- Failing (iii) is the headline *against*: "any-direction damage, not workspace
  selectivity."
- Targeted arms report explicit − automatic hit-rate differences with Newcombe CIs,
  tagged UNDERPOWERED (per D26), as texture only.
- Degeneracy guard (S2's top-token share) runs on every ablated generation;
  guard-fired cells are reported but excluded from CI claims.

## Deviations table (starter — grows as S3 runs)

| Deviation | From | Owned reason |
|---|---|---|
| Model scale 0.5B–3B vs Claude | paper | The project's standing frame |
| No LLM-graded battery tasks | paper Fig 24 | CLAUDE.md hard rule: deterministic oracles only |
| Two-hop set stands in for the battery's multihop | paper Fig 22/24 | In-project baselines exist (M2); like-for-like deltas |
| Invariants replace AGREE for the ablation operator | lineage bar | Reference ships no ablation code (M1-D6 precedent) |
| Repo's 2-condition language contrast vs paper's 4 | paper Fig 20 | Verbatim shipped item set (D19 precedent) |
| Linecount target set = number-words only (30 ids) | repo README | Qwen tokenizes multi-digit numbers digit-by-digit — the "two-digit token" half is empty on this tokenizer; single digits excluded (would swamp the readout) |
| Per-position clean-top-10 exclusion | paper §ablation | The paper's "top-10 tokens of a clean forward pass" read per position — the reading that scales to teacher-forced scoring |

## Wall-clock plan (forward passes only — no new fits, models, or bands)

Everything is forward-only with hooks; the expensive backward-pass machinery is not
involved. Targeted arms: 19 passages × ≤4 conditions × 3 subjects — minutes.
Ablation: 81 two-hop items × (3 tiers + 1 random control + 1 clean re-check) × 3
subjects, plus ~100 wikitext prompts × (3 tiers + control + clean) × 3 subjects —
comfortably an evening on MPS at M2/S2 per-forward rates, 3B included (local
forwards were fine throughout M2–S2; only *fitting* 3B needed the rented GPU).

## Frozen decisions (2026-07-17, Kyle — all recommendations)

- **D23** — both bundles: targeted reading contrasts + the ablation arms.
- **D24** — k=10 + clean-top-10 exclusion verbatim (per-position reading,
  owned); ablation = span-projection removal; start-anchored tiers (light =
  first third of the frozen band, medium = first two-thirds, heavy = full);
  random-direction control at medium, frozen seed 20260717. Selection per band
  layer × position by M0's lens-readout logits; directions by M1's raw-row
  convention.
- **D25** — flexible task = M2's two-hop items verbatim (81 gradable, plan-
  checked at startup); automatic task = fresh WikiText records 101–200 under
  the D3 rule, with a startup proof that the streamed first 100 still equal
  the recorded fit corpus (disjointness is checked, not assumed).
- **D26** — targeted cells verbatim + pre-declared UNDERPOWERED; presence =
  target token in lens top-10 at any band layer; only ablation cells carry
  CI-gated statements, per the three-leg would-gate above.

Full rationale in `DECISIONS.md`. Implementation: `intervention.ablate` +
`s3_selectivity.py`; pre-committed gates in `test_selectivity.py` (16 checks:
analytic-oracle projection invariants, exclusion honoring, rigged-operator
read-back INVALID, wrong-arm lens INVALID).

### Build-phase discoveries (before any result — owned in the table below)

- **Real lens direction sets defeated two textbook projections.** The top-10
  lens tokens at a (layer, position) can include near-duplicate and untrained
  reserved vocab tokens whose J-lens directions are numerically identical: a
  least-squares projection exploded (the runtime read-back caught it — its
  first catch), and LAPACK's iterative SVD then refused to converge outright.
  The shipped operator uses modified Gram-Schmidt with re-orthogonalization,
  which cannot fail either way; the read-back stayed on for every real run.
- **A silent MPS transfer bug.** `tensor.to("cpu", torch.float64)` in one
  step silently corrupts values coming off MPS (measured ~5 abs error on
  unit-scale data, no exception). Fixed as move-then-cast; recorded in
  project memory for every future stage.
- **Qwen digit tokenization empties half the linecount target set.** The
  README's "any two-digit token" half is vacuous on Qwen (multi-digit numbers
  tokenize digit-by-digit); the tracked set is number-words only (30 ids).
  Answer texture for the direct condition falls back to first-sub-token rank.

## What S3 does NOT decide

- No new model, no new fit, no band changes, no re-litigation of the descriptive
  re-scope.
- The experiential-report ablation (paper Figures 25–26) and anything LLM-graded:
  permanently out.
- The Figure 69 avoidance/inclusion appendix experiment: not in this stage; it can
  be a future menu item if S3's operator lands and Kyle wants a third stretch.
- Whether S3 is the last stage: Kyle's call at stage close, via /seed-hunt or a
  wrap decision — not pre-committed here.
