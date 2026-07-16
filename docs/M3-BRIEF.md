# M3 start-of-stage brief — directed modulation

*2026-07-16 · status: options presented, decisions D12–D14 pending Kyle*
*Sources: the paper (`refs/workspace-paper.md`, §The J-space is subject to
directed modulation; Figure 10; appendix Figure 65) and the reference repo's
experiment data (`refs/jacobian-lens/data/experiments/README.md`
§directed-modulation, `directed-modulation.json`). Neither is committed;
refetch commands live in `readability.py`'s error messages.*

## What M3 is, in plain terms

Can the model **steer its own workspace on command?** We tell it *"Think about
citrus fruits while you write"* (or *"citrus fruits is irrelevant — ignore
it"*), teacher-force it through an unrelated **carrier sentence** ("The old
painting hung crookedly on the wall."), and point the J-lens at the carrier's
token positions — text that has nothing to do with fruit. If the workspace is
under top-down control, *orange* should surface in the lens under the focus
instruction, stay rare under the ignore instruction, and be absent with no
instruction at all. The paper also finds the human **"white bear" effect**:
telling a model *not* to think about something puts it in the workspace more
than saying nothing.

**One correction to the record:** M1's D8 rationale said the steering operator
was "required for M3." That was wrong — M3 is a **reading** milestone: the
modulation comes from the *instruction in the prompt*, and the lens only
*reads*. M3 reuses M0's readout machinery and M1's chat/teacher-forcing
construction; `intervention.py` sits this one out. (D8's other merits stand —
introspection was in KICKOFF scope and produced the 1.5B dose–response — but
that rationale line was mistaken and is owned here.)

## Design extraction (verbatim, source-cited)

**The protocol** (paper, §directed modulation):

> We test this with a protocol in which the model is given an instruction
> specifying what to hold in mind while copying a passage of text. We then
> apply the Jacobian lens at a token position in the model's output, where
> the surface text is unrelated to the mental instruction, and inspect the
> readout across layers.

> … we measure the rate at which the target concept appears in the Jacobian
> lens readout while the model is copying text … We compare three
> instruction conditions: positive instructions ("think about X"), negative
> instructions ("ignore X") and a no-instruction baseline. The baseline rate
> is approximately zero in all conditions …

**The grading** (Figure 10 caption, verbatim): "Category and math: a trial is
positive if the target reaches **J-lens top-1 at any (layer, position)**."

**The data** (reference README, verbatim):

> The model is given an instruction about a target X, then teacher-forced to
> write an unrelated carrier sentence; we check whether X surfaces in the
> lens readout over that response span. `phrasings` holds 24 instruction
> templates — each `text` has an `{x}` slot — in four `group`s, which
> `group_kind` collapses to {focus, suppress, control}. A trial pairs one
> phrasing with one entry from `carrier_sentences` and one target. Targets
> are drawn from `math_problems` (`expr` fills `{x}`; `answer` is the
> tracked token; `tier` is difficulty) or `topic_categories` (`name` fills
> `{x}`; every string in `members` is a tracked token).

Shipped shape: 24 phrasings (focus 5, mention 6, dismissal 6, negated-think 7;
`group_kind`: mention → control, dismissal + negated-think → suppress), 20
carriers (the paper's worked example is `carrier_sentences[0]`), 24 math
problems (tiers 1–4), 22 topic categories. The four groups preserve appendix
Figure 65's finding: **a bare mention primes almost as strongly as focus**;
"don't think about X" suppresses poorly (≈ mention) while "ignore"-style
dismissals suppress well below mention.

### Anchors

| Anchor | Value | Source |
|---|---|---|
| No-instruction baseline | **≈ 0** on every model — the KICKOFF-cited anchor | §directed modulation; KICKOFF M3 row |
| Focus condition | "a substantial fraction of trials," **increasing with model size** | §directed modulation (Figure 10 rates are figure-only — no numeric text anchor exists; ours reports absolute rates without a numeric target) |
| Ordering | focus > suppress > baseline ≈ 0; suppress > baseline (**white bear**); mention ≈ focus; dismissal < mention < negated-think ≲ mention | §directed modulation; Figure 65 |

### What the reference does NOT ship (checked 2026-07-16)

- **The line-width family's stimuli** — the third Figure 10 family wraps
  unreleased prose at random widths (the README describes the construction
  but ships no text, and it grades by a different metric: numeric precision).
  KICKOFF scopes the core three to "the paper's own shipped stimuli" (D12).
- **The prompt frame** joining instruction + copy request + carrier — the
  exact wording that asks the model to copy is unspecified; ours to freeze
  (D13, deviations row 2).
- **A no-instruction phrasing** — the baseline condition is constructed, not
  shipped: the copy request alone (D13).

## Mode: descriptive (settled)

Triple readability NULL → characterization, never reproduction claims; gate
wording still freezes now (D14). Worth naming: M3's protocol is the closest
cousin of M0's null result — it asks the lens to read *instructed* content at
the same band where *evoked* content was unreadable. The J = I logit-lens
falsification arm returns (a reading milestone again), free per trial.

## Decisions to freeze (Kyle picks each)

### D12 — Task families in scope

- **A. The two shipped families (recommended).** Category-instance + math-
  expression, verbatim from `directed-modulation.json`, plus the constructed
  no-instruction baseline. Line-width family OUT: no shipped stimuli, a
  different metric (top-1 numeric precision), and an RNG convention we would
  own end-to-end — the project's shipped-stimuli scope line (KICKOFF) draws
  itself. *Trade-off:* one of Figure 10's three panels goes unmeasured
  (deviations row 3).
- **B. Add line-width from WikiText.** Reproduce the construction from our
  D3 corpus (alpha-heavy prose, seeded widths). *Merit:* full Figure 10
  coverage. *Trade-off:* whole-cloth stimulus construction — the largest
  owned deviation in the project — for the family the paper itself grades
  differently; not recommended for v1.

### D13 — Trial construction and readout conventions

- **A. Deterministic full grid (recommended).**
  (1) Chat template (M1 precedent); user turn = the filled phrasing, a
  space, then the owned copy frame `Copy this sentence exactly: "<carrier>"`;
  baseline turn = the copy frame alone. (2) Assistant reply teacher-forced
  to the carrier verbatim (M1's `build_prompt` pattern); **readout span =
  the carrier's token positions** in the assistant turn. (3) J-lens at every
  band layer over the span; **hit iff a tracked token is the top-1 readout
  at any (layer, position)** — Figure 10's grading, verbatim. Tracked
  tokens: the math `answer` / every category `member`, each as its
  single-token {`w`, `␣w`} forms; targets with no tracked form drop and are
  counted. (4) Trials = every phrasing × every target (24 × 22 + 24 × 24 =
  1,104), carrier assigned deterministically as
  `carriers[(phrasing_index + target_index) mod 20]`; baseline = every
  target × `carriers[target_index mod 20]` (46 trials). No RNG anywhere.
  *Trade-off:* one carrier per pair (the paper's pairing is unstated —
  deviations row 4).
- **B. All-carriers grid.** Every pair × all 20 carriers (~22,000 forwards).
  *Merit:* averages out carrier idiosyncrasy. *Trade-off:* hours at 3B for
  a texture the 20-carrier rotation already samples; not recommended.

### D14 — Verdict conditions (frozen wording; descriptive mode)

- **A. Contrast-first, two arms (recommended).** Cells per family ×
  condition (focus / control / suppress / baseline) × arm (J-lens /
  logit-lens J = I): hit rate + Wilson 95%. Contrasts (Newcombe 95%):
  **focus − suppress** (the modulation contrast — the frozen would-gate:
  "modulates" iff its CI excludes 0 on **both** shipped families AND the
  baseline is clean, Wilson UB ≤ 0.10, the pre-declared reading of the
  paper's "approximately zero"); **suppress − baseline** (the white-bear
  check); **focus − control** (does instructing add anything over merely
  mentioning); J − logit per condition (the standing falsification arm).
  Dismissal vs negated-think reported separately (the Figure 65 split the
  shipped groups preserve). *Merit:* the phenomenon IS the contrast; no
  numeric absolute anchor exists in the text to gate against.
- **B. Gate on the focus rate alone** (absolute Wilson LB ≥ 0.5). *Merit:*
  one number. *Trade-off:* misses that modulation is a *contrast* claim —
  a model could hit 60% under every condition (pure priming, zero control)
  and "pass"; not recommended.

### Assumption on record

Subjects = all three; bands = the frozen table; per-trial readout uses the
same lens conventions as M0 (`readability.py` rank machinery). Costs nothing
to change before coding starts.

## Deviations table (starter)

| # | Paper / reference | Ours | Why | Status |
|---|---|---|---|---|
| 1 | Subjects: Claude tiers (Figure 10 spans sizes) | Qwen2.5 0.5B/1.5B/3B Instruct | The project's thesis | Owned, standing |
| 2 | Prompt frame joining instruction + copy request unspecified | `Copy this sentence exactly: "<carrier>"`, owned; baseline = frame alone | Not shipped | Owned, pre-declared |
| 3 | Three task families | Two (line-width stimuli unshipped; different metric) | KICKOFF shipped-stimuli scope | Owned, pre-declared (if D12 = A) |
| 4 | Phrasing × carrier × target pairing unstated | Deterministic rotation `(i + j) mod 20`; one carrier per pair | No RNG, reproducible | Owned, pre-declared |
| 5 | Claude chat format | Qwen chat template; readout span = teacher-forced carrier tokens | M1 precedent (deviations row 2 analog) | Owned, standing |
| 6 | Tracked tokens as shipped | Single-token {`w`, `␣w`} pre-filter; zero-form targets dropped + counted | Rank grading needs single tokens | Owned, mechanical |

## Wall-clock plan (from measured M1/M2 rates; forward passes only)

Per subject: 1,104 + 46 ≈ 1,150 chat-length forwards, each with a band-layer
lens readout over a ~10-token span — comparable per-forward cost to M1's
introspection run. Minutes at 0.5B, ~15–20 min at 3B on MPS. Serial GPU
discipline unchanged.

## What M3 does NOT decide

- Nothing here is harness code; the brief precedes the build. The runner
  lands with D12–D14 frozen in the same PR (definition of done).
- v1 close (README honesty contract, stretch gates) — after M3.

## Frozen decisions

*Pending Kyle. D12–D14 will be recorded here and in `DECISIONS.md` when
picked; relitigating after that is a deviation row, not a conversation.*
