# dim-stage

**Is the global workspace readable in small language models?**

**Measured answer, at the pre-registered bar: no.** Three model scales, six evaluation
distributions, zero passes — the paper's own open question, answered for Qwen2.5
0.5B–3B with a pre-registered null. The three downstream workspace properties were
measured anyway, as descriptive characterization, and the structure inside the null
turned out to be worth the trip: a clean dose–response for injected thoughts at 1.5B,
one CI-clean Jacobian-transport advantage (and two CI-clean reversals), and a
directed-modulation signal ordered exactly as the paper describes at ~1/10 the size.

## What this is

Anthropic's ["Verbalizable Representations Form a Global Workspace in Language
Models"](https://transformer-circuits.pub/2026/workspace/index.html) (2026-07-06) shows
that a **Jacobian lens** — a per-layer linear map that transports any residual-stream
activation (the vector flowing between transformer layers) into the final layer's basis,
where the model's own unembedding matrix decodes it into vocabulary — reads out a sparse,
mid-layer "workspace" that the model verbally reports, computes with, and can be steered
through. The method is demonstrated at 27B and above; below that was unexplored, and the
paper names it an open question.

dim-stage rebuilt the lens independently from the paper's spec, validated it against the
[reference implementation](https://github.com/anthropics/jacobian-lens) (AGREE gate:
bitwise-identical Jacobians at every layer; 3220/3220 readout agreement), and measured
the core three workspace properties — **verbal report, two-hop internal reasoning,
directed modulation** — on **Qwen2.5-0.5B/1.5B/3B-Instruct**, locally, at ~$0/trial
(total outside spend: **$0.83** of rented GPU for the one lens fit the laptop couldn't do).

**The honest framing, always:** this project *reproduced and measured a published
finding — here is the narrow, measured slice.* Nothing here is invented.

## Verdicts (measurement complete 2026-07-16)

Every gate below was frozen as code — wording included — before its first real run.
After M0's triple null, M1–M3 run under a pre-registered re-scope: identical protocols
and numbers, **descriptive framing** (characterization, not reproduction claims).

| Property | Paper anchor (Claude / ≥27B) | dim-stage 0.5B / 1.5B / 3B | Verdict |
|---|---|---|---|
| **M0 Readability** — J-lens pass@10, six eval distributions | readable (the premise) | **0/6, 0/6, 0/6** distributions at Wilson LB ≥ .5 | **NULL ×3** (pre-registered kill-risk fired) |
| **M1 Verbal report** — swapped workspace token reaches top-5 of the spoken report | .88 (N=90) | **.175 / .124 / .105** | far below anchor; report does not follow the swap |
| **M1 Verbal introspection** — steered-in concept gets reported | (no published grid) | α=0→8: **0→0 / 0→30 of 101 / 0→5** | **dose–response at 1.5B only** (.297 [.217, .392]; control exactly 0) |
| **M2 Two-hop swap** — swapped intermediate flips the top-1 answer | .60 (N=90; 54–70% tiers) | **.286 / .073 / .116** | far below anchor; at 3B works **only through the J-transport** |
| **M3 Directed modulation** — instructed concept appears in the workspace | no-instruction baseline ≈ 0 | baseline **reproduces** (0/46 ×3); focus 2→6→9 of 110 | gate: **"does not modulate"** ×3 — but the focus signal is real, ordered as the paper describes, and grows with scale |

Full tables, CIs, and grading conventions per milestone:
[`docs/M0-BRIEF.md`](docs/M0-BRIEF.md) · [`docs/M1-BRIEF.md`](docs/M1-BRIEF.md) ·
[`docs/M2-BRIEF.md`](docs/M2-BRIEF.md) · [`docs/M3-BRIEF.md`](docs/M3-BRIEF.md) —
raw per-trial JSONs in [`results/`](results/).

### The milestone stories, one sentence each

- **M0 — the lens fits, agrees, and does not read.** The independent fitter matches the
  reference bitwise; at the frozen bar (pass@10 Wilson 95% lower bound ≥ .5 on ≥3/6
  distributions) all three scales are NULL, with abstract content (association, poetry)
  a hard zero everywhere and surface-adjacent content partial but sub-bar and
  non-monotone (multihop peaks at 1.5B 54%, regresses to 39% at 3B).
- **M1 — writing the workspace mostly doesn't move the report** (.105–.175 vs .88) —
  *except* that a steered-in thought becomes reportable at 1.5B specifically, rising
  monotonically 0 → 30/101 with steering strength against an α=0 control that is exactly
  0/101 everywhere; the 1.5B–3B gap is CI-clean.
- **M2 — redirecting the unspoken bridge mostly fails to redirect the answer**
  (.073–.286 vs .60), but where it works at all (3B) it works *only* through the
  Jacobian transport: raw unembedding rows flip **0/43**, J − I = **+.116
  [+.011, +.245]** — the project's first CI-clean J-transport advantage — with no
  answer-smuggling signature.
- **M3 — small Qwen models show a genuine, dose-ordered, scale-growing trace of
  top-down workspace control** on concrete category content (focus ≫ control ≈ suppress
  ≈ baseline ≈ 0; CI-clean at 1.5B and 3B; 3B−0.5B growth +.064 [+.004, +.131]) — an
  order of magnitude below Claude, absent entirely for computed math content (the frozen
  both-families gate therefore reads "does not modulate"), and at 3B read *better
  without* the Jacobian transport (J − logit = −.091 [−.181, −.002]).

### Three findings that cut across milestones

1. **The Jacobian transport earns its keep only sometimes.** Every milestone carried a
   standing falsification arm: the identical operation with J = I — the classic **logit
   lens**, reading the residual stream directly with the unembedding, no transport. The
   verdict is content-dependent, not uniform: for *writing*, the transport is the only
   thing that works at 3B in M2 (+.116 CI-clean) and adds nothing measurable in M1; for
   *reading*, it helps on one distribution at 3B (order-ops +16pp), hurts on another
   with increasing depth as scale grows (typo: +43pp → −27pp → −32pp), and at 3B reads
   instructed category content at roughly *half* the rate of the plain logit lens
   (M3's −.091 CI-clean reversal).
2. **1.5B, not 3B, is where the strongest phenomenon lives.** The largest effect in the
   project is M1's injected-thought dose–response at 1.5B (0 → .297 with a clean
   control), with 3B at 5/101 and the gap CI-clean; multihop readability also peaks at
   1.5B. Scale structure below 27B is non-monotone — bigger is not uniformly closer to
   the paper.
3. **Where the paper's anchors apply, everything lands well below them** — roughly an
   order of magnitude on verbal report, and 2–8× on two-hop flips — while the *shape*
   of the phenomena (dose ordering, focus ≫ suppress, baseline ≈ 0) tracks the paper
   where any signal exists at all.

## What was built

- **`fitter.py`** — independent Jacobian-lens fitter from the paper's spec (cotangents
  summed over target positions, averaged over source positions), fp32, deterministic;
  checkpointing; MPS or CUDA.
- **AGREE gate** (`m0_agree_gate.py`) — the independent build vs the reference
  implementation on identical model + prompts: per-layer relative Frobenius distance
  exactly 0 (bitwise), top-1 readout agreement 3220/3220 (Wilson LB .9988). The
  reference package is a pinned dev-dependency used by the cross-check **only** — no
  measurement code imports it.
- **Interventions** (`intervention.py`) — the paper's swap `h ← h + V(σ(c) − c)` and
  steering operators. The reference ships no intervention code, so correctness is gated
  by **pre-committed invariants** (`test_intervention.py`): a rigged analytic subject
  with a hand-computed Jacobian where every post-patch logit is asserted with exact
  equality, coordinate read-back, exact null-ops, and agreement with the literal
  pseudoinverse form — plus the read-back repeated as a runtime self-check on every
  real swap (silent across ~1,000+ applications).
- **Milestone runners** (`m0_readability_gate.py`, `m1_verbal_report.py`,
  `m1_introspection.py`, `m2_two_hop.py`, `m3_directed_modulation.py`) — each loads the
  paper's own shipped stimuli, applies the frozen conventions, and writes a per-trial
  results JSON; each supports `--dry-run` (validate inputs and stop — a wrong-arm input
  exits `VERDICT: INVALID`, code 2) and `--limit` (smoke only).
- **Stats ruler** (`stats.py`) — Wilson 95% CIs on cells, Newcombe 95% CIs on
  differences; ported from the lineage and pytest-covered.

## Methodology (the honesty contract)

Inherited from the project's selection bar; every item held through v1:

- Framing is always *"reproduced and measured a published finding — here is the narrow,
  measured slice."* Never "I invented this."
- **Deterministic oracles only.** Every outcome is a logit ranking read from tensors —
  rank-1 hits over the workspace band, greedy next-token, swap grading = rank of the
  swapped-in candidate. No LLM judges, no text parsing.
- **Every verdict gate pre-committed as code** — wording included — and dry-run against
  wrong-arm input before any real run. A pre-committed null is a reportable headline.
- **Wilson 95% CIs on cells, Newcombe 95% CIs on differences** decide every gate
  (like-for-like: the paper reports Wilson CIs natively). A cell whose CI overlaps its
  neighbor is not a result. N < 20 in a cell is pre-declared UNDERPOWERED.
- **Deviations are owned.** Every departure from the paper is a row in the relevant
  brief's deviations table, never a silent move.

## Owned deviations (headline rows)

The per-milestone tables live in the briefs; the ones that matter most:

- **Model scale:** Qwen2.5 0.5B/1.5B/3B vs the paper's ≥27B — this *is* the experiment.
- **Fit corpus:** WikiText-103, N=100 prompts (deterministic, byte-pinned) vs
  paper-grade ~1,000 generic web text.
- **Single-token stimulus pre-filter** under Qwen's tokenizer (94–100% of stimuli
  survive; every dropped item counted).
- **Hardware/numerics:** fp32 on Apple-silicon MPS; the 3B lens was fit on a rented
  RTX 4090 (CUDA fp32, ~$0.83) after MPS hit a measured ~25× working-set cliff —
  cross-device fp32 noise is orders below any gate's sensitivity.
- **Owned conventions where the release ships none:** eval grading synonym table, the
  M3 prompt frame, the introspection strength grid — each pre-declared in code before
  its first run.

## Reproducing

```bash
git clone https://github.com/ksdisch/dim-stage && cd dim-stage
git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens  # stimuli + eval data (gitignored)

uv run pytest                     # 86 analytic tests — no model downloads, no fitted lenses needed

# Fit a lens (0.5B ≈ 71 min on M-series MPS; 1.5B ≈ 6 h; 3B needs a CUDA GPU — see remote-fit-3b.sh)
uv run fitter.py --model-id Qwen/Qwen2.5-0.5B-Instruct --out lenses/qwen2.5-0.5b-instruct-n100.pt

# Dry-run first (wrong inputs exit INVALID), then measure
uv run m1_verbal_report.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --lens lenses/qwen2.5-0.5b-instruct-n100.pt --dry-run
uv run m1_verbal_report.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --lens lenses/qwen2.5-0.5b-instruct-n100.pt
```

Everything is local: no API keys, no `.env`; models pull from HuggingFace on first use.
Fitted lenses (`lenses/*.pt`) are gitignored — refit to reproduce; the shipped
`results/*.json` are the recorded measurements this README reports.

## Repo map

| Path | Role |
|---|---|
| [`docs/KICKOFF.md`](docs/KICKOFF.md) | The approved brief — scope, anchors, risks, decisions on record. **Source of truth.** |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestone status vs the plan, updated at each stage close |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Append-only log of every frozen decision (D1–D14) with rationale and outcome |
| [`docs/LEARNING.md`](docs/LEARNING.md) | Plain-English teaching notes + lessons, milestone by milestone |
| `docs/M0…M3-BRIEF.md` | Per-milestone design extraction, frozen conventions, deviations table, full results |
| `results/*.json` | Per-trial deterministic outcomes for every run this README cites |

## Lineage

Repro #5: forge-gap → decay-pin → lossy-wall → ghost-patch → **dim-stage**.
First internals/interpretability project in the line — activations and gradients instead
of API text.
