# CLAUDE.md — dim-stage

Project conventions and guardrails for working in this repo. Read this first each session.

## What this is

Reproduce and measure, at hobby scale, whether the **global workspace** from Anthropic's
workspace/J-lens paper (transformer-circuits.pub/2026/workspace — **no arXiv ID; cite the
URL**) is readable in small local Qwen models (0.5B + 1.5B). Independently build the
Jacobian lens, cross-check it against the reference `anthropics/jacobian-lens`
implementation (AGREE gate), then measure the core-three properties: **verbal report,
two-hop swap, directed modulation**.

**Source of truth: `docs/KICKOFF.md`** — the approved kickoff brief (scope, phased plan,
anchors, risks, decisions on record). Scope decisions there are settled; don't relitigate
them. Headlines: **two subjects, both measured fully** (3B only if both null on
readability); **generalization + selectivity are OUT of v1**; the paper's LLM-judge and
Claude-only sections are **never** in scope; the reference jlens is a **cross-check
oracle only — pinned as a dev-dependency, never imported by harness code** (lossy-wall D1
pattern).

The honest framing, always: *reproduced and measured a published finding — here is the
narrow, measured slice.* Never "I invented this."

## Where we are

**Current milestone: M0 — the fit pilot** (not started; fresh scaffold, 2026-07-15).
Hour-one gate: MPS backward pass on Qwen-0.5B, wall-clock measured. Then the
design-extraction rider (band-selection + fitting procedure, verbatim from paper/repo),
the independent build → cross-check AGREE gate, the readability gate on both subjects
(J-lens vs logit-lens `J=I` baseline on the six `lens-eval-*` distributions), and the
single-token stimulus pre-filter.

**Riskiest assumptions — keep them front-of-mind:**
1. **A readable workspace exists at small scale** — the thesis and the kill-risk in one.
   A null is a headline (it answers the paper's stated open question) but decapitates
   M1–M3, which re-scope to descriptive. Pre-declared, like ghost-patch's awareness leg.
2. **The backward pass runs on MPS** — lens fitting is dominated by it. If it doesn't,
   a ~$1 rented-GPU fallback is an *owned deviation row*, never a silent move (amended
   bar entry 2).

## How to run

- Anything: `uv run <script>` — `uv` (Python 3.12+) manages the venv. Application, not a
  package (`package = false`).
- `uv run pytest` greens the stats ruler (`test_stats.py`). No harness code exists yet;
  building starts at M0.
- No API keys, no `.env` — everything is local. Models pull from HuggingFace on first use.

## Methodology guardrails (load-bearing — do not drift)

- **Deterministic oracles only.** Outcomes are **logit rankings read from tensors**
  (rank-1 hits over the workspace band, greedy next-token, swap grading = rank of the
  swapped-in candidate — the conventions in the reference repo's data READMEs). Never an
  LLM judge, never text parsing.
- **Wilson CIs on cells + Newcombe CIs on differences decide every gate.** The paper
  reports Wilson 95% CIs natively, so anchor comparisons are like-for-like.
- **N ≥ 20 per cell or the verdict is pre-declared UNDERPOWERED.** Trials are free —
  N is bound by wall-clock, not dollars — so prefer N=50–100 where the clock allows.
- **A cell whose CI overlaps its neighbor is not a result.** Every gate has an explicit
  CI condition; a claim that doesn't clear its gate doesn't get made.
- **Pre-commit gates as code and dry-run them** (wrong-arm input exits INVALID) before
  any real run. A pre-committed null is a reportable result.
- **Design-extraction before design-signing** (bar entry 9): extract the paper's/repo's
  actual procedure verbatim as a free pre-commit step in every milestone brief.
- **Deviations are owned.** Model scale, fit-corpus size (~100 vs paper-grade 1000),
  single-token tokenizer filter, MPS vs CUDA — each is a row in a deviations table.

## Working with Kyle — teaching standard + per-stage rhythm (load-bearing)

Kyle is driving this project to learn interpretability internals deeply and is sharp but
**new to coding jargon** — no CS degree. The job isn't just to ship code; it's to leave
him able to *defend every decision*. These rules bind every session and tab.

- **Explain-clearly standard.** Plain English first; define **every** jargon term the
  first time it appears, inline; clearer, not longer. This project is jargon-dense
  (Jacobian, residual stream, unembedding, VJP, logit lens) — the standard matters more
  here than in any prior repro.
- **Decision-brief format.** For any real choice, lay out 2–3 options in plain terms,
  each with its trade-off, plus a recommendation *and the reason*. Kyle decides.
- **Per-stage rhythm (the docs spine).** *Start of a stage:* plain-terms brief + options
  into `docs/` before coding. *End of a stage:* update `ROADMAP.md`, append to
  `DECISIONS.md`, add teaching notes to `LEARNING.md`, ask 3 recall questions. The spine
  beyond `KICKOFF.md` starts with M0's start-of-stage brief.
- **Definition of done:** once the spine exists, a stage isn't finished until its spine
  updates are committed in the same PR as the code.

## Working conventions

- **Keep it lean.** No premature abstractions; one legible deliverable per milestone.
- **Refetch the paper** when needed with `defuddle parse <url> --md -o <file>`
  (~325KB — never commit it; it's Anthropic's content).
- Conventions beyond the lineage standard: TBD as the harness takes shape.
