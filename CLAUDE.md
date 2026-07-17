# CLAUDE.md — dim-stage

Project conventions and guardrails for working in this repo. Read this first each session.

## What this is

Reproduce and measure, at hobby scale, whether the **global workspace** from Anthropic's
workspace/J-lens paper (transformer-circuits.pub/2026/workspace — **no arXiv ID; cite the
URL**) is readable in small local Qwen models (Qwen2.5 0.5B + 1.5B, escalated to 3B when
both went null — KICKOFF decision 2's trigger). Independently build the
Jacobian lens, cross-check it against the reference `anthropics/jacobian-lens`
implementation (AGREE gate), then measure the core-three properties: **verbal report,
two-hop swap, directed modulation**.

**Source of truth: `docs/KICKOFF.md`** — the approved kickoff brief (scope, phased plan,
anchors, risks, decisions on record). Scope decisions there are settled; don't relitigate
them. Headlines: **two subjects, both measured fully** (3B only if both null on
readability); **generalization + selectivity are OUT of v1** (generalization later
shipped as post-v1 stretch S2); the paper's LLM-judge and
Claude-only sections are **never** in scope; the reference jlens is a **cross-check
oracle only — pinned as a dev-dependency, never imported by harness code** (lossy-wall D1
pattern).

The honest framing, always: *reproduced and measured a published finding — here is the
narrow, measured slice.* Never "I invented this."

## Where we are

**v1 is CLOSED (2026-07-16) — M0–M3 plus two stretch stages (S1, S2) all complete.**
Full status lives in `docs/ROADMAP.md`; per-stage detail in the `docs/*-BRIEF.md` files,
`docs/DECISIONS.md` (D1–D22 frozen), and `docs/LEARNING.md`. The two pre-declared risks
both resolved: the readability kill-risk **fired** (see M0 below), and MPS handled the
backward pass at 0.5B/1.5B but not 3B (rented RTX 4090 fallback — owned deviation).

- **M0 — readability: triple NULL.** 0/6 distributions at Wilson LB ≥ 0.5 on 0.5B,
  1.5B, *and* the 3B escalation. The kill-risk fired and held; everything downstream is
  **descriptive characterization, never a reproduction claim** (the standing re-scope).
- **M1 verbal report** — swaps don't move the report; but a *steered-in* thought becomes
  reportable at 1.5B specifically (dose–response, exact-zero α=0 control).
- **M2 two-hop swap** — mostly fails; where it works at all (3B) it works *only* through
  the Jacobian transport (first CI-clean J-advantage for writing).
- **M3 directed modulation** — "does not modulate" at the frozen gate, but a real,
  dose-ordered, scale-growing focus signal on concrete category content.
- **S1 (stretch)** — the 1.5B dose–response hardened: CI-cleanly a J-transport effect,
  saturates like the paper's Figure 7, localizes to the mid-band (L16–20).
- **S2 (stretch)** — generalization: "does not route" at anchor level, but a CI-clean
  J-specific routing signal at α=1 on all three subjects, with the paper's category
  ordering; the paper's α=2 rescue *inverts* here (overdose extinguishes routing).
- **S3 (stretch)** — selectivity: **the only would-gate to HOLD on all three
  subjects** — J-space ablation kills two-hop chains while random damage and
  ordinary WikiText prediction survive CI-cleanly above them (relative selectivity,
  owned as such); presence-on-demand texture in both targeted arms.

**Open decision (Kyle's, not a default):** wrap the project (/seed-hunt) or a
further stretch (e.g. the paper's Figure-69 avoidance experiment, now cheap since
the ablation operator exists). Nothing is scheduled until he picks.

## How to run

- Anything: `uv run <script>` — `uv` (Python 3.12+) manages the venv. Application, not a
  package (`package = false`).
- `uv run pytest` greens the full suite (stats ruler + per-stage invariant/gate tests).
  Runners live at the repo root (`m0_*.py` … `s2_generalization.py`); fitted lenses in
  `lenses/`, per-run JSONs in `results/`, run logs as `*.log` at root.
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
