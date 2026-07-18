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
- **S4 (stretch)** — naming vs avoiding (Fig 69): **NOT shown** — the late-band
  concept direction is a hard output off-switch at every scale, but the paper's
  early-band suppression copy is absent; only 1.5B can even do the exclusion task
  (gate 22/60 vs 5 and 8). First constructed item set, frozen pre-run.
- **S4b (follow-up)** — the off-switch is **concept-specific on 1.5B, CI-clean**
  (primed 0/22 vs control 16/22, +.727) with specificity emerging by scale
  (0.5B: any-direction damage; 3B: perfectly clean control). Shared cells
  reproduced bit-for-bit on the re-run.

**Open decision (Kyle's, not a default):** wrap the project (/seed-hunt) or
further work he names. Nothing is scheduled until he picks.

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

## Claude tooling for this repo

Global commands (`.claude/commands/`) and skills (`.claude/skills/`) vendored from `ksdisch/claude-config` via `/claudify-repo`, so they work in cloud/web sessions and for collaborators. ✅ = cloud-safe (pure reasoning + repo edits). 💻 = **local-only** — needs local tools (browser MCP, Chrome, local TTS/voice, or the local `nlm` CLI / NotebookLM MCP) and will NOT work in a cloud/web session.

### Commands

- ✅ `/autonomous-milestone` — plan/build/test/verify a target end-to-end, or triage the backlog into ranked candidates; ultracode multi-agent orchestration.
- ✅ `/begin` — open a session: orient on branch/commits/open PRs, recap the last `/wrap` log, route into the session-start spec. (Optional audio recap is local-only.)
- 💻 `/boot_server` — detect how the project is served, start the dev server in the background, open it in Chrome.
- ✅ `/brainstorm` — multi-mode structured brainstorm (Moonshot default; QuickWin, Subtract, Harden, Premortem, Friction, Delight, Positioning, Reach); blind agent teams + critic gate → `docs/ideas/` vision docs + backlog stubs.
- 💻 `/catchup` — mid-session audio catch-up as an MP3 (local TTS); keeps working after.
- ✅ `/claudify-repo` — vendor global commands/skills into this repo and/or brainstorm repo-specific automations.
- 💻 `/envsetup` — open `.env` in the editor + the credential's generation page in Chrome, with a key stub pre-added.
- ✅ `/explore-plan` — explore → plan → confirm before any code; proposes 2–3 ranked approaches and waits for a pick.
- ✅ `/handoff` — generate a paste-ready handoff prompt for a fresh session; captures lessons + plan state. (Optional audio is local-only.)
- 💻 `/mock-sql-audio` — full simulated SQL mock interview as an MP3 (local two-voice TTS).
- ✅ `/mock-sql-demo` — text self-play mock SQL interview (interviewer + ideal candidate), then a debrief.
- 💻 `/mock-sql-interview` — live voice mock SQL interview (local voice mode).
- ✅ `/prompt-optimize` — one-shot prompt rewrite: diagnose, pick a workflow archetype + model + effort, return a ready-to-paste prompt. Advisory only.
- ✅ `/reframe-orchestrator` — reframe `.claude/orchestrator.md` into a mode-independent invariants & gates doc; docs-only.
- 💻 `/screenshot-iterate` — visual loop: implement against a mock, screenshot the running app, compare, iterate.
- 💻 `/smoke-test` — set up a manual smoke test: opens the needed pages in Chrome (auto-boots the dev server) and hands over a do-this-see-that checklist saved under `docs/smoke/`.
- ✅ `/tdd` — test-first loop: write failing tests, confirm they fail for the right reason, commit, then code until green without touching the tests.
- ✅ `/trim-context` — find and fix Claude Code token bloat (oversized CLAUDE.md, bloated memory, `.claude/` cruft); auto-applies fixes.
- ✅ `/wrap` — end-of-session recap: the why, vocabulary, active-recall quiz, next moves; saves a dated file. (Optional audio is local-only.)

### Skills (auto-trigger by description, or invoke by name)

- ✅ `artifacts-audit` — audit which engineering artifacts the repo should have; writes `docs/artifacts-plan.md`. Plans only.
- ✅ `artifacts-generate` — generate artifacts from `docs/artifacts-plan.md` (one-at-a-time or batch). Companion to `artifacts-audit`.
- 💻 `audio-series` — episodic NotebookLM audio series for an existing notebook (needs `nlm`/NotebookLM MCP).
- ✅ `bug-hunt` — proactive bug hunt: fan out finder agents, adversarially verify findings, ranked triage list; optional hand-off to a fix flow.
- 💻 `interview-prep` — init/maintain a NotebookLM interview-prep notebook from the local job-search dossier (needs `nlm`/NotebookLM MCP).
- ✅ `kickoff` — deep one-question-at-a-time discovery interview → approved kickoff brief + phased plan → scaffold the project + GitHub repo.
- 💻 `match-the-mock` — implement a UI against a mock and iterate via browser screenshots until it matches.
- ✅ `mini` — kick off a new mini project under `~/Projects/mini/` (short interview + scaffold).
- 💻 `narrate` — turn a short brief into a single-voice MP3 narration (local Kokoro TTS).
- 💻 `nlm-skill` — expert guide for the NotebookLM CLI (`nlm`) and MCP server.
- 💻 `notebook-assist` — refine artifacts / brainstorm / manage sources for an existing NotebookLM notebook.
- 💻 `notebook-init` — initialize a new NotebookLM notebook end-to-end.
- 💻 `notebook-merge` — merge 2+ overlapping NotebookLM notebooks into one unified notebook.
- ✅ `project-guide` — comprehensive point-in-time guide to the project (purpose, architecture, history, interview lens); saves a dated file. (Optional audio is local-only.)
- ✅ `research-paper` — end-of-project research paper + presenter pack from a completed repo's recorded results; opens a PR for review, never merges.
- ✅ `seed-hunt` — end-of-project seed hunt: verify closure, harvest lessons into the selection bar, sweep arXiv, decision brief. (Optional audio is local-only.)
- ✅ `ship-and-route` — land outstanding git work behind a review gate, walk the findings, route the next move with a starter prompt.
- 💻 `video-series` — episodic NotebookLM video series for an existing notebook (needs `nlm`/NotebookLM MCP).

To vendor more global tooling or brainstorm repo-specific automations, run `/claudify-repo`.

## Operating Constraints

@.claude/operating-constraints.md
