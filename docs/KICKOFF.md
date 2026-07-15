# Kickoff Brief — dim-stage
*Created 2026-07-15 · status: scoped*

## One-liner
Independently build the Jacobian lens from Anthropic's workspace paper, validate it against the reference implementation, and measure whether a readable global workspace exists in small local Qwen models (0.5B + 1.5B) — the paper's own open question — at ~$0/trial under pre-committed statistical gates.

## Seed
"Verbalizable Representations Form a Global Workspace in Language Models" (Anthropic, 2026-07-06).
Canonical URL: https://transformer-circuits.pub/2026/workspace/index.html — **no arXiv ID exists; cite the transformer-circuits URL.**
Reference code: github.com/anthropics/jacobian-lens (Apache-2.0, "not maintained"). Refetch the paper with `defuddle parse <url> --md -o <file>` (~325KB, ~2233 lines).

## Why now / the problem
Repro #5 in the lineage (forge-gap → decay-pin → lossy-wall → ghost-patch). The paper is 9 days old and the ecosystem is replicating fast — timeliness was a deciding factor at the hunt. The skill axis is the point: this is the first internals/interpretability project in the lineage (activations, gradients, logit readouts vs. API text), and the method is proven only at 27B (Qwen3.6-27B on Neuronpedia); below that is unexplored, which **is** the thesis. Honest framing, always: *"reproduced and measured a published finding — here is the narrow, measured slice."*

## Who it's for
Kyle — portfolio + internals skill-building. Today's alternative: the paper's claims exist only at ≥27B / Claude scale; nobody has published whether the workspace is readable on laptop-sized models.

## What success looks like
- **v1 done means:** M0 cross-check gate AGREE (independent build vs reference jlens, same model + prompts); a readability verdict on **both** subjects (either direction — a null is a headline); the core-three properties (verbal report, two-hop swap, directed modulation) measured on both subjects with Wilson/Newcombe CIs against the paper's anchors, every gate pre-committed as code and dry-run before real runs; README carries the honesty contract.
- **Would be amazing:** the scale transition bracketed (e.g. readable at 1.5B but not 0.5B — an emergence point below 27B); the paper's own predictor ("workspace loading cosine-sim predicts swap success") replicates; 3B escalation confirms the trend.
- **Explicitly NOT trying to:** invent anything; touch the LLM-judge-graded sections (experiential-language, ablation-flattening — bar-banned) or the Claude-only auditing/reflection-training sections; run any training.

## Scope
**In (v1):**
- Independent lens-fitting implementation from the paper + `jlens.fitting` docstring spec (cotangents summed over target positions, averaged over source positions)
- Oracle cross-check vs reference jlens — the lossy-wall AGREE pattern, $0 here
- Logit-lens (J=I) baseline as the standing falsification arm at early/mid layers
- Two subjects: Qwen 0.5B + 1.5B, both measured fully
- Core three properties, using the paper's own shipped stimuli: `verbal-report.json` (+ `verbal-introspection.json`), `probe-swap.json` (90 two-hop prompts), `directed-modulation.json`
- Lens-quality evals (`lens-eval-*`, six distributions, pass@k) powering both the AGREE gate and the readability gate

**Out / deferred / never:**
- Generalization + selectivity properties (stretch milestones only if v1 lands early)
- 3B escalation unless BOTH subjects null on readability
- Judge-graded and Claude-only paper sections (never)
- OpenRouter / tool loops / text parsing — outcomes are logit rankings read from tensors (never)

## Shape
Local Python repo, milestone scripts + pre-committed gate code, lineage-style (`m0.py`…`m3.py` verdict pattern). No server, no UI.

## Inputs & data
The paper's actual stimuli ship in `anthropics/jacobian-lens` under `data/experiments/` and `data/evaluations/`, each with a README specifying the mechanical grading convention (rank-1 hits over the workspace band, greedy next-token, swap grading = rank of swapped-in candidate). Fit corpus: generic web text, ~100 prompts usable (1000 = paper grade) — corpus choice is an owned deviation row. Mechanical single-token pre-filter of stimuli under Qwen's tokenizer (free).

## Integrations & dependencies
HuggingFace `transformers` + PyTorch on MPS; reference `jlens` (Apache-2.0) pinned as a dev-dependency **for the cross-check only**; `stats.py` Wilson/Newcombe ported from lossy-wall. No accounts, no API keys.

## Constraints
Apple-silicon laptop; $0/trial — N bound by wall-clock, not dollars; no training runs or rented-GPU dependence (a ~$1 GPU fallback would be an owned deviation row per the amended bar entry 2, never a silent move).

## Riskiest assumptions & unknowns
1. **A readable workspace exists at small scale** — the thesis and the kill-risk in one. *Cheap test:* M0 readability gate — lens-eval pass@k / rank-1 hits vs the logit-lens baseline on a ~100-prompt fit. Pre-declared: a null is a headline (answers the paper's open question) but decapitates M1–M3, which re-scope to descriptive — the ghost-patch awareness-leg pattern.
2. **The backward pass runs on MPS** — lens fitting is dominated by it. *Cheap test:* M0 hour-one gate on Qwen-0.5B; measure wall-clock/prompt → measured-rate feasibility for the full fit.
3. **Rate compression at small scale** — the paper's mid-range anchors (54–70% flips) may compress toward floor/ceiling on tiny models; powerable only because trials are free.
4. **The estimator has subtleties** — independent build could silently diverge. *Mitigation:* the AGREE gate is the whole point; its metric gets frozen in the M0 brief before any comparison runs.

## Quantitative anchors (from the paper, for milestone gates)
- Two-hop intermediate swap → top-1 answer flip: Haiku 54% / Sonnet 70% / Opus 70% (N=50)
- Component contrast: J-space 59–61% vs non-J-space 5–28% (→6% clamped)
- Pure lens vectors: 88% top-5 (N=90; paper uses Wilson 95% CIs natively)
- Directed modulation no-instruction baseline ≈ 0
- Generalization (stretch): 76/192 → 101/192 at α=2
- "Workspace loading (cosine sim) predicts swap success" is the paper's own testable predictor
- J-space: ≤10% of activation variance; mid-layers only; flips to next-token "motor" content in the last few layers

## Open questions
- Workspace-band selection procedure at small scale (paper defines a mid-layer band — the extraction rider in M0 pins the exact procedure)
- AGREE-gate metric definition (readout overlap vs pass@k-within-CI) — frozen in M0's brief pre-run
- Which web-text corpus for fitting (repo docs suggest user-supplied)

## Phased plan
### Milestone 0 — Fit pilot: does the lens fit, agree, and read?
- Hour-one gate: MPS backward pass on Qwen-0.5B, wall-clock measured
- Design-extraction rider: verbatim extraction of band-selection + fitting procedure from paper/repo (free, pre-committed)
- Independent build → reference cross-check → **AGREE gate**
- **Readability gate** on both subjects: J-lens vs logit-lens baseline on the six lens-eval distributions
- Single-token stimulus pre-filter
### Milestone 1 — Verbal report
- `verbal-report.json` swap protocol + `verbal-introspection.json` steering; anchors: 88% top-5 (N=90), rank-1 grading
### Milestone 2 — Internal reasoning (two-hop swap)
- `probe-swap.json`, 90 prompts; anchors: 54–70% top-1 flip (N=50); J-space vs non-J-space component contrast (59–61% vs 5–28%); workspace-loading predictor as free correlational arm
### Milestone 3 — Directed modulation
- `directed-modulation.json`; anchor: no-instruction baseline ≈0
### Stretch (explicitly gated behind v1 close)
- Generalization (needs N≈200/arm — free but slow), selectivity, 3B escalation

## Honesty contract (inherited from the selection bar)
- Framing is always "reproduced and measured a published finding — here is the narrow, measured slice." Never "I invented this."
- Every verdict gate pre-committed as code and dry-run against wrong-arm input before real runs; a pre-committed null is a reportable headline.
- Deterministic oracles only — outcomes are logit rankings read from tensors; no LLM judges, no text parsing.
- N≥20/arm or the verdict is pre-declared UNDERPOWERED.
- Deviations from the paper (model scale, fit-corpus size, single-token filter, MPS vs CUDA) are owned rows in a deviations table, never silent moves.

## Decisions on record (kickoff interview, 2026-07-15)
1. Independent lens build + reference cross-check (over using the lib directly or hybrid)
2. Two subjects measured fully: Qwen 0.5B + 1.5B; escalate to 3B only if BOTH null on readability
3. Core-three property scope: verbal report, two-hop swap, directed modulation; generalization + selectivity deferred
4. Slug `dim-stage`, GitHub public
5. Reuse from lineage = stats.py Wilson/Newcombe + honesty contract ONLY (no OpenRouter, no tool-loop, no text parsing)

## Tech stack
Python 3.12 + uv · PyTorch (MPS) + transformers · reference jlens as pinned dev-dep (cross-check only) · lossy-wall `stats.py` port · pytest for gate dry-runs. *Rationale: the only stack that gives local gradient access on the hardware; everything else is lineage-proven.*
