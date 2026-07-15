# dim-stage

**Is the global workspace readable in small language models?**

Anthropic's ["Verbalizable Representations Form a Global Workspace in Language
Models"](https://transformer-circuits.pub/2026/workspace/index.html) (2026-07-06) shows
that a **Jacobian lens** — a linear map that transports any residual-stream activation
into the final-layer basis and decodes it with the model's own unembedding — reads out a
sparse, mid-layer "workspace" that the model verbally reports, computes with, and can be
steered through. The method is demonstrated at 27B and above; **below that is unexplored,
and the paper names it an open question.**

dim-stage independently rebuilds the lens from the paper's spec, validates it against the
[reference implementation](https://github.com/anthropics/jacobian-lens) (AGREE gate), and
measures three workspace properties — **verbal report, two-hop internal reasoning,
directed modulation** — on Qwen **0.5B and 1.5B**, locally, at ~$0/trial.

## The honest framing

This project **reproduces and measures a published finding — here is the narrow, measured
slice.** Nothing here is invented. Deviations from the paper (model scale, fit-corpus
size, single-token stimulus filter, MPS vs CUDA) are owned rows in a deviations table,
never silent moves.

## What v1-done means

- Independent lens build **cross-checked against the reference jlens** on the same model
  and prompts — the AGREE gate.
- A **readability verdict on both subjects** (J-lens vs a logit-lens `J=I` baseline on
  the paper's six lens-eval distributions) — either direction is a headline; a null
  answers the paper's open question.
- The three core properties measured on both subjects against the paper's quantitative
  anchors, with **Wilson/Newcombe 95% CIs** deciding every gate, all gates pre-committed
  as code and dry-run before real runs.
- Outcomes are **logit rankings read from tensors** — no LLM judges, no text parsing.

## Status

**Milestone 0 — fit pilot** (not started; fresh scaffold, 2026-07-15).
Hour-one gate: does the backward pass run on Apple-silicon MPS for Qwen-0.5B?

## Lineage

Repro #5: forge-gap → decay-pin → lossy-wall → ghost-patch → **dim-stage**.
First internals/interpretability project in the line — activations and gradients instead
of API text.

Source of truth: [`docs/KICKOFF.md`](docs/KICKOFF.md).
