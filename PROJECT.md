# PROJECT.md

## Purpose
Reproduce and measure, at hobby scale, whether the global workspace from Anthropic's workspace/J-lens paper (transformer-circuits.pub/2026/workspace) is readable in small local Qwen2.5 models (0.5B/1.5B/3B) — an independent Jacobian-lens build, cross-checked against the reference implementation, then measured on the core-three workspace properties. (Fact — `docs/KICKOFF.md`, `README.md`.)

## Scope
**In (v1, all complete):** independent J-lens fitter + AGREE gate vs `anthropics/jacobian-lens`; M0 readability gate; M1 verbal report + introspection; M2 two-hop swap; M3 directed modulation; stretch stages S1 (dose–response localization), S2 (generalization), S3 (selectivity), S4/S4b (naming vs avoiding + specificity control). Post-v1: research paper + presenter pack + ELI5 rewrite in `docs/paper/`.

**Out / never:** the paper's LLM-judge and Claude-only sections; importing the reference implementation into any measurement code (cross-check oracle only, pinned dev-dependency). Scope decisions in `docs/KICKOFF.md` are settled — don't relitigate. (Fact — `CLAUDE.md`, `docs/KICKOFF.md`.)

## Current status
**Complete (v1 CLOSED 2026-07-16; paper deliverables landed 2026-07-19).** Headline: a pre-registered triple NULL on readability (0/6 distributions at Wilson LB ≥ .5 on all three scales) — the paper's open question answered "no" for Qwen2.5 0.5B–3B — with descriptive structure inside the null: a 1.5B dose–response for injected thoughts, one CI-clean J-transport advantage (M2 at 3B) plus two CI-clean reversals, a dose-ordered scale-growing M3 focus signal, and a concept-specific late-band output off-switch (S4b). Full status: `docs/ROADMAP.md`. (Fact — `docs/ROADMAP.md`, `README.md`.)

## Next actions
1. Kyle's wrap decision (open, his — not a default): close the project via `/seed-hunt`, or name a targeted follow-up. Nothing is scheduled until he picks. (Fact — `CLAUDE.md` "Open decision", `docs/ROADMAP.md` "Remaining stretch — none scheduled".)

## Boundaries
- Local-first: no API keys, no `.env`; models pull from HuggingFace on first use; `uv run` for everything.
- Hardware: fp32 on Apple-silicon MPS; 3B lens fit required a rented RTX 4090 (~$0.83 total outside spend) after a measured MPS working-set cliff — an owned deviation.
- Methodology guardrails (load-bearing, see `CLAUDE.md`): deterministic oracles only (logit rankings, never LLM judges or text parsing); Wilson/Newcombe 95% CIs decide every gate; gates pre-committed as code and dry-run before real runs; every deviation owned in a briefs deviations table.
- Fitted lenses (`lenses/*.pt`) and `refs/` (reference repo clone + paper mirror) are gitignored; `results/*.json` are the recorded measurements of record.
