# Sources

| Source | Location | Type | Authoritative for |
|--------|----------|------|-------------------|
| Anthropic workspace/J-lens paper | https://transformer-circuits.pub/2026/workspace/index.html (no arXiv ID — cite the URL; local mirror `refs/workspace-paper.md`, gitignored) | paper | The finding being reproduced: method spec, anchors (.88 report, .60 two-hop), stimuli conventions |
| Reference implementation | https://github.com/anthropics/jacobian-lens (cloned at `refs/jacobian-lens/`, gitignored; pinned dev-dependency) | code + data | AGREE-gate cross-check oracle, shipped stimuli/eval data, grading conventions in its data READMEs — never imported by measurement code |
| Kickoff brief | `docs/KICKOFF.md` | brief | Scope, phased plan, anchors, risks, decisions on record — **project source of truth**; scope there is settled |
| Decision log | `docs/DECISIONS.md` | log (append-only, D1–D31) | Every frozen decision with rationale and outcome — the wiki's decisions ledger; do not duplicate at root |
| Roadmap | `docs/ROADMAP.md` | status doc | Milestone status vs plan, updated at each stage close |
| Milestone/stage briefs | `docs/M0…M3-BRIEF.md`, `docs/S1…S4-BRIEF.md` | briefs | Per-stage design extraction, frozen conventions, deviations tables, full result tables + CIs |
| Recorded measurements | `results/*.json` (+ root `*.log` run logs) | raw exports | Per-trial deterministic outcomes behind every number the README and paper cite |
| Paper set | `docs/paper/` (paper, presenter pack, figures + scripts, ELI5 rewrite) | write-up | The official narrative of the recorded results (derived — defers to `results/` and the briefs) |
| Teaching notes | `docs/LEARNING.md` | notes | Plain-English explanations + lessons, milestone by milestone |
