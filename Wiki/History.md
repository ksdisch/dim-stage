# History — dim-stage

> How this project got here: a chronological narrative of eras and milestones,
> reconstructed from merged PRs, git history, wrap logs, and ADRs.
> PR numbers, merge dates, tags, and SHAs are **Fact** by construction; rationale
> lines carry explicit labels (**Fact** when quoted from a PR body/ADR, **Inference**
> when reconstructed). Decisions are anchored by ID to the project's decision
> ledger — never restated here. **Append-only:** new milestones are added at the
> bottom (above the Mining coverage footer); existing entries are never rewritten.

## Origin — 2026-07

Fifth project in the reproduce-and-measure lineage (forge-gap → decay-pin →
lossy-wall → ghost-patch): an independent build of the Jacobian lens from
Anthropic's workspace paper (transformer-circuits.pub/2026/workspace, no arXiv
ID), cross-checked against the reference `anthropics/jacobian-lens`, measuring
whether a readable global workspace exists in small Qwen models — the paper's
own open question below 27B. First commit `c918968` (2026-07-15, "kickoff:
dim-stage — scaffold from brief"); kickoff brief at `docs/KICKOFF.md`.

## Era: M0 — fit pilot and the triple NULL (2026-07-15 – 2026-07-16)

The stage that decided everything downstream: the independent fitter proved
bitwise-faithful to the reference, and the pre-registered readability gate
returned NULL at all three scales, re-scoping M1–M3 to descriptive.

### M0 start-of-stage brief + first decision freeze — 2026-07-15
- **Landed:** design extraction from paper + reference repo, hour-one gate spec, deviations table (PR #1)
- **Why:** conventions and gates frozen before any code or result [Fact — PR #1 body] — see D1–D4 in `docs/DECISIONS.md`

### Hour-one gate PASS — 2026-07-15
- **Landed:** MPS backward-pass wall-clock gate; extrapolated 7.55 h ≤ 12 h, no rented-GPU fallback needed (PR #2)
- **Why:** kill the project in hour one if local fits were infeasible [Inference — from gate spec in PR #1/#2 bodies]
- **Tradeoff:** `dim_batch` frozen at 8 — raising it is math-neutral but ~25× slower on MPS [Fact — PR #2 body]

### Independent fitter + AGREE gate — 2026-07-15
- **Landed:** torch-only `fitter.py` (zero `jlens` imports), analytic test oracles, D1 gate verdict AGREE — bitwise-identical J at all 23 layers, readout 3220/3220 (PR #3)
- **Why:** lossy-wall oracle pattern — independent build validated against the reference before any measurement — see D1 in `docs/DECISIONS.md`

### Readability verdict NULL/NULL + M0 close — 2026-07-15
- **Landed:** `readability.py` + two-arm gate, official results for 0.5B and 1.5B — NULL on both (0/6 distributions), plus the closing spine (ROADMAP, DECISIONS, LEARNING) (PR #4)
- **Why:** pre-registered verdict; "a null is a headline" per the kickoff success criteria [Fact — PR #4 body, `docs/KICKOFF.md`] — see D4 in `docs/DECISIONS.md`

### 3B escalation → triple NULL — 2026-07-16
- **Landed:** pre-registered 3B escalation (PR #5), `--prompts-file` fix for the HF streaming wedge (PR #7), rented-GPU rescue package after the MPS 24 GB cliff (PR #8), and the verdict: 3B NULL too — 0/6 at all three scales (PR #9)
- **Why:** the double null met the kickoff's pre-registered escalation trigger; Kyle escalated [Fact — PR #9 body] — see the "3B escalation" and "3B rescue" sections in `docs/DECISIONS.md`
- **Tradeoff:** 3B fit moved to a rented RTX 4090 (~57 min, ~$0.83) with byte-identical procedure; gate stayed local on MPS [Fact — PR #9 body]

## Era: v1 core three — descriptive measurement (2026-07-15 – 2026-07-16)

With readability NULL ×3, the three core workspace properties were measured
descriptively on all three subjects, each stage in the same rhythm: brief →
freeze → runner + pre-committed gates → three subjects → spine, mostly same-day.

### M1 — verbal report + introspection — 2026-07-16
- **Landed:** M1 brief (PR #6), intervention operators + D6 invariant gate (PR #10), verbal-report runner + results (PR #11), introspection runner + M1 close (PR #12)
- **Result:** report does not follow the swap at any scale; a steered-in thought becomes reportable at 1.5B specifically (0 → 30/101, control exactly 0) — the project's first dose–response curve [Fact — PR #12 body]
- **Why:** the reference ships no intervention code, so the AGREE gate was replaced by an analytic exact-equality oracle — see D5–D8 in `docs/DECISIONS.md`

### M2 — two-hop bridge swap — 2026-07-16
- **Landed:** brief (PR #13), runner + results + stage close (PR #14)
- **Result:** the chain mostly doesn't follow the bridge swap, but at 3B it works only through the Jacobian transport — first CI-clean J-advantage for writing [Fact — PR #14 body] — see D9–D11 in `docs/DECISIONS.md`

### M3 — directed modulation — 2026-07-16
- **Landed:** brief (PR #15), runner + results + stage close (PR #16); v1 core-three measurement complete
- **Result:** frozen would-gate reads "does not modulate" ×3, but the focus signal is paper-ordered and grows CI-cleanly with scale; at 3B the plain logit lens beats the J-lens on instructed content [Fact — PR #16 body] — see D12–D14 in `docs/DECISIONS.md`

### v1 close — README honesty contract — 2026-07-16
- **Landed:** outcome-first README rewrite (verdicts table, honesty contract, reproduce path); ROADMAP flipped to v1 CLOSED (PR #17)
- **Why:** the README is the repo's public face, so it merged on Kyle's explicit sign-off rather than autonomously [Fact — PR #17 body]

## Era: Stretch season — S1–S4b (2026-07-16 – 2026-07-17)

Five follow-up stages hardening and extending the v1 findings, one to two
stages per day, each with its own frozen decision bundle.

### S1 — introspection dose–response localized — 2026-07-17
- **Landed:** brief (PR #18), runner + three-subject results + close (PR #19)
- **Result:** the 1.5B dose–response is a genuine J-transport effect (CI-clean falsification arm), saturates from α=8, and localizes to the middle third L16–20 [Fact — PR #19 body] — see D15–D18 in `docs/DECISIONS.md`

### S2 — flexible generalization — 2026-07-17
- **Landed:** brief (PR #20), freeze + α operator + runner + results + close (PR #21), CLAUDE.md state update (PR #22)
- **Result:** "does not route" ×3 against the would-gate, but a CI-clean J-specific routing signal at α=1 on all three subjects; the paper's dose direction inverts (α=2 blurts instead of rescuing) [Fact — PR #21 body] — see D19–D22 in `docs/DECISIONS.md`

### S3 — selectivity — 2026-07-17
- **Landed:** brief (PR #23), freeze + ablation operator + runner + results + close (PR #24)
- **Result:** selectivity-consistent on all three subjects — the project's first pre-committed would-gate to hold everywhere; owned as *relative* selectivity [Fact — PR #24 body] — see D23–D26 in `docs/DECISIONS.md`
- **Tradeoff:** ablation operator shipped as modified Gram-Schmidt after the read-back gate caught a least-squares blow-up and LAPACK SVD failed to converge [Fact — PR #24 body]

### S4 — naming vs avoiding — 2026-07-17
- **Landed:** brief (PR #25), freeze + first constructed item set + runner + results + close (PR #26)
- **Result:** avoidance dissociation NOT shown ×3, but the late-band off-switch reproduces as a hard switch at every scale, and only 1.5B passes the competence gate [Fact — PR #26 body] — see D27–D30 in `docs/DECISIONS.md`

### S4b — late-tier specificity control — 2026-07-17
- **Landed:** follow-up brief (PR #27), control cells + three-subject re-run + close (PR #28); every shared cell reproduced bit-for-bit vs the S4 JSONs
- **Result:** the late-band off-switch is concept-specific on the powered subject (1.5B +.727 CI-clean), with specificity emerging with scale [Fact — PR #28 body] — see D31 in `docs/DECISIONS.md`

## Era: Write-up and wrap (2026-07-18 – 2026-07-26)

Measurement done; the repo turned to documentation artifacts — no code changes,
no new measurements.

### Tooling vendor sweep — 2026-07-18
- **Landed:** fleet-wide /claudify-repo sweep vendoring global commands + skills into `.claude/` (PR #30)

### Research paper + presenter pack — 2026-07-19
- **Landed:** `docs/paper/` paper + presenter pack from recorded results (PRs #29, #31), M0/S4b figures added (PR #32); every statistic mechanically re-verified from `results/*.json` [Fact — PR #31 body]
- **Why:** two parallel write-up runs landed; the skill-generated set was kept as the official paper set [Fact — merge commit `5e0cfee` message]

### Plain-English rewrite + paper consolidation — 2026-07-19
- **Landed:** 1:1 plain-English rewrite of the paper via /paper-eli5 (PR #33); both paper folders consolidated into `docs/paper/` (PR #34)

### Project wiki initialized — 2026-07-26
- **Landed:** PROJECT.md, HANDOFF.md, Sources.md + CLAUDE.md wiring (PR #35)
- **Why:** no root Decisions.md was created — `docs/DECISIONS.md` (D1–D31) is already the authoritative append-only ledger [Fact — PR #35 body]

---

## Mining coverage
_Backfilled 2026-07-26 by project-wiki BACKFILL. Entries after this date are
appended live by MAINTAIN._
- PR title sweep: all 35 merged PRs — no cap
- Deep reads: 20 of 35 PRs (size/label/title signal; cap 20): #1–4, #8–12, #14, #16, #17, #19, #21, #24, #26, #28, #30, #31, #35
- Also swept: git log (merges/no-merges), tags (none exist), `docs/DECISIONS.md` (ID anchoring only), `docs/KICKOFF.md`, `docs/ROADMAP.md`, `docs/LEARNING.md`, stage briefs (`docs/M0–M3-BRIEF.md`, `docs/S1–S4-BRIEF.md`)
- Wrap logs: none found (no `docs/session-logs/` or `.claude/session-logs/`)
- Not mined: closed-unmerged PRs, issues
