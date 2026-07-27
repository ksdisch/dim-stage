# Methodology Guardrails

## Purpose
Captures the full integrity system — deterministic oracles, CI rules, N thresholds, gate pre-commitment, design-extraction rider, and deviation ownership — assembled from the places it lives: [CLAUDE.md](../CLAUDE.md), [docs/KICKOFF.md](../docs/KICKOFF.md), the per-stage briefs, and [docs/DECISIONS.md](../docs/DECISIONS.md). CLAUDE.md lists the rules as bullets; the briefs show how each was applied stage by stage; DECISIONS.md records the one measured deviation (the 3B MPS cliff). This page synthesizes the *system* — what each guardrail is, why it exists, and the evidence from actual runs that it worked.

## Key understanding

### 1. Deterministic oracles only

**Fact** ([CLAUDE.md](../CLAUDE.md), "Methodology guardrails"): outcomes are logit rankings read from tensors — rank-1 hits over the workspace band, greedy next-token, swap grading = rank of the swapped-in candidate. Never an LLM judge, never text parsing.

**Fact** (D1 outcome, [docs/DECISIONS.md](../docs/DECISIONS.md)): MPS fp32 backward pass is bitwise deterministic on this hardware (torch 2.13.0): two reference refits produced identical matrices and the run-to-run noise floor was exactly 0. The calibrated tolerance of 1e-4 stood in for a measured floor that came back as zero.

**Why it matters**: the paper's "honesty contract" is about faithfulness to mechanical outputs, not interpreted outputs. A logit ranking from a greedy pass is an exact number; an LLM-judge verdict is not. Every gate threshold in the project is a CI condition on a binary count — which requires a binary oracle in the first place.

### 2. Wilson CIs on cells; Newcombe CIs on differences

**Fact** ([CLAUDE.md](../CLAUDE.md)): "Wilson CIs on cells + Newcombe CIs on differences decide every gate." The paper reports Wilson 95% CIs natively (see KICKOFF quantitative anchors), making anchor comparisons like-for-like.

**Fact** (D4, [docs/M0-BRIEF.md](../docs/M0-BRIEF.md)): the readability gate uses Wilson 95% *lower bound* ≥ 0.5, not the raw pass@10 rate. Using the lower bound is the conservative choice: it answers "even pessimistically, is the rate ≥ 0.5?" A raw rate above 0.5 with a wide CI (small N) would pass under a naive threshold and fail under a Wilson LB gate.

**Fact** (D10, D14, D26, etc., [docs/DECISIONS.md](../docs/DECISIONS.md)): every "J-advantage" or "J-reversal" claim across all stages uses Newcombe 95% CI on the difference. A CI that straddles zero produces no gap claim. This was the mechanism that prevented "no clear gap" from being confused with "CI-clean reversal" — the console-label fix (adding J-REVERSAL vs no-gap labels, DECISIONS.md 3B rescue section) made this explicit in the oracle's own output.

### 3. N ≥ 20 per cell or pre-declared UNDERPOWERED

**Fact** ([CLAUDE.md](../CLAUDE.md)): "N ≥ 20 per cell or the verdict is pre-declared UNDERPOWERED. Trials are free — N is bound by wall-clock, not dollars."

**Fact** (D29, [docs/DECISIONS.md](../docs/DECISIONS.md)): at S4, the competence gate produced n=5 at 0.5B and n=8 at 3B — below the threshold. These were tagged UNDERPOWERED before any run and reported as such. The S4 gate is "NOT shown" on all three subjects; the UNDERPOWERED tag on the smaller subjects is owned alongside, not hidden.

**Inference**: the threshold is not a floor for excluding data — it is a floor for making claims. Every sub-threshold cell still appears in the results with its CI; the UNDERPOWERED label tells the reader what can't be concluded, not what wasn't measured.

### 4. Pre-commit gates as code + dry-run before real runs

**Fact** ([CLAUDE.md](../CLAUDE.md)): "Pre-commit gates as code and dry-run them (wrong-arm input exits INVALID) before any real run. A pre-committed null is a reportable result."

**Fact** (D6, [docs/DECISIONS.md](../docs/DECISIONS.md)): the M1 intervention operators could not have an M0-style AGREE gate (the reference ships no intervention code). The replacement was pre-committed invariant tests: a rigged tiny model with an analytic exact-equality oracle, coordinate read-back, and null-op checks — merged before any real run. The D6 runtime read-back then re-verified the core invariant (c′ = σ(c)) on every swap application across all stages. It stayed silent across every real run, which is itself evidence.

**Fact** (D4, M0-BRIEF): the readability gate's wrong-arm dry-run (0.5B model + 1.5B lens) exited INVALID before the official runs, verifying the guard fired.

**Fact** (D26, S3 outcomes, [docs/DECISIONS.md](../docs/DECISIONS.md)): the S3 ablation operator failed the D6-style read-back on the first real smoke run — a least-squares projection blew up on ill-conditioned real direction sets, then LAPACK SVD failed to converge. The pre-committed read-back caught this before any stage result was recorded; the operator shipped as modified Gram-Schmidt.

**Fact** (S3, [docs/DECISIONS.md](../docs/DECISIONS.md)): the same stage surfaced a silent MPS `.to("cpu", torch.float64)` value-corruption bug — the cast happens device-side where float64 is unsupported, no exception raised, ~unit-scale errors. Fixed (move-then-cast: `.cpu().to(float64)`). Without the read-back gate this would have produced plausible-looking garbage as a stage result.

**Inference**: the dry-run + runtime-check pair is the project's most operationally valuable guardrail: it caught two implementation failures (S3 operator, S3 MPS bug) that would have silently invalidated results.

### 5. Design-extraction before design-signing

**Fact** ([docs/KICKOFF.md](../docs/KICKOFF.md), honesty contract, "bar entry 9"): extract the paper's/repo's actual procedure verbatim as a free pre-commit step in every milestone brief.

**Fact** (all milestone briefs): each begins with a "Design extraction" section quoting the paper or reference README verbatim. Conventions the reference does NOT specify (intermediate→token grading, synonym tables, strength-grid values, sub-band thirds) are owned in deviations rows — not paper claims.

**Fact** (D3, M0-BRIEF): the grading convention for the intermediate→token mapping had to be invented because the paper's eval-grading machinery doesn't ship. It was frozen in code and listed as a deviations row *before* any run. "Every added synonym monotonically helps pass@k — choosing them after seeing results would be forking paths."

**Inference**: the extraction-before-signing discipline is the primary defense against the "forking paths" failure mode — where analysis choices are made or adjusted after partial results are visible, unconsciously selecting for findings. By extracting what the paper specifies, then listing what isn't specified as owned deviations, every degree of freedom is visible before it's exercised.

### 6. Deviation ownership

**Fact** ([CLAUDE.md](../CLAUDE.md)): "Deviations are owned. Model scale, fit-corpus size (~100 vs paper-grade 1000), single-token tokenizer filter, MPS vs CUDA — each is a row in a deviations table."

**Fact** (M0 deviations table, [docs/M0-BRIEF.md](../docs/M0-BRIEF.md)): eight rows by M0 close, covering: subjects (scale), fit-corpus (WikiText vs web-text, N=100 vs 1000), hardware (MPS vs datacenter), band selection (proportional vs derived), single-token pre-filter, intermediate→token mapping, 3B CUDA fit, and HF-streaming corpus delivery.

**Fact** (3B rescue, [docs/DECISIONS.md](../docs/DECISIONS.md)): the 3B fit moved to a rented CUDA GPU (the pre-declared fallback, ~$0.83) rather than being abandoned or silently run on different hardware. The deviation was owned: cross-device fp32 noise on a corpus-mean matrix is ~1e-7 (orders below gate sensitivity); the fit log's per-prompt `seq_len=128 n_valid=111` signature matched the MPS runs, confirming corpus identity.

**Inference**: deviation ownership is what distinguishes a project with a clear evidentiary claim from one where "we got 30%" might mean any of several different measurements. Readers can assess which deviations matter for which claims.

### 7. Verdict wording frozen before results

**Fact** (D4, D7, D10, D14, D22, D26, D30, D31, all in [docs/DECISIONS.md](../docs/DECISIONS.md)): for every milestone, the exact wording of "READS / NULL," "routes / does not route," "modulates / does not modulate," "selectivity-consistent," "avoidance-dissociation-consistent / NOT shown," and "concept-specific" was frozen before the runners produced any real-subject output. Wrong-arm dry-runs confirmed the guards fired before gate-wording was exercised.

**Fact** (M3 example, D14 outcome, [docs/DECISIONS.md](../docs/DECISIONS.md)): the gate needed both families (category + math); math is a hard zero. The frozen two-families wording means the CI-clean category signal cannot quietly become the gate after the fact. Both facts are reported — the gate verdict AND the structure inside it — and neither edits the other.

**Inference**: the "wording frozen before results" rule is what gives pre-registered null legs credibility. S4 D30 Leg 2 ("naming is spared") holds everywhere — that is a claim of no-effect that would be unfalsifiable if invented after the data. It was pre-declared, so it counts.

### Owned deviations that bound interpretation

**Fact** (standing across all stages, [docs/KICKOFF.md](../docs/KICKOFF.md) and M0 deviations table): the four structural deviations that most bound what can be concluded from this project:

1. **Model scale**: 0.5B–3B vs paper's Claude/Qwen3.6-27B. The project is explicitly measuring whether the workspace is readable at a scale the paper hasn't tested — the null is the thesis, not a failure.
2. **Fit corpus**: N=100 WikiText-103 vs paper's N=1000 "pretraining-like web text." The paper's §9.3 says quality saturates quickly (~10 prompts for J to beat logit lens); this deviation is small in practice.
3. **Single-token tokenizer filter**: Qwen's digit-by-digit tokenization drops some multi-digit numbers and other multi-token targets. Every result is conditioned on the gradable items surviving this filter; every cell reports N after filtering.
4. **MPS vs CUDA** (3B fit only): cross-device fp32 noise on a corpus-mean matrix is ~1e-7, orders below any gate's sensitivity. The 3B lens was transferred sha256-identical and its per-prompt diagnostics matched the MPS runs.

## Sources
- [CLAUDE.md](../CLAUDE.md) — "Methodology guardrails (load-bearing)" bullet list; the compact authoritative statement of all seven rules
- [docs/KICKOFF.md](../docs/KICKOFF.md) — "Honesty contract" section; "Riskiest assumptions" for the statistical design rationale; deviations 1–4 at origin
- [docs/M0-BRIEF.md](../docs/M0-BRIEF.md) — all four decision sections (D1–D4) with the full options and rationales; M0 deviations table; AGREE and readability gate results
- [docs/DECISIONS.md](../docs/DECISIONS.md) — D1–D31: every instance where a guardrail was invoked, exercised, or caught something
- [docs/LEARNING.md](../docs/LEARNING.md) — per-stage teaching notes; hard-won lessons 1–36 describe specific guardrail incidents in plain English

## Uncertainties & contradictions

- **Unresolved**: the "wrong-arm dry-run exits INVALID" check was verified for M0 and described in briefs for M1–S4b, but the DECISIONS log does not record a dry-run failure for every stage beyond M0. It is possible later stages' dry-runs ran silently-correct rather than being independently verified.
- **Inference**: the sub-band-thirds localization (D18, S1) is an owned convention rather than a design extracted verbatim from the paper. The paper's localization machinery uses CKA block structure over all layers; the project's version (three contiguous thirds steered separately) is a rougher but legible substitute. Results from the two approaches may not be directly comparable.
- **Contradiction on record** (from [docs/DECISIONS.md](../docs/DECISIONS.md), M3 section): "M1-D8's rationale claimed the steering operator was 'required for M3.' Wrong: M3 is a reading milestone." Owned and retracted; D8's other merits stand. Cited here as an example of the project's self-correction discipline.

## Related pages
- [Results-Synthesis](Results-Synthesis.md) — the full cross-stage results that the guardrails produced
- [History](History.md) — when each guardrail was first applied, in chronological order

## Relevance to current work
v1 is closed. The guardrails are the project's permanent methodological record. Relevant for: portfolio defense (each claim's basis), any future project in the lineage (this system is portable and is already the lineage standard), and as a reference when the honesty-contract question comes up in an interview context.

_Last reviewed: 2026-07-26_
