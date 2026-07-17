# LEARNING — dim-stage

*Teaching notes, appended at each stage close. Plain English first; every
jargon term defined at first use. The goal: Kyle can defend every decision.*

## M0 — the fit pilot (2026-07-15)

### The one-paragraph story

We built our own copy of the paper's **Jacobian lens** — a tool that reads a
transformer's *middle* layers by first transporting each middle-layer vector
into the final layer's coordinate system (multiplying by `J_l`, the average
sensitivity matrix) and then decoding it with the model's own output matrix.
We proved our copy computes the *exact same object* as Anthropic's reference
implementation (bit for bit), fitted it to both small Qwen models, and then
asked the frozen question: can this lens recover known "workspace content"
(an evoked concept, a latent reasoning step, a planned rhyme) from the middle
layers at least half the time? **Answer: no, on both models — a clean,
pre-registered null.** That null is the headline result the kickoff predicted
might happen, and it now reshapes the rest of the project.

### Vocabulary that now has meaning from experience

- **Residual stream** — the running vector each transformer layer reads and
  adds to; one per token position. The thing the lens reads.
- **Cotangent / VJP** — backprop's "direction of interest": put a 1 in one
  output coordinate, run backward, and you get that coordinate's sensitivity
  to every input — a **v**ector–**J**acobian **p**roduct. Build `J_l` one row
  (or 8 rows — `dim_batch`) at a time this way.
- **Frobenius norm** — a matrix's overall size (square root of the sum of
  squared entries). "Relative Frobenius distance 0" = the matrices are
  identical.
- **Wilson interval** — the honest range a true rate sits in given only n
  trials; never escapes [0,1]. Our gate turned on its *lower bound*: "even
  pessimistically, is the rate ≥ 0.5?"
- **Newcombe interval** — the same idea for a *difference* between two rates;
  if it straddles 0, no gap may be claimed. This is what made "J-advantage"
  a claim on typo at 0.5B (+43pp, CI [+31, +53]) and *forbade* it at 1.5B.
- **Logit lens (`J = I`)** — the old trick of decoding a middle layer
  directly, no transport. Kept as a standing falsification arm: any J-lens
  claim must beat "doing nothing."
- **Forking paths** — tuning analysis choices after seeing results until
  something passes. Every convention here (band, synonym table, token
  variants, tolerance) was frozen *before* the first official number existed.

### Hard-won, non-obvious lessons

1. **MPS fp32 backward is bitwise deterministic** (this machine, torch
   2.13.0): two reference refits produced *identical* matrices, so the
   run-to-run "noise floor" was exactly 0 and our pre-declared 1e-4 stand-in
   carried the tolerance. Determinism is not hand-waving here — we measured it.
2. **The AGREE gate result was stronger than required** — bitwise identity,
   not just within-tolerance. Two independently written codebases hitting the
   same bits is what "the estimator spec was extracted correctly" looks like.
3. **Batching knobs can be math-neutral and still 25× slower** — dim_batch=32
   replicated the retained graph past MPS unified-memory locality. Measure,
   don't assume.
4. **Even CPU-side work slows an MPS job** (~8% on our fit) — GPU kernels are
   driven by CPU threads. Serial beats concurrent for measurements.
5. **A null with structure beats a bare null.** Abstract-content
   distributions were hard zeros at both scales; surface-adjacent ones sat at
   33–54%; and the Jacobian correction's value *inverted* with scale (strongly
   positive at 0.5B on typo/multilingual, significantly *negative* on typo at
   1.5B as the plain logit lens caught up). Those are reportable observations
   even though no gate claim is made.
6. **Grading conventions the paper doesn't ship are decisions you own.** The
   intermediate→token mapping (leading-space variants, synonym table) had to
   be invented — so it was frozen in code, pre-declared, and rowed in the
   deviations table before any run.

### Recall questions (answers in this repo's docs)

1. Why did the AGREE tolerance need a *measured* noise floor instead of a
   guessed one — and what did we do when the floor came back exactly 0?
2. The 1.5B typo cell shows the J-lens at 17.7% and the logit lens at 44.8%.
   Why can we *report* that reversal but not claim "the J-lens hurts" as a
   gated finding? (Hint: which arm was the gate pre-registered on?)
3. Why does the readability verdict use the Wilson lower bound rather than
   the raw pass@10 percentage — what failure mode does that choose to avoid?

## M0 — 3B escalation (2026-07-15 evening)

### The one-paragraph story

Both small models came back null, which was the *pre-registered trigger* to try
one bigger model (3B) — the only way to ask "does the workspace become readable
if we add scale?" We froze the 3B analysis band **before** fitting (so we
couldn't move the goalposts after seeing results), discovered the laptop
physically can't fit a 3B model in full precision, rented a cloud GPU for **83
cents** to do just the fit, then graded locally exactly like the other two. The
answer: **still null.** No distribution's readout clears the bar at 0.5B, 1.5B,
or 3B. That's the strongest honest version of this project's headline — a
clean, three-scale, pre-registered *no* to the paper's open question, at the
scales a hobbyist can reach.

### What was learned

7. **A memory ceiling is a hardware fact you measure, not guess.** The Mac's
   24 GB of unified memory (shared by CPU and GPU) holds a 1.5B model's
   full-precision weights (6.2 GB) with room for the backward pass, but a 3B
   model's weights (12.4 GB) sit right at the edge — and the backward pass tips
   it over a "working-set cliff" where the GPU thrashes and runs ~25× slower.
   Halving the batch didn't help because the *weights themselves*, not the
   batch, are what overflow. Lesson: **fp32 fitting on this machine tops out
   between 1.5B and 3B** — a boundary now written into memory, so no future
   session re-discovers it by burning an evening.
8. **Rent, don't fight, when the tool is wrong for the job.** A datacenter GPU
   (CUDA, no such cliff) fit the 3B lens in ~57 minutes at 34.5 s/prompt — the
   same prompt the Mac couldn't finish in 40. The whole detour was ~$0.83 and an
   *owned deviation row*, not a silent move: cross-device numeric noise on a
   corpus-averaged matrix is ~1e-7, orders below any gate's sensitivity, and the
   fit was byte-identical procedure (same code, same prompts). We verified the
   returned file by checksum before trusting it.
9. **A null that survives escalation is stronger than the null that triggered
   it.** The double null was a headline; the triple null *closes the question* —
   it says the emergence point, if one exists, is above 3B, outside the reachable
   range. And the structure sharpened: association and poetry (the abstract,
   not-yet-spoken content — the most workspace-like) are hard zeros at *all
   three* scales, while the Jacobian correction's value turned out **content-
   dependent, not a scale trend** (typo's reversal deepens with scale; order-ops
   *regains* a J-advantage at 3B).
10. **Make the oracle's own output honest.** The gate's console line printed "no
    clear gap" for a statistically-clean *reversal* (the J-lens reading
    significantly *worse* than the plain baseline). We added a distinct
    `J-REVERSAL` label — a finding hidden by a lazy print is still a hidden
    finding, and in a project whose whole worth is faithful measurement, the
    console should never under-report what the data shows.

### Recall questions (answers in this repo's docs)

4. Why did we freeze the 3B band (L14–L32) and merge it *before* the fit ran,
   rather than after we had the lens in hand? What would have been wrong with
   waiting?
5. The 3B fit ran on a rented CUDA GPU while the grading ran on the local Mac.
   Why is that split defensible — what makes the device change a *small* owned
   deviation rather than a result-invalidating one?
6. "association" and "poetry" are hard zeros at 0.5B, 1.5B, *and* 3B. Why is
   that the most interesting part of the null, given what those two
   distributions are asking the lens to read?

## M1 — verbal report + introspection (2026-07-16)

### The one-paragraph story

M1 moved from *reading* the workspace to *writing* it. We built the paper's
two intervention operators — the **swap** (exchange one concept's coordinates
for another inside an activation, touching nothing orthogonal to the pair)
and **steering** (add a scaled concept direction) — and, because Anthropic
shipped no intervention code, gated them with **pre-committed invariant
tests** instead of a reference diff: a rigged tiny model whose Jacobian is
known exactly, with numbers chosen so every post-patch logit is asserted *to
the bit*. Then both protocols ran on all three subjects, descriptive per the
triple null. Swapping the model's ready answer mid-forward barely moves its
spoken report (top-5 rates 10–18% against the paper's 88% anchor). But
*steering a thought in and asking the model to name it* produced the
project's first dose–response curve: at 1.5B the report rate climbs 0 → 30%
with strength while the zero-strength control stays exactly 0/101 — and the
same curve is flat at 0.5B and barely rises at 3B.

### What was learned

11. **When there is no oracle, invariants are the gate.** The AGREE pattern
    needs a reference to diff against; none exists for interventions. The
    replacement discipline: hand-build a model where the *true* answer is
    analytic (dyadic numbers → exact equality, no tolerances to hide a sign
    or transpose bug), and re-verify the core property (coordinates exchange)
    on every *real* application at runtime. The runtime check stayed silent
    across ~550 swaps — that silence is now evidence, not hope.
12. **An intervention can "work" and still miss the headline.** Steering
    moved the median concept's rank ~3× at 0.5B, ~300× at 1.5B, ~10× at 3B —
    directionally right on every subject. Only 1.5B converts movement into
    rank-1 reports. "Did the needle move" and "did it arrive" are different
    questions; report both or the summary lies.
13. **A control earns its keep in one number.** α = 0 giving exactly 0/101 on
    every subject kills the best alternative story ("the prompt begs for
    concept words at an open quote") for free. Every strength cell above it
    inherits that interpretability.
14. **Scale structure keeps refusing to be monotone.** M0's multihop peaked
    at 1.5B and regressed at 3B; the introspection curve does the same, with
    the 1.5B–3B gap CI-clean. At hobby scale, "bigger model" is not a
    monotone knob — and the interesting subject in this repo is repeatedly
    the middle one.

### Recall questions (answers in this repo's docs)

7. The 1.5B introspection curve is the most striking result in the project so
   far — yet it can never gate M1's verdict. Which frozen decision made that
   so, and what forking-paths discipline does it protect?
8. The α = 0 control came back exactly 0/101 on all three subjects. Which
   alternative explanation of the 1.5B curve does that single number
   eliminate?
9. M1's correctness gate is invariant tests instead of M0's AGREE diff. What
   made the AGREE pattern impossible here, and which class of bug do the
   invariants provably catch — and which class can they *not*?

## M2 — two-hop swap (2026-07-16)

### The one-paragraph story

M2 aimed the M1 swap one hop upstream: not the answer the model is about to
say, but the **bridge entity** it silently computes on the way ("the country
where the Amazon River ends" → *Brazil* → *Portuguese*). Swap Brazil for
Mexico mid-reasoning and ask: does the model now say *Spanish*? On the
paper's own 90 prompts Claude flips 60% of the time. Our subjects mostly
don't — flip rates run 7–29% on the chains that worked unswapped — but the
milestone still produced the project's most surprising cell: at 3B, swaps
along **raw unembedding rows flip nothing at all (0/43)** while J-lens
vectors flip 5/43, a CI-clean difference. Everywhere before this, the
Jacobian transport had looked useless or harmful; here it is, for the first
time, the entire effect.

### What was learned

15. **A standing falsification arm cuts both ways.** The J = I arm existed to
    *falsify* J-lens claims (and did, at M0's typo cell). At M2's 3B cell it
    did the opposite: raw rows 0/43, J-lens 5/43, Newcombe +.116
    [+.011, +.245] excluding zero — the first evidence in this project that
    the transport itself carries causal freight for writing. Keeping the
    boring control in every run is what made the interesting cell credible.
16. **Condition on capability or conflate two failures.** Baseline two-hop
    accuracy is .35 / .51 / .53 (Claude: ~1.0). The 0.5B flip rate *looks*
    highest (.286) — but only 28 flimsy chains existed to flip. D9's
    baseline conditioning is what keeps "the swap redirects reasoning" from
    being polluted by "there was no reasoning to redirect."
17. **Name the confound, then spend 180 forwards on it.** The paper worried
    intermediate swaps might work by smuggling in answer components. Our
    cheap version of their check — swap the answers directly and compare —
    came out clean at 3B (6 vs 5 of 43, and a smuggled answer would survive
    J = I, which flips nothing). Not the paper's depth sweep; owned as a
    deviation, but the confound didn't go unexamined.

### Recall questions (answers in this repo's docs)

10. The 3B J − identity CI (+.116 [+.011, +.245]) is the project's first
    clean J-transport advantage for writing. Arm 2 never gates — so what
    exactly may we say about this cell, and in which framing?
11. Why does M2's primary cell only count items the model answered correctly
    before the swap — and what does the 0.5B flip rate (the "highest" of the
    three) illustrate about the unconditioned alternative?
12. The answer-swap arm flips twice the intermediate rate at 0.5B but equals
    it at 3B. Which confound was that arm built to probe, and why is our
    single-band version weaker evidence than the paper's layer-range sweep?

## M3 — directed modulation (2026-07-16)

### The one-paragraph story

M3 asked whether the model can steer its own workspace **on command**: told
to think about citrus fruits (or to ignore them) while copying an unrelated
sentence, does *orange* surface in the lens over text that gives it no
reason to appear? This is a **reading** milestone — the modulation is the
instruction; no activations are edited — which also meant owning a
correction: M1-D8's claim that M3 needed the steering operator was wrong.
The frozen verdict came back "does not modulate" on all three subjects
(math content never surfaces, and the gate needs both families), but the
inside of that verdict is the most paper-shaped structure the project has
produced: on concrete category content the signal is **ordered exactly as
the paper describes** (focus ≫ mention ≈ suppress ≈ baseline ≈ 0), CI-clean
at 1.5B and 3B, growing CI-cleanly with scale — and, at 3B, read almost
twice as well by the *plain logit lens* as by the J-lens.

### What was learned

18. **A frozen gate can say "no" while the data says "something."** The
    both-families wording was frozen before any run; math's hard zero makes
    the verdict "does not modulate" even though the category family shows
    the paper's exact ordering with CI-clean contrasts. Both facts get
    reported — the gate verdict AND the structure inside it — and neither
    edits the other after the fact. That is what pre-registration buys: the
    interesting cell can't quietly become the gate.
19. **An effect can reproduce while its instrument choice inverts.** The
    modulation ordering reproduces qualitatively; but at 3B the J-lens is
    the *worse* reader of it (9/110 vs the logit lens's 19/110, CI-clean).
    The paper's claim is about the J-space specifically; at our scales the
    instructed content sits somewhere the raw unembedding reads more
    directly. Keeping the J = I arm on every reading experiment is what
    made this visible at all.
20. **Some phenomena have preconditions, and small models can fail the
    precondition rather than the phenomenon.** The white-bear effect needs
    thoughts that intrude uninstructed — but our baselines and suppress
    cells are all ≈ 0: nothing enters the workspace on its own, so there is
    nothing for suppression to visibly fail at. "No white-bear effect" here
    means the *setup* for it doesn't arise, not that the effect was tested
    and absent. Distinguishing those two readings is the difference between
    a null and a non-test.

### Recall questions (answers in this repo's docs)

13. M3's would-gate says "does not modulate" on every subject, yet the brief
    calls the category-family result "the most paper-shaped structure the
    project has produced." How are both statements true at once, and which
    frozen decision keeps them from contradicting?
14. At 3B the logit lens reads the focused concept on 19/110 trials against
    the J-lens's 9/110, with the difference CI-clean. Why does this
    *strengthen* rather than undermine the case for keeping a J = I arm on
    every experiment — and what does it say about where instructed content
    lives at these scales?
15. Why is "no white-bear effect" at these scales a non-test rather than a
    null result — what precondition of the effect never arises?

## S1 — introspection dose–response follow-up (stretch) (2026-07-16)

### The one-paragraph story

M1 left the project with one live, positive result: at 1.5B a steered-in
thought becomes reportable, 0 → 30/101 as we push harder. But it was the
*only* headline finding with no **falsification arm** — nothing tested whether
that curve was really about the Jacobian transport (the "workspace"), or just
about steering along *any* direction built from the token. S1 (the first
stretch stage) closed that hole and two more. It re-ran the whole 1.5B curve a
second way — **`J = I`**, steering along the raw unembedding row with the
transport removed (the plain logit lens) — and the J-lens beat it CI-cleanly
from α=1, roughly *doubling* the report rate: the dose–response is a genuine
transport effect, exactly the paper's specificity claim. It **extended the
strength grid** past M1's still-rising α=8 and watched the curve *saturate*
(~30/101, the paper's Figure-7 shape) without the model ever breaking — a
deterministic **collapse guard** confirmed the plateau was real reporting, not
garbage. And it **localized** the effect: steering only the middle five layers
(L16–20) recovered essentially the whole thing (29 of the full band's 31) — the
paper's mid-layer "middle block," found at hobby scale.

### What was learned

21. **A single strong result isn't defensible until it survives its own
    falsification arm.** The 1.5B dose–response *looked* like a workspace effect
    for a whole milestone, but nothing ruled out "steering along any
    token-direction does this." Re-running it with the transport removed
    (`J = I`) and getting a CI-clean gap (J-lens ~2× the raw arm) is what turns
    "a curve" into "a *J-transport* curve." The lesson generalizes: the arm that
    could have killed the finding is the one that certifies it.
22. **Extending a grid is cheap and can pay twice.** M1 stopped at α=8 with the
    curve still rising, so "does it saturate?" was simply unanswered. Adding
    {12, 16, 24} cost minutes and (a) showed 1.5B *plateaus* — the paper's shape,
    not a runaway — and (b) revealed something M1 never saw: 3B's small reporting
    signal keeps climbing to 9/101 and is *entirely* transport-specific (its
    raw-unembedding arm is dead). A convention frozen too tight (α≤8) had hidden a
    real cross-scale result.
23. **"Where" is a different question from "how much," and it has a cleaner
    answer.** Steering all 14 band layers at once told us the effect's *size*;
    steering thirds separately told us its *location* — and the middle five layers
    alone reproduced it. A rate became an internals claim ("the reportability lives
    in the middle of the band, matching the paper's block structure") without any
    new model or fit — just by restricting *where* the same operator applies.
24. **A good degeneracy guard needs the token's identity, not just its
    frequency.** "One token wins for most concepts" means opposite things at low α
    (the model ignoring a weak steer, still saying its default word) and high α
    (collapse to a junk fixed point). The guard only flags collapse by also
    checking the winning token is *new* (≠ the α=0 control token) — so weak
    steering and broken steering don't get confused. It never fired here, which is
    itself the finding: the extended grid broke nothing.

### Recall questions (answers in this repo's docs)

16. The 1.5B injected-thought curve was already the project's strongest result
    after M1. What specifically did S1's `J = I` arm add that M1's version
    couldn't claim — and why does a *doubling* over the raw-unembedding arm
    matter more than the raw 30/101 number?
17. Steering only layers L16–20 recovered 29/101, and the full 14-layer band
    only 31/101, with the difference's CI overlapping zero. What does that let
    you say about *where* the 1.5B workspace lives — and which paper claim does
    it echo?
18. S1's collapse guard flags an α only when one token wins ≥50% of concepts
    AND that token differs from the α=0 control token. Why is the second
    condition essential — what would go wrong if the guard used the 50%
    share alone?
