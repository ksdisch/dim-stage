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

## S2 — flexible generalization (stretch) (2026-07-16)

### The one-paragraph story

The workspace story's boldest property is **broadcast**: one representation,
written once, consumable by *many* downstream circuits. The paper's test is
elegant — sixteen different one-line questions ("the capital of {arg}", "most
people in {arg} speak", …), one identical swap of the argument's lens
coordinates clamped everywhere, and the question is whether *every* circuit
now computes on the swapped-in concept (76/192 at α=1 for Sonnet, 101/192
when the swap is doubled to α=2). At our scales the answer has a sharp and
unexpected shape. There **is** a routing signal, and it is CI-cleanly
**J-transport-specific** (J−I: 0.5B +.078, 1.5B +.061, 3B +.078 — the raw
unembedding rows manage only 3–5/180) — but it lives *only at α=1*, an order
of magnitude below the anchor (17/16/18 of 180 across 0.5B/1.5B/3B). At α=2 —
the dose that *rescues* the paper's near-misses — our subjects don't improve,
they **blurt**: the greedy output becomes the swapped-in argument itself
(" France", " China") instead of the function's answer, and the true target
answer falls out of the top ranks (at 1.5B all the way to the vocabulary
floor). Category structure reproduces where it can
(1.5B matches the paper's order exactly: countries > months > animals >
numbers; the numbers category loads weakest in the lens at every scale,
just as the paper predicts) — and the conditioned frame shows the zeros are
often about *knowledge*, not routing (no subject can even answer "Two times
three equals" unswapped — they all continue "… what").

### What was learned

25. **An anchor's dose direction is itself a scale-dependent claim.** The paper
    reads its α=1 failures as "moved the right way, not far enough," and α=2 as
    the fix (76 → 101). Our subjects run the same protocol and *invert* it
    (17 → 1, 16 → 0, 18 → 1). The per-trial records say why: at α=2 the model
    stops computing *with* the injected concept and starts *saying* it — the
    argument token becomes the greedy output and the function's answer is
    anti-ranked to the vocab floor (median rank ~151,844 of 151,936 at 1.5B).
    "Double strength" presupposes headroom; a small model's routing window is
    narrower, not a scaled-down copy of the same curve.
26. **A conditioned frame turns a null into a diagnosis.** Unconditionally,
    numbers is 0/48 at every scale — but the diagonals show *zero* subjects
    answer the unswapped numbers prompts at all (they continue "Two times three
    equals" with " what", or bare whitespace at 3B: a pragmatics failure of raw
    completion, not arithmetic). Where both facts are demonstrably known,
    routing at α=1 is 13/42 (0.5B), 12/62 (1.5B), and 16/56 (3B) — roughly
    3–4× the unconditional rate. Without D22's conditioning
    (M2's lesson, now standing equipment), "can't route" and "never knew" would
    have been one indistinguishable zero.
27. **A mechanism's cheap correlate can reproduce even when its headline
    doesn't.** Workspace loading — the paper's own predictor — puts numbers
    lowest at every scale (matching its worst-category prediction exactly), and
    it *sharpens with scale*: at 0.5B/1.5B the top end doesn't transfer (animals
    loads highest but routes poorly), while at 3B even the top aligns (countries
    load highest and route best, .125 over .04–.11). Partial reproduction is
    signal, not failure: "absent from the lens ⇒ can't be routed" appears
    scale-stable, while "present ⇒ will be consumed" only becomes true as the
    subject grows.
28. **A guard that mostly stays silent is what makes both its silences and its
    one catch meaningful.** The generalized D6 read-back ((1−α)·c + α·σ(c))
    stayed silent at every α on every subject — the operator always did exactly
    what the algebra says. The degeneracy guard stayed silent through the α=2
    cliff (the blurted outputs are *real words*: behavioral failure, not
    numerical) and fired exactly once, at 3B α=8, where both arms collapse to a
    `!` attractor at share 1.00 and the logits saturate into mass rank-1 ties —
    which is also why success is graded on greedy membership, never on rank;
    rank texture is meaningless inside a collapse. Three regimes, cleanly
    instrumented: routed (α=1), blurted (α=2, guard silent), junk (3B α=8,
    guard fires).

### Recall questions (answers in this repo's docs)

19. S2 found J − identity CI-clean at α=1 on all three subjects
    (+.078/+.061/+.078). M2 found identity rows flip *nothing* at 3B. Why does
    running the identity arm still matter here even though it "already lost" in
    M2 — what specifically would an identity-arm *win* at α=1 have said about
    the 17/180?
20. The paper's α=2 rescues near-misses; ours destroys hits. Describe the
    measured failure mode at α=2 in one sentence, and name the two pieces of
    evidence (greedy identity, target-answer rank) that distinguish "blurting
    the argument" from "generic model collapse."
21. The numbers category scored 0/48 unconditionally at every scale. Why is
    that *not* evidence that small Qwen models can't route number concepts —
    and which two S2 readouts (one from the diagonals, one from the paper's
    own predictor) tell the fuller story?

## S3 — selectivity (stretch) (2026-07-17)

### The one-paragraph story

The last unmeasured workspace property is the converse of all the others: not
"does the workspace deliver when needed" but "**what runs fine without it**."
The paper's strongest form is the ablation experiment — at every position, find
the ten most active lens directions and surgically delete the residual stream's
component along them (a *projection removal*: subtract exactly the part of the
activation vector pointing along those directions, leave everything
perpendicular untouched), then watch which behaviors survive. The prediction:
flexible reasoning (two-hop chains) should die while routine prediction
(continuing ordinary text) should coast. At our scales it does — with margins:
all three subjects clear all three pre-committed gate legs. Two-hop
retention collapses under J-ablation (0.5B to 1/28 at *every* tier; 1.5B down
the paper's graded curve 21 → 13 → 5 of 41; 3B sharper still, 27 → 17 → 3 of 43) while matched-count
random directions barely dent it, and ordinary WikiText prediction under the
same heavy ablation stays CI-cleanly above the two-hop wreckage at every scale.
The two tiny targeted experiments add the reading-side texture: the passage's
language enters the lens almost only when the task needs it (7/8 and 8/8
passages explicit vs 0/8 and 2/8 automatic at 0.5B/1.5B), and line-length
counts surface most when directly asked for, least under the automatic
linewrap task — presence on demand, the paper's own selectivity signature.

### What was learned

29. **A pre-committed runtime gate can catch what tests cannot.** The 16
    analytic invariants all passed — on constructed data. The first *real*
    smoke run tripped the D6-style read-back immediately: genuine top-k lens
    direction sets are brutally ill-conditioned (near-duplicate tokens, Qwen's
    untrained reserved vocab slots), and the textbook least-squares projection
    silently exploded, then LAPACK's iterative SVD refused to converge. The
    shipped operator is modified Gram-Schmidt with re-orthogonalization —
    orthogonalize each direction against the accepted basis (twice, the
    classical fix for floating-point cancellation), drop directions with
    nothing new to add. Real activations are the adversarial test suite.
30. **Silent corruption exists one abstraction below your math.**
    `tensor.to("cpu", torch.float64)` in a single call garbles values coming
    off MPS — the cast happens device-side where float64 isn't supported, no
    exception raised, ~unit-scale errors. Move first, cast second
    (`.cpu().to(float64)`). Without the read-back gate this would have shipped
    plausible-looking garbage as a stage result; it's now in project memory as
    a standing hazard.
31. **"Selective" is a comparison, not an absolute.** The paper's ablated
    Claude keeps most ordinary prediction intact; our ablated subjects lose
    78% / 63% / 57% (0.5B / 1.5B / 3B) of next-token predictions under
    the heavy tier. What survives is the *contrast*: at every scale the
    flexible task is hit CI-cleanly harder than the automatic one (leg ii),
    and targeted directions hit CI-cleanly harder than random ones (leg iii).
    A small model's workspace band is a third of its whole depth — deleting
    ten directions per layer there is major surgery, and the honest claim is
    relative selectivity, not surgical precision.
32. **Presence itself can be the small-scale signature.** At Claude scale the
    language label sits in the lens under *every* task and only its causal
    role separates flexible from automatic (Fig 20b/c). Our subjects show the
    dissociation one level earlier: the label mostly isn't *in* the lens until
    the task demands it. Same selectivity story, expressed in loading rather
    than causation — consistent with M0's finding that these workspaces are
    weakly readable at rest.

### Recall questions (answers in this repo's docs)

22. The ablation operator went through three implementations (least-squares,
    SVD, modified Gram-Schmidt). What property of real top-k lens direction
    sets broke the first two, and which pre-committed piece of S3 machinery
    caught the failure before any result was recorded?
23. Gate leg (iii) compares J-lens ablation to a random-direction control at
    the medium tier. If that leg had *failed* — random hurting two-hop as much
    as targeted removal — what would the honest headline have been, and why
    does leg (i) alone not establish selectivity?
24. The wikitext top-1 match under heavy ablation is only .22–.37 across
    subjects, far from the paper's "mostly intact." Why does the would-gate
    still hold, and what exactly is the claim S3 makes (and refuses to make)
    about small-model selectivity?

## S4 — naming vs avoiding (stretch) (2026-07-17)

### The one-paragraph story

The paper's Figure-69 appendix experiment is the sharpest within-task version
of selectivity: *not* saying a word should take workspace machinery that
saying it doesn't (the inclusion/exclusion paradigm — resisting a primed
response as the mark of deliberate processing). A clue implies a concept
("croissants, the Louvre, the famous iron tower"), and you either name it or
name-something-else; deleting the concept's single lens direction early in
the band should break only the avoiding. At our scales the experiment
returned a clean split verdict. The **late**-band copy of the concept is a
hard output switch on every subject: remove that one direction and the model
cannot say the word at all (naming 0/n, concept probability ≈ .000). But the
**early**-band suppression machinery — the paper's actual claim — is absent:
the models that can do the exclusion task at all (mostly just 1.5B, 22/60
gated items) keep avoiding the concept perfectly with its early copy deleted
(0/22 failures). And upstream of all of it sits the project's most human
finding: small models largely *can't do the task* — 0.5B and 3B blurt the
forbidden concept unablated on 17 and 13 of 60 items. The frozen gate said
NOT shown, and NOT shown is the verdict.

### What was learned

33. **When you must construct items, the gate is the experiment.** S4 is the
    project's first invented item set (the reference ships nothing), made
    honest the same way the paper's own constructed items were: a pre-declared
    competence gate filters to items each subject can demonstrably do. The
    gate then delivered the headline before any ablation ran — the exclusion
    *task itself* is mostly beyond these models, and where it isn't (1.5B,
    again the M1/S1 subject) the workspace story gets its real test cell.
34. **A dissociation's two halves can scale apart.** Figure 69 is one claim
    with two prongs: late = intention-to-say (ablate → can't say it), early =
    suppression copy (ablate → can't NOT say it). At 0.5B–3B the late prong
    reproduces as an exact zero on every subject, while the early prong never
    clears any CI. Small models have the output switch; they lack — or don't
    need — the deliberate-suppression copy. A reproduction that treats "the
    dissociation" as one atomic finding would have missed that the halves
    separate.
35. **A null leg is only credible because it was pre-declared.** Leg 2
    ("naming is spared") *holds* everywhere — a claim of no-effect that
    would be unfalsifiable hand-waving if invented after the data, but was
    frozen in D30 before any run, with its own CI condition. The lineage
    lesson: gates should encode the *shape* of the predicted result,
    including the legs where the prediction is "nothing happens."

### Recall questions (answers in this repo's docs)

25. S4's items were constructed, not shipped — the project's first. What two
    frozen mechanisms made a constructed set honest, and which one produced
    the stage's headline before a single ablation ran?
26. Describe the split verdict: which half of Figure 69 reproduced at every
    scale, what exactly happens to naming and concept mass under late-third
    ablation, and why is that half still only "texture" rather than a gated
    claim in our records?
27. At 1.5B, 22 items passed the gate and early ablation produced 0/22
    avoidance failures. Why does that number, on its own, not distinguish
    "small models lack the early suppression copy" from "small models avoid
    by a different mechanism" — and which auxiliary readout (recorded in the
    results JSON) gives the sub-threshold hint that something early still
    moves?

### S4b addendum (2026-07-17) — the control cell that changed the sentence

36. **A follow-up control can promote texture to claim — and reveal a scale
    story on the way.** S4's late-band off-switch was uncontrolled texture;
    S4b added the matched same-category control at the middle and late tiers
    (D31, one decision, one afternoon cell). Result: on the powered subject
    (1.5B) the switch is **concept-specific, CI-clean** (primed 0/22 vs
    control 16/22, +.727 [+.471, +.868]) — and the specificity *emerges with
    scale*: 0.5B's late tier breaks under any single-direction removal
    (control survives 1/5), 3B's is perfectly clean (8/8, control mass ≈
    clean). Bonus discipline: the full re-run reproduced every shared cell
    bit-for-bit — deterministic greedy readouts make "did anything drift?" a
    free, exact check rather than a worry.

### Recall question (S4b)

28. Before S4b, why could "removing the concept's late direction silences the
    word" not be distinguished from "any late-tier removal breaks output" —
    and after S4b, at which scale is each of those two descriptions actually
    the correct one?
