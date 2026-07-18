# dim-stage — Presenter Pack

*Companion to `dim-stage-paper.md`. Purpose: defend the paper claim-by-claim in front
of a mentor. Every number traces to a repo file via the provenance table.*

## The 60-second story

Anthropic published a paper showing that language models keep a "global workspace" —
a band of middle layers whose contents, read through a tool called the Jacobian lens,
the model can verbally report, reason with, and steer on command. They showed it at
27B parameters and up, and explicitly left open whether it exists at small scale. I
rebuilt their lens from scratch, proved my implementation computes the exact same
matrices as their reference code — bit for bit — and measured their claims on three
small Qwen models on my own hardware, with every pass/fail bar frozen in code before
any experiment ran. The headline is a pre-registered null: at the bar I committed to
in advance, the workspace is not readable at 0.5B, 1.5B, or 3B — I escalated to the
third scale specifically because the first two came back null, and it was null too.
But the band isn't inert. Deleting its top lens directions selectively kills
multi-step reasoning while ordinary prediction survives — the one property that
passed its full pre-committed gate at every scale. And at 1.5B specifically, a
concept steered into that band becomes something the model *reports*, along a clean
dose–response curve with an exactly-zero control, an effect that needs the Jacobian
transport and lives in the middle five layers of the band. So: the paper's readable
workspace does not appear below 3B, and I can show you exactly what does.

## Results at a glance

| Stage | Question | Would-gate verdict | The number to remember |
|---|---|---|---|
| M0 | Is the workspace readable? | **NULL ×3** (kill-risk fired) | 0/6 distributions at Wilson LB ≥ .5, at all three scales |
| M1 report | Does the report follow a workspace swap? | far below anchor | .175 / .124 / .105 vs the paper's .88 |
| M1 introspection | Is a steered-in thought reported? | dose–response at 1.5B only | 0 → 30/101 with α; control exactly 0/101 |
| M2 | Does the answer follow a bridge swap? | far below anchor | .286 / .073 / .116 vs .60; at 3B J−I = +.116 [+.011, +.245] |
| M3 | Does instruction modulate the workspace? | **"does not modulate" ×3** | but focus−suppress CI-clean at 1.5B/3B; baseline 0/46 ×3 |
| S1 | Is the 1.5B curve real workspace machinery? | hardened on all 3 fronts | J−I +.178 [+.067, +.286] at α=8; L16–20 alone ≈ full band |
| S2 | Is one swap broadcast to many circuits? | **"does not route" ×3** | 17/16/18 of 180 at α=1 (all CI-clean over J=I); α=2 kills it |
| S3 | Does removing the workspace spare automatic tasks? | **selectivity-consistent ×3** (only full pass) | two-hop dies (5/41 heavy at 1.5B), random control survives (33/41) |
| S4 | Does avoiding a word need early workspace? | **NOT shown ×3** | 0/22 avoidance failures under early ablation at 1.5B |
| S4b | Is the late-band off-switch concept-specific? | yes at 1.5B (CI-clean) | primed 0/22 vs control 16/22, +.727 [+.471, +.868] |

## Provenance table (claim → number → source)

| Claim | Number | Source file |
|---|---|---|
| Readability null, every scale | 0/6 distributions pass; `subject_reads: false` | `results/readability-qwen2.5-{0.5b,1.5b,3b}-instruct.json` |
| Abstract content is a hard zero | association 0/99, poetry 0/98 (1/98 at 1.5B) ×3 scales | same files |
| Typo J-reversal deepens with scale | +.427 → −.271 → −.323 (all CI-stated) | same files; table in `docs/M0-BRIEF.md` |
| Independent build = reference | rel-Frobenius 0.000e+00; 3220/3220 readout | `docs/M0-BRIEF.md` (AGREE gate result) |
| Verbal report far below anchor | 17/97, 11/89, 9/86 top-5 | `results/verbal-report-qwen2.5-*.json` |
| Dose–response at 1.5B | 0/7/16/23/26/30 of 101 across α; control 0/101 | `results/introspection-qwen2.5-*.json` |
| Two-hop: 3B J-transport advantage | 5/43 vs 0/43; +.116 [+.011, +.245] | `results/two-hop-qwen2.5-3b-instruct.json` |
| Modulation gate + structure | gate false; focus 2/6/9 of 110; baseline 0/46 UB .077 | `results/directed-modulation-qwen2.5-*.json` |
| Modulation grows with scale | 3B−0.5B +.064 [+.004, +.131] | `results/derived-contrasts.json` (recomputed via `stats.py`); original record `docs/DECISIONS.md` (M3 outcomes) |
| 3B reads modulation better without J | 19/110 logit vs 9/110 J; −.091 [−.181, −.002] | `results/directed-modulation-qwen2.5-3b-instruct.json` |
| S1 transport specificity + saturation | J−I +.178 [+.067, +.286] at α=8; 30/30/29/31 plateau | `results/s1-introspection-qwen2.5-1.5b-instruct.json` |
| S1 mid-band localization | L16–20: 29/101 vs full band 31/101; full−mid +.020 [−.105, +.144] | same file; CIs recomputed in `results/derived-contrasts.json`, original record `docs/S1-BRIEF.md` |
| S2 routing signal + cliff | α=1: 17/16/18 of 180, J−I all CI-clean; α=2: 1/0/1 | `results/s2-generalization-qwen2.5-*.json` |
| S2 α=2 failure mode | target answer median rank 151,844.5 of 151,936 (1.5B) | `results/derived-contrasts.json` (median over the per-trial ranks in the S2 JSON); original records `docs/ROADMAP.md` / `docs/LEARNING.md` (S2), erratum in `docs/S2-BRIEF.md` |
| S3 full gate pass | legs (i) +.964/+.878/+.930, (ii) +.187/+.244/+.358, (iii) +.536/+.488/+.395 | `results/s3-selectivity-qwen2.5-*.json` |
| S3 honest qualifier | heavy ablation still changes 78%/63%/57% of wikitext predictions | same files (match .223/.366/.428); `docs/S3-BRIEF.md` |
| S4 competence gate is the finding | gated 5/22/8 of 60; clean avoidance blurts 17/6/13 | `results/s4-avoidance-qwen2.5-*.json` |
| S4b concept-specific off-switch | 1.5B: 0/22 vs 16/22, +.727 [+.471, +.868] | `results/s4-avoidance-qwen2.5-1.5b-instruct.json` |
| 3B fit detour | rented RTX 4090, ~57 min, ~$0.83 | `docs/DECISIONS.md` (3B rescue) |

## Anticipated Q&A

**Q: Why did you construct your own items for S4 — doesn't that break "reproduce,
don't invent"?**
A: The reference ships no item set for that experiment; the paper's own n=63 items
were also author-constructed. We kept it honest three ways: concepts came only from
vocabularies earlier stages had already measured on these subjects, the set was
frozen in-repo before any run, and the paper's own competence-gate pattern filtered
items — the gate, not authorship, is what makes items honest. It's labeled as the
project's first constructed set everywhere it appears.

**Q: Why is a null a result and not a failure?**
A: The bar was frozen before any measurement, the readability question is the
paper's own stated open question, and the null survived a pre-registered escalation
to a third scale. "Not readable at Wilson LB ≥ .5 on Qwen2.5 0.5B–3B" is exactly as
informative as a pass would have been — it brackets where the phenomenon starts. A
null chosen after peeking would be a failure; a pre-committed null is an answer.

**Q: Why Wilson intervals, and what does "CI-clean" mean here?**
A: Wilson 95% intervals behave sensibly at small N and at rates near 0 or 1 — where
this project lives — and never leave [0,1]; the paper itself reports Wilson
intervals, so anchor comparisons are like-for-like. Differences between arms get
Newcombe 95% intervals (the Wilson-based method for a difference of proportions).
"CI-clean" means the interval excludes zero; anything else is reported as no-claim.

**Q: What can't the null rule out — the un-validatable residual?**
A: Two things, owned in the limitations. First, the band: we transplanted the
paper's 38–92% depth band proportionally and pre-registered it to kill forking
paths — but if a small model's workspace lives elsewhere, our lens was pointed at
the wrong floors and the null is about the band, not the model. Second, the fit
corpus: 100 WikiText prompts vs the paper's ~1000 web-text sequences (their own
saturation data says quality plateaus early, but it's still a deviation). Inside the
frozen conventions the null is airtight; the conventions themselves are the residual.

**Q: Why these models?**
A: The thesis is "below 27B is unexplored" — so subjects had to be small, local, and
instruction-tuned (the shipped stimuli are chat-format). Qwen2.5 0.5B and 1.5B fit
in fp32 on a 24 GB Apple-silicon laptop; 3B was added only when the pre-registered
trigger fired (both smaller scales null) and needed $0.83 of rented GPU for the one
fit the laptop physically couldn't do.

**Q: If the workspace isn't readable, why did anything downstream work at all?**
A: Readability and causal load-bearing dissociate. The lens can't recover workspace
content at rank ≤ 10, yet removing the band's top lens directions kills two-hop
reasoning while random damage doesn't (S3), and steering along transported
directions makes a concept reportable at 1.5B (M1/S1). Weakly readable ≠ unused —
that dissociation is the project's most interesting descriptive fact.

**Q: What would you do next?**
A: Roads not taken, in order of pull: single-layer resolution on S1's L16–20 result
(the sub-band design was deliberately coarse); the S2 α window mapped finer than
{1, 2} to find where routing dies; a re-derived band (the paper's diagnostics) as a
descriptive check on the transplant; and the two permanently-out-of-scope areas —
LLM-judge-graded tasks and Claude-only sections — stay out. Nothing is scheduled;
the wrap decision is the owner's.

## Vocabulary crib

- **Residual stream** — the running vector each transformer layer reads and adds to;
  one per token position. What the lens reads.
- **Logit / unembedding** — the final matrix turns a residual vector into one score
  (logit) per vocabulary token; highest logit = the model's next-token pick.
- **Logit lens** — decode a *middle* layer with the final layer's unembedding, no
  correction. Our standing `J = I` falsification arm.
- **Jacobian / `J_l`** — a matrix of sensitivities: how much each final-layer
  coordinate moves per nudge of a layer-`l` coordinate, averaged over a corpus.
- **Jacobian lens (J-lens)** — multiply a middle-layer vector by `J_l` (transport it
  into the final layer's basis), then unembed.
- **VJP / cotangent** — backprop's "direction of interest": put a 1 in one output
  coordinate, run backward, read off that coordinate's sensitivity to everything.
- **Workspace band** — the contiguous middle-layer range (38–92% of depth) the paper
  says holds workspace content; transplanted proportionally here.
- **Swap (patching in lens coordinates)** — exchange two concepts' coordinates
  inside an activation, leaving everything orthogonal to the pair untouched.
- **Steering** — add a scaled concept direction to the residual stream; α is the
  strength in mean-residual-norm units.
- **Ablation (projection removal)** — subtract exactly the component of the
  activation lying along chosen directions; everything perpendicular survives.
- **Teacher-forcing / prefill** — write the model's reply for it up to a point, then
  read the next-token distribution there. Deterministic; no sampling anywhere.
- **Greedy next token** — the single highest-logit token; our top-1 oracle.
- **pass@k / rank** — a target "hits" if it appears at lens rank ≤ k; rank is its
  position in the sorted logits.
- **MRR** — median reciprocal rank (1/rank, median over trials); the paper's
  introspection metric.
- **Wilson 95% interval** — the honest range a true rate sits in given n trials;
  gates use its lower bound.
- **Newcombe 95% interval** — the same idea for a *difference* of two rates;
  "CI-clean" = excludes zero.
- **UNDERPOWERED** — pre-declared tag for any cell with N < 20; no gated claim.
- **Would-gate** — a verdict whose wording was frozen pre-run but which reports
  descriptively (the pre-registered re-scope after the readability null).
- **Forking paths** — tuning analysis choices after seeing results; everything here
  was frozen before first official numbers to prevent it.
- **AGREE gate** — the cross-check that the independent fitter equals the reference
  implementation on identical inputs (it did, bitwise).
- **Degeneracy/collapse guard** — flags a run where one token wins ≥ half of all
  trials with a new fixed point — model breakage, not measurement.
- **Competence gate** — restrict analysis to items a subject can do unablated
  (paper's own pattern; greedy form here).
- **Kill-risk** — the pre-declared way the project could die: "no readable workspace
  at small scale." It fired; the pre-registered fallback (descriptive re-scope) ran.
