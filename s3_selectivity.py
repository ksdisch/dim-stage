"""s3_selectivity.py — S3: what runs *without* the workspace (frozen D23–D26).

The question, in plain terms: every prior stage asked whether the workspace
delivers when a task needs it. Selectivity is the converse — the paper's claim
that the workspace is engaged only for *flexible* thinking (reporting a fact,
handing it to an arbitrary downstream operation) while *automatic* processing
(continuing text, tracking a line width) uses the same information without
routing it through the workspace. Two kinds of evidence, both frozen in (D23):

1. **Targeted reading contrasts** (the repo's two shipped item sets, verbatim):
   - `selectivity-language.json`: 8 passages (2 per fr/de/es/it), explicit
     (name an author of the passage's language) vs automatic (continue the
     passage). Track the language-label tokens in the lens over the question
     positions. Paper: presence comparable in every condition (Fig 20b).
   - `selectivity-linecount.json`: 11 passages wrapped at a fixed width;
     continue / bare-prefill / direct-count / first-letter conditions. Track
     two-digit and number-word tokens over all prompt positions. Paper:
     presence ~zero under linewrap, highest under first-letter (Fig 21b).
   Both cells are tiny by construction and pre-declared UNDERPOWERED (D26) —
   texture, not CI-gated claims.
2. **J-space ablation** (the new operator, D24): at every prompt position and
   band layer, select the k=10 most strongly activated lens tokens (by M0's
   lens-readout logits, the layer's own J_l), excluding tokens in the clean
   forward pass's per-position output top-10 (never ablate what the model
   intends to say — our per-position reading, owned), and zero the residual's
   projection onto the span of their J-lens directions (M1's raw-row direction
   convention). Tiers are start-anchored prefixes of the frozen band: light =
   first third, medium = first two-thirds, heavy = full band. Control: k
   random directions at the medium tier, frozen seed.

   Eval sets (D25): M2's two-hop items verbatim (flexible task — does the
   answer survive?; primary cell = baseline-correct items, our own M2-matched
   baselines) and fresh WikiText-103 records 101–200 under the D3 selection
   rule — provably disjoint from every fit corpus — scored as per-position
   top-1 agreement between the ablated and clean model (automatic task).

Would-gate wording (D26, frozen before any run; descriptive mode throughout —
triple readability NULL): **"selectivity-consistent"** iff (i) heavy ablation
drops two-hop accuracy on the primary cell CI-cleanly below its unablated 100%
(Newcombe excludes 0), (ii) wikitext top-1 match under heavy ablation stays
CI-cleanly above the two-hop retention rate, and (iii) the random control
disrupts two-hop CI-cleanly less than J-ablation at the medium tier. Failing
(iii) is the headline against: any-direction damage, not workspace
selectivity. D6-style runtime read-back (post-ablation lens coordinates on
every selected direction ~0) on every edit; INVALID (exit 2) on any wrong-arm
input; `--dry-run` validates + plans and stops; `--limit N` is smoke, never a
result.

Run:  uv run python -u s3_selectivity.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct \
          --lens lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap
import time
from collections import Counter

import torch
import transformers

from fitter import (
    SubjectModel,
    _record_residuals,
    load_wikitext_prompts,
    valid_position_mask,
)
from intervention import Edit, ablate, edit_residuals, jlens_vector
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import output_logits, token_forms, word_rank
from m2_two_hop import load_items as load_two_hop_items
from m2_two_hop import plan_item
from readability import proportional_band
from stats import excludes_zero, newcombe_diff, wilson

LANGUAGE_PATH = "refs/jacobian-lens/data/experiments/selectivity-language.json"
LINECOUNT_PATH = "refs/jacobian-lens/data/experiments/selectivity-linecount.json"
FIT_CORPUS_PATH = "wikitext-n100-prompts.json"

ABLATION_K = 10  # D24: verbatim paper k
CLEAN_EXCLUDE_K = 10  # D24: verbatim clean-top-10 output exclusion
LENS_HIT_K = 10  # D26: presence = target token in lens top-10, any band layer
TEXTURE_K = 5
RANDOM_SEED = 20260717  # D24: frozen seed for the random-direction control
READBACK_TOL = 1e-4  # relative |⟨v, h'⟩| / (‖v‖·‖h‖) ceiling, fp32 headroom
WIKITEXT_SKIP = 100  # D25: fit corpus = first 100 qualifying records
WIKITEXT_MAX_TOKENS = 128
EXPECTED_TWO_HOP_GRADABLE = 81  # measured M2 2026-07-16, shared Qwen tokenizer
TIERS = ("light", "medium", "heavy")
CONDITIONS = ("jlens_light", "jlens_medium", "jlens_heavy", "random_medium")
MIN_N = 20
COLLAPSE_SHARE = 0.5  # S1/S2 degeneracy convention

TENS_WORDS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}
TEENS_WORDS = (
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
UNITS_WORDS = (
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
)


def count_word(n: int) -> str:
    """10..99 spelled out ('forty-six') — the letter condition's ground truth."""
    if not 10 <= n <= 99:
        raise ValueError(f"count_word covers 10..99, got {n}")
    if n < 20:
        return TEENS_WORDS[n - 10]
    tens, units = divmod(n, 10)
    return TENS_WORDS[tens] + ("-" + UNITS_WORDS[units] if units else "")


def load_language(path: str = LANGUAGE_PATH) -> dict:
    """The shipped 8-passage language set, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    data = json.load(open(path))
    cats = sorted({p.get("category") for p in data.get("passages", [])})
    if (
        len(data.get("passages", [])) != 8
        or len(cats) != 4
        or any("{text}" not in data.get("task", {}).get(q, "")
               for q in ("explicit_q", "automatic_q"))
        or sorted(data.get("intermediates", {})) != cats
        or sorted(data.get("authors", {})) != cats
    ):
        raise ValueError(
            "selectivity-language.json drifted from the extracted 8 passages / "
            "4 categories with task + intermediates + authors — re-extract "
            "before running"
        )
    return data


def load_linecount(path: str = LINECOUNT_PATH) -> dict:
    """The shipped 11-passage line-count set, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    data = json.load(open(path))
    if (
        len(data.get("passages", [])) != 11
        or sorted(data.get("conditions", {})) != ["direct", "letter", "none"]
        or any(
            set(c) != {"question", "prefill"}
            for c in data.get("conditions", {}).values()
        )
        or not data.get("explicit_q")
        or any(
            not all(k in p for k in ("tag", "width", "text"))
            for p in data.get("passages", [])
        )
    ):
        raise ValueError(
            "selectivity-linecount.json drifted from the extracted 11 passages "
            "/ 3 prefill conditions + explicit_q — re-extract before running"
        )
    return data


def tier_bands(band: list[int]) -> dict[str, list[int]]:
    """D24 start-anchored tiers: light = first third of the frozen band,
    medium = first two-thirds, heavy = the full band."""
    n = len(band)
    return {
        "light": band[: max(1, n // 3)],
        "medium": band[: max(1, (2 * n) // 3)],
        "heavy": band,
    }


def target_ids(words: list[str], tokenizer) -> list[int]:
    """Union of single-token forms ({w, ␣w}, capitalized variants included)
    over a word list — the lens target sets for both targeted arms."""
    ids: set[int] = set()
    for word in words:
        for variant in {word, word.capitalize()}:
            ids.update(token_forms(variant, tokenizer))
    return sorted(ids)


def linecount_targets(tokenizer) -> list[int]:
    """README's target set: any two-digit token or English number-word token.
    On the Qwen tokenizer the two-digit half is empty — multi-digit numbers
    tokenize digit-by-digit — so the tracked set is number-words only (owned
    deviation; single digits are excluded as they would swamp the readout)."""
    words = [str(n) for n in range(10, 100)]
    words += list(TEENS_WORDS) + list(TENS_WORDS.values())
    return target_ids(words, tokenizer)


def first_token_rank(logits: torch.Tensor, text: str, tokenizer) -> int:
    """Rank of the first sub-token of `text`'s space form — answer texture for
    fields with no single-token form (Qwen splits '46' into '4','6')."""
    ids = tokenizer(" " + text, add_special_tokens=False).input_ids
    return word_rank(logits, [ids[0]])


def lens_hits(
    subject: SubjectModel,
    input_ids: torch.Tensor,
    band: list[int],
    jacobians: dict[int, torch.Tensor],
    targets: list[int],
) -> dict[str, list[bool]]:
    """Clean forward; per prompt position, whether any target token sits in the
    lens top-k at any band layer (M0's readout convention: unembed(J_l·h)).
    Returns per-position hit lists for k = LENS_HIT_K (primary) and TEXTURE_K."""
    target_tensor: torch.Tensor | None = None
    seq = input_ids.shape[1]
    hits = {str(k): [False] * seq for k in (LENS_HIT_K, TEXTURE_K)}
    with torch.no_grad(), _record_residuals(
        subject.layers, band, graph_root=None
    ) as res:
        subject.forward(input_ids)
        for layer in band:
            lensed = res[layer][0].float() @ jacobians[layer].T
            logits = subject.unembed(lensed)
            top = logits.topk(LENS_HIT_K, dim=-1).indices  # [seq, k]
            if target_tensor is None:
                target_tensor = torch.tensor(targets, device=top.device)
            member = torch.isin(top, target_tensor)  # [seq, k]
            for k in (LENS_HIT_K, TEXTURE_K):
                pos_hit = member[:, :k].any(dim=-1).cpu()
                key = str(k)
                hits[key] = [a or bool(b) for a, b in zip(hits[key], pos_hit)]
    return hits


def clean_pass(
    subject: SubjectModel, input_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One unmodified forward. Returns (per-position top-10 output ids
    [seq, 10] — the D24 exclusion set, per-position greedy ids [seq] — the
    wikitext match reference, final-position logits [vocab] — grading)."""
    final = subject.n_layers - 1
    with torch.no_grad(), _record_residuals(
        subject.layers, [final], graph_root=None
    ) as res:
        subject.forward(input_ids)
        logits = subject.unembed(res[final][0].float())  # [seq, vocab]
        top10 = logits.topk(CLEAN_EXCLUDE_K, dim=-1).indices
        greedy = logits.argmax(dim=-1)
        return top10, greedy.cpu(), logits[-1].float().cpu()


def ablation_edits(
    subject: SubjectModel,
    jacobians: dict[int, torch.Tensor],
    layers: list[int],
    unembed_rows: torch.Tensor,
    clean_top10: torch.Tensor,
    *,
    random_directions: torch.Generator | None = None,
) -> dict[int, Edit]:
    """Per-layer ablation edits (D24). J-lens mode selects, per position, the
    k tokens with the largest lens logits at that layer (clean output top-10
    masked out), builds their J-lens directions with the layer's own J_l, and
    projects the span out of the residual. Random mode substitutes k fresh
    Gaussian directions from the frozen-seed generator — no selection, no
    exclusion (they are not tokens). Every edit runs the D6-style read-back:
    the surviving projection onto each removed direction must be ~0, else
    INVALID."""
    edits: dict[int, Edit] = {}
    for layer in layers:

        def edit(h: torch.Tensor, layer=layer) -> torch.Tensor:
            hs = h[0]  # [seq, d_model]
            if random_directions is None:
                lensed = hs.float() @ jacobians[layer].T
                logits = subject.unembed(lensed)  # [seq, vocab]
                logits.scatter_(
                    -1, clean_top10.to(logits.device), float("-inf")
                )
                ids = logits.topk(ABLATION_K, dim=-1).indices  # [seq, k]
                u = unembed_rows[ids]  # [seq, k, d_model]
                directions = u.float() @ jacobians[layer]  # rows J_lᵀu (M1 conv.)
            else:
                directions = torch.randn(
                    hs.shape[0], ABLATION_K, hs.shape[1],
                    generator=random_directions,
                ).to(device=hs.device, dtype=torch.float32)
            out = ablate(hs.float(), directions).to(hs.dtype)
            leftover = torch.einsum("skd,sd->sk", directions, out.float()).abs()
            scale = directions.norm(dim=-1) * hs.float().norm(dim=-1, keepdim=True)
            worst = float((leftover / scale.clamp_min(1e-30)).max())
            if worst > READBACK_TOL:
                fail_invalid(
                    f"ablation read-back failed at layer {layer}: surviving "
                    f"projection {worst:.2e} > {READBACK_TOL:.0e} "
                    "(D6-style runtime self-check)"
                )
            return out.unsqueeze(0)

        edits[layer] = edit
    return edits


def ablated_position_greedy(
    subject: SubjectModel, input_ids: torch.Tensor, edits: dict[int, Edit]
) -> torch.Tensor:
    """Per-position greedy ids [seq] under the given edits (wikitext arm)."""
    final = subject.n_layers - 1
    with torch.no_grad(), edit_residuals(subject.layers, edits):
        with _record_residuals(subject.layers, [final], graph_root=None) as res:
            subject.forward(input_ids)
        return subject.unembed(res[final][0].float()).argmax(dim=-1).cpu()


def degeneracy(greedy_ids: list[int], tokenizer) -> dict:
    """S1/S2 guard for one condition cell: most common greedy token's share."""
    if not greedy_ids:
        return {"attractor_token": None, "share": 0.0, "collapsed": False}
    token, count = Counter(greedy_ids).most_common(1)[0]
    share = count / len(greedy_ids)
    return {
        "attractor_token": tokenizer.decode([token]),
        "share": share,
        "collapsed": share >= COLLAPSE_SHARE,
    }


def validate(args, artifact: dict, subject: SubjectModel) -> list[int]:
    """Wrong-arm input checks (M0 gate pattern). Returns the frozen band."""
    if artifact.get("model_id") != args.model_id:
        fail_invalid(
            f"lens artifact was fitted on {artifact.get('model_id')!r}, "
            f"not {args.model_id!r}"
        )
    if artifact.get("d_model") != subject.d_model:
        fail_invalid(
            f"lens d_model={artifact.get('d_model')} != subject d_model={subject.d_model}"
        )
    if sorted(artifact["J"]) != list(range(subject.n_layers - 1)):
        fail_invalid(f"lens layers != expected 0..{subject.n_layers - 2}")
    band = proportional_band(subject.n_layers)
    if band != FROZEN_BANDS.get(subject.n_layers):
        fail_invalid(
            f"proportional band {band} disagrees with frozen table "
            f"{FROZEN_BANDS.get(subject.n_layers)} for {subject.n_layers} layers"
        )
    for loader in (load_language, load_linecount, load_two_hop_items):
        try:
            loader()
        except (FileNotFoundError, ValueError) as exc:
            fail_invalid(str(exc))
    if not os.path.exists(FIT_CORPUS_PATH):
        fail_invalid(
            f"{FIT_CORPUS_PATH} missing — needed to prove the fresh wikitext "
            "slice is disjoint from the fit corpus (D25)"
        )
    return band


def question_positions(
    template: str, text: str, encode
) -> tuple[list[int], int]:
    """Token positions of everything after the passage in a formatted prompt
    (the language arm's tracked span). Returns (positions, seq_len)."""
    pre = template.split("{text}")[0]
    prefix_len = encode(pre + text).shape[1]
    seq_len = encode(template.format(text=text)).shape[1]
    return list(range(min(prefix_len, seq_len), seq_len)), seq_len


def rate_cell(k: int, n: int) -> dict:
    lb, ub = wilson(k, n)
    return {
        "hits": k, "n": n, "rate": k / n if n else None,
        "wilson_95": [lb, ub], "underpowered": n < MIN_N,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lens", required=True, help="fitted lens artifact (*.pt)")
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument("--dry-run", action="store_true", help="validate + plan, then stop")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cpu"),
        help="cpu keeps a smoke run off the GPU while something else owns it",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="limit two-hop items and wikitext prompts — SMOKE ONLY, never a result",
    )
    parser.add_argument(
        "--wikitext-n", type=int, default=100,
        help="fresh wikitext records for the automatic arm (plan detail; a "
        "reduction below 100 is an owned deviation row)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"loading {args.model_id} (fp32, {device})")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32
    ).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model_id)
    subject = SubjectModel(hf, tok)

    try:
        artifact = torch.load(args.lens, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        fail_invalid(f"lens artifact {args.lens} not found")
    band = validate(args, artifact, subject)
    tiers = tier_bands(band)

    language = load_language()
    linecount = load_linecount()
    two_hop_items = load_two_hop_items()
    planned = [(item, *plan_item(item, tok)) for item in two_hop_items]
    gradable = [(item, forms) for item, forms, missing in planned if not missing]
    dropped = [
        {"name": item["name"], "missing": missing}
        for item, _, missing in planned
        if missing
    ]
    if len(gradable) != EXPECTED_TWO_HOP_GRADABLE:
        fail_invalid(
            f"two-hop plan: {len(gradable)} gradable items != frozen "
            f"{EXPECTED_TWO_HOP_GRADABLE} (measured M2 2026-07-16) — the "
            "tokenizer or data file is not the one the freeze was measured on"
        )

    lc_targets = linecount_targets(tok)
    lang_targets = {
        cat: target_ids(words, tok)
        for cat, words in language["intermediates"].items()
    }
    if not lc_targets or any(not v for v in lang_targets.values()):
        fail_invalid("empty lens target set — tokenizer mismatch with the item sets")

    print(
        f"validated: {args.model_id} n_layers={subject.n_layers}, lens n_prompts="
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]} | tiers "
        + ", ".join(f"{t} L{tiers[t][0]}–L{tiers[t][-1]} ({len(tiers[t])})" for t in TIERS)
        + f" | two-hop {len(gradable)} gradable / {len(dropped)} dropped | "
        f"targets: linecount {len(lc_targets)} ids, language "
        + ", ".join(f"{c} {len(v)}" for c, v in sorted(lang_targets.items()))
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    rng = torch.Generator().manual_seed(RANDOM_SEED)

    # --- targeted arm 1: language (8 passages × 2 conditions) -------------------
    lang_records = []
    for passage in language["passages"]:
        cat = passage["category"]
        for cond in ("explicit", "automatic"):
            template = language["task"][f"{cond}_q"]
            prompt = template.format(text=passage["text"])
            input_ids = subject.encode(prompt, max_length=512)
            positions, seq_len = question_positions(
                template, passage["text"], lambda t: subject.encode(t, max_length=512)
            )
            hits = lens_hits(subject, input_ids, band, jacobians, lang_targets[cat])
            _, _, final_logits = clean_pass(subject, input_ids)
            author_forms = [
                f
                for a in language["authors"][cat]
                for f in (token_forms(a, tok) + token_forms(a.capitalize(), tok))
            ]
            lang_records.append(
                {
                    "key": passage["key"],
                    "category": cat,
                    "condition": cond,
                    "question_positions": len(positions),
                    "hit_positions": sum(hits[str(LENS_HIT_K)][p] for p in positions),
                    "hit_positions_top5": sum(hits[str(TEXTURE_K)][p] for p in positions),
                    "any_hit": any(hits[str(LENS_HIT_K)][p] for p in positions),
                    "greedy": tok.decode([int(final_logits.argmax())]),
                    "best_author_rank": (
                        word_rank(final_logits, author_forms) if author_forms else None
                    ),
                }
            )
        print(f"language {passage['key']} done")

    # --- targeted arm 2: linecount (11 passages × 4 conditions) -----------------
    lc_records = []
    for passage in linecount["passages"]:
        wrapped = textwrap.fill(passage["text"], passage["width"])
        truth = len(wrapped.split("\n")[0])
        letter = count_word(truth)[0].upper()
        prompts = {
            "continue": linecount["explicit_q"] + "\n\n" + wrapped,
            **{
                name: (
                    (cond["question"] + "\n\n" if cond["question"] else "")
                    + wrapped
                    + cond["prefill"]
                )
                for name, cond in linecount["conditions"].items()
            },
        }
        for name, prompt in prompts.items():
            input_ids = subject.encode(prompt, max_length=512)
            hits = lens_hits(subject, input_ids, band, jacobians, lc_targets)
            _, _, final_logits = clean_pass(subject, input_ids)
            texture_forms = {
                "direct": token_forms(str(truth), tok),
                "none": token_forms(str(truth), tok),
                "letter": token_forms(letter, tok),
            }.get(name)
            lc_records.append(
                {
                    "tag": passage["tag"],
                    "condition": name,
                    "truth": truth,
                    "positions": input_ids.shape[1],
                    "hit_positions": sum(hits[str(LENS_HIT_K)]),
                    "hit_positions_top5": sum(hits[str(TEXTURE_K)]),
                    "any_hit": any(hits[str(LENS_HIT_K)]),
                    "greedy": tok.decode([int(final_logits.argmax())]),
                    "truth_rank": (
                        word_rank(final_logits, texture_forms)
                        if texture_forms
                        else None
                    ),
                    "truth_first_token_rank": (
                        first_token_rank(final_logits, str(truth), tok)
                        if name in ("direct", "none")
                        else None
                    ),
                }
            )
        print(f"linecount {passage['tag']} (first line {truth} chars) done")

    # --- ablation arm 1: two-hop (flexible task) ---------------------------------
    items = gradable[: args.limit] if args.limit else gradable
    records = []
    for i, (item, forms) in enumerate(items):
        start = time.perf_counter()
        input_ids = subject.encode(item["prompt"], max_length=128)
        clean_top10, _, final_logits = clean_pass(subject, input_ids)
        greedy = int(final_logits.argmax())
        record = {
            "name": item["name"],
            "category": item["category"],
            "baseline_correct": greedy in forms["answer"],
            "baseline_greedy": tok.decode([greedy]),
            "conditions": {},
        }
        for condition in CONDITIONS:
            tier = condition.split("_")[1]
            edits = ablation_edits(
                subject, jacobians, tiers[tier], unembed_rows, clean_top10,
                random_directions=rng if condition.startswith("random") else None,
            )
            logits = output_logits(subject, input_ids, edits)
            g = int(logits.argmax())
            record["conditions"][condition] = {
                "correct": g in forms["answer"],
                "answer_rank": word_rank(logits, forms["answer"]),
                "greedy": tok.decode([g]),
                "greedy_id": g,
            }
        records.append(record)
        marks = " ".join(
            f"{c.split('_')[0][0]}{c.split('_')[1][0]}="
            + ("OK" if record["conditions"][c]["correct"] else "x")
            for c in CONDITIONS
        )
        print(
            f"[{i + 1}/{len(items)}] {item['name']}: baseline="
            f"{'OK' if record['baseline_correct'] else 'WRONG'} {marks} "
            f"({time.perf_counter() - start:.1f}s)"
        )

    primary = [r for r in records if r["baseline_correct"]]
    n_primary = len(primary)
    two_hop = {
        "baseline_accuracy": rate_cell(n_primary, len(records)),
        "primary": {
            c: rate_cell(
                sum(r["conditions"][c]["correct"] for r in primary), n_primary
            )
            for c in CONDITIONS
        },
        "unconditioned": {
            c: rate_cell(
                sum(r["conditions"][c]["correct"] for r in records), len(records)
            )
            for c in CONDITIONS
        },
    }
    guard = {
        c: degeneracy([r["conditions"][c]["greedy_id"] for r in records], tok)
        for c in CONDITIONS
    }
    for r in records:
        for c in CONDITIONS:
            del r["conditions"][c]["greedy_id"]

    # --- ablation arm 2: wikitext (automatic task) -------------------------------
    n_wiki = args.wikitext_n
    if args.limit:
        n_wiki = min(n_wiki, args.limit)
    all_prompts = load_wikitext_prompts(WIKITEXT_SKIP + n_wiki)
    fit_corpus = json.load(open(FIT_CORPUS_PATH))
    if all_prompts[: len(fit_corpus)] != fit_corpus:
        fail_invalid(
            "the first 100 streamed wikitext records no longer match the fit "
            "corpus JSON — the D25 disjointness proof fails; do not proceed"
        )
    fresh = all_prompts[WIKITEXT_SKIP:]
    wiki_counts = {c: [0, 0] for c in CONDITIONS}  # condition -> [matches, n]
    per_prompt_rates: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    for i, prompt in enumerate(fresh):
        start = time.perf_counter()
        input_ids = subject.encode(prompt, max_length=WIKITEXT_MAX_TOKENS)
        mask = valid_position_mask(input_ids.shape[1])
        clean_top10, clean_greedy, _ = clean_pass(subject, input_ids)
        for condition in CONDITIONS:
            tier = condition.split("_")[1]
            edits = ablation_edits(
                subject, jacobians, tiers[tier], unembed_rows, clean_top10,
                random_directions=rng if condition.startswith("random") else None,
            )
            greedy = ablated_position_greedy(subject, input_ids, edits)
            match = (greedy[mask] == clean_greedy[mask])
            wiki_counts[condition][0] += int(match.sum())
            wiki_counts[condition][1] += int(mask.sum())
            per_prompt_rates[condition].append(
                float(match.float().mean()) if int(mask.sum()) else 0.0
            )
        print(
            f"[wiki {i + 1}/{len(fresh)}] {int(mask.sum())} positions "
            + " ".join(
                f"{c.split('_')[0][0]}{c.split('_')[1][0]}="
                f"{per_prompt_rates[c][-1]:.2f}"
                for c in CONDITIONS
            )
            + f" ({time.perf_counter() - start:.1f}s)"
        )
    wikitext = {
        c: {
            **rate_cell(*wiki_counts[c]),
            "per_prompt_mean": (
                sum(per_prompt_rates[c]) / len(per_prompt_rates[c])
                if per_prompt_rates[c]
                else None
            ),
            "per_prompt_min": min(per_prompt_rates[c], default=None),
        }
        for c in CONDITIONS
    }

    # --- the frozen would-gate (D26) --------------------------------------------
    k_heavy = two_hop["primary"]["jlens_heavy"]["hits"]
    k_medium = two_hop["primary"]["jlens_medium"]["hits"]
    k_random = two_hop["primary"]["random_medium"]["hits"]
    k_match, n_match = wiki_counts["jlens_heavy"]
    gate_i = newcombe_diff(k_heavy, n_primary, n_primary, n_primary)
    gate_ii = newcombe_diff(k_heavy, n_primary, k_match, n_match)
    gate_iii = newcombe_diff(k_medium, n_primary, k_random, n_primary)
    legs = {
        "i_heavy_drops_two_hop": {
            "newcombe_clean_minus_heavy": list(gate_i),
            "holds": excludes_zero(gate_i[1], gate_i[2]) and gate_i[0] > 0,
        },
        "ii_wikitext_survives_above_retention": {
            "newcombe_match_minus_retention": list(gate_ii),
            "holds": excludes_zero(gate_ii[1], gate_ii[2]) and gate_ii[0] > 0,
        },
        "iii_random_disrupts_less_at_medium": {
            "newcombe_random_minus_jlens_medium": list(gate_iii),
            "holds": excludes_zero(gate_iii[1], gate_iii[2]) and gate_iii[0] > 0,
        },
    }
    consistent = all(leg["holds"] for leg in legs.values())

    # --- targeted-arm cells (UNDERPOWERED texture, D26) ---------------------------
    def pooled_positions(recs, cond_field, cond_value):
        subset = [r for r in recs if r[cond_field] == cond_value]
        k = sum(r["hit_positions"] for r in subset)
        n = sum(r.get("question_positions", r.get("positions", 0)) for r in subset)
        return subset, k, n

    lang_cells, lc_cells = {}, {}
    for cond in ("explicit", "automatic"):
        subset, k, n = pooled_positions(lang_records, "condition", cond)
        lang_cells[cond] = {
            "passages": len(subset),
            "any_hit_passages": sum(r["any_hit"] for r in subset),
            "pooled_positions": rate_cell(k, n),
            "mean_position_rate": (
                sum(
                    r["hit_positions"] / r["question_positions"]
                    for r in subset
                    if r["question_positions"]
                )
                / len(subset)
                if subset
                else None
            ),
            "underpowered": True,  # D26: 8 passages, pre-declared
        }
    ek, en = (
        lang_cells["explicit"]["pooled_positions"]["hits"],
        lang_cells["explicit"]["pooled_positions"]["n"],
    )
    ak, an = (
        lang_cells["automatic"]["pooled_positions"]["hits"],
        lang_cells["automatic"]["pooled_positions"]["n"],
    )
    lang_contrast = newcombe_diff(ak, an, ek, en)
    for cond in ("continue", "none", "direct", "letter"):
        subset, k, n = pooled_positions(lc_records, "condition", cond)
        lc_cells[cond] = {
            "passages": len(subset),
            "any_hit_passages": sum(r["any_hit"] for r in subset),
            "pooled_positions": rate_cell(k, n),
            "underpowered": True,  # D26: 11 passages, pre-declared
        }

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "tiers": {t: tiers[t] for t in TIERS},
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "ablation_k": ABLATION_K,
            "clean_exclude_k": CLEAN_EXCLUDE_K,
            "lens_hit_k": LENS_HIT_K,
            "random_seed": RANDOM_SEED,
            "readback_tol": READBACK_TOL,
            "wikitext": {
                "skip": WIKITEXT_SKIP,
                "n": n_wiki,
                "max_tokens": WIKITEXT_MAX_TOKENS,
                "scored_positions": "valid_position_mask (fit convention)",
            },
            "min_n": MIN_N,
            "collapse_share": COLLAPSE_SHARE,
            "note": "targeted cells pre-declared UNDERPOWERED (D26); pooled "
            "position CIs are texture — positions within a passage are not "
            "independent trials",
            "linecount_target_deviation": "Qwen tokenizes multi-digit numbers "
            "digit-by-digit, so the README's two-digit half of the target set "
            "is empty on this tokenizer; tracked set = number-words only "
            f"({len(lc_targets)} ids). Owned deviation.",
        },
        "two_hop": {
            **two_hop,
            "dropped_single_token_prefilter": dropped,
            "items": records,
        },
        "wikitext": wikitext,
        "would_gate": {
            "legs": legs,
            "selectivity_consistent": consistent,
        },
        "degeneracy_guard": guard,
        "language": {"cells": lang_cells,
                     "newcombe_explicit_minus_automatic_pooled": list(lang_contrast),
                     "records": lang_records},
        "linecount": {"cells": lc_cells, "records": lc_records},
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    heavy = two_hop["primary"]["jlens_heavy"]
    medium = two_hop["primary"]["jlens_medium"]
    rand = two_hop["primary"]["random_medium"]
    wik = wikitext["jlens_heavy"]
    collapsed = [c for c in CONDITIONS if guard[c]["collapsed"]]
    print(
        f"\nDESCRIPTIVE VERDICT: {smoke}two-hop baseline {n_primary}/{len(records)}"
        f"{' UNDERPOWERED' if n_primary < MIN_N else ''} | retention under "
        f"J-ablation light/med/heavy: "
        f"{two_hop['primary']['jlens_light']['hits']}/{n_primary}, "
        f"{medium['hits']}/{n_primary}, {heavy['hits']}/{n_primary} | random@med "
        f"{rand['hits']}/{n_primary} | wikitext top-1 match heavy "
        f"{wik['hits']}/{wik['n']} Wilson[{wik['wilson_95'][0]:.3f},"
        f"{wik['wilson_95'][1]:.3f}] | gate legs "
        f"i={'HOLDS' if legs['i_heavy_drops_two_hop']['holds'] else 'fails'} "
        f"ii={'HOLDS' if legs['ii_wikitext_survives_above_retention']['holds'] else 'fails'} "
        f"iii={'HOLDS' if legs['iii_random_disrupts_less_at_medium']['holds'] else 'fails'}"
        f" → {'selectivity-consistent' if consistent else 'NOT shown'} "
        f"(descriptive) | language explicit−automatic pooled "
        f"{lang_contrast[0]:+.3f} [{lang_contrast[1]:+.3f},{lang_contrast[2]:+.3f}]"
        f" (UNDERPOWERED texture) | linecount hit-rates "
        + " ".join(
            f"{c}={lc_cells[c]['pooled_positions']['rate']:.3f}"
            for c in ("continue", "none", "direct", "letter")
        )
        + f" | degeneracy: {', '.join(collapsed) if collapsed else 'none'} | "
        f"no property claim (triple readability NULL; descriptive mode)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/s3-selectivity-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
