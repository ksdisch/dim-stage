"""m2_two_hop.py — M2: swap the unspoken bridge, watch the final answer (frozen D9–D11).

The question, in plain terms: a **two-hop prompt** ("The language spoken in the
country where the Amazon River ends is ") forces the model to compute a bridge
entity it never says (*Brazil*) before answering (*Portuguese*). We swap the
bridge's J-lens coordinates for a replacement (*Brazil → Mexico*) at every band
layer and every prompt position, let the forward pass continue, and check
whether the model's top prediction becomes the redirected answer (*Spanish*).
The paper's anchor on these exact 90 shipped prompts: 60% top-1 flips
(Sonnet 4.5); 54–70% across Claude tiers on the 50-prompt variant.

Frozen conventions (D9–D11 + inherited M1 conventions, all pre-declared):

- **Raw-text prompts** (`Fact: …` completions — M0's lens-eval treatment, no
  chat template); readout = the final prompt token.
- **Baseline** (D9): the unswapped greedy next token; a trial's baseline
  passes iff it is among the `answer`'s single-token forms ({`w`, `␣w`} of the
  shipped string, strict — no synonym table; the README's "greedy next-token
  == answer" read verbatim). **The primary cell = baseline-passing items**;
  baseline accuracy and the unconditioned all-items rate are reported
  alongside. n < 20 in the primary cell is pre-declared UNDERPOWERED.
- **Success** (D10): the swapped run's greedy token is among `swap_answer`'s
  single-token forms — **top-1, the anchor's grading**. Wilson 95% CI on the
  primary cell; the frozen would-gate wording is LB ≥ 0.5, and **descriptive
  framing applies** (triple readability NULL — no property claims).
  Reported descriptively: `swap_answer`'s rank (top-5/top-10), the
  **displaced** rate (the original answer is no longer the greedy token),
  and the per-category breakdown.
- **Arm 2** (D10): every swap repeated with raw unembedding rows (**J = I**);
  Newcombe 95% CI on the (J − identity) difference — the standing
  falsification arm.
- **Answer-swap comparison** (D11, the paper's smuggling confound): every
  trial repeated swapping the **answer tokens** (`answer → swap_answer`)
  instead of the intermediates, both arms, same band — if intermediate swaps
  only work by smuggling in answer components, the two should look alike.
  Descriptive only; single band (no depth sweep — M2-BRIEF deviations row 4).
- An item lacking a single-token form for **any** of `intermediate`,
  `swap_to`, `answer`, `swap_answer` is dropped and counted (the M0
  pre-filter, applied to all four fields). Swap tokens are the bare forms;
  grading is min/membership over both forms (M1's frozen conventions).
- The D6 runtime read-back runs on every swap application; INVALID (exit 2)
  fires on any wrong-arm input before grading. `--dry-run` validates and
  stops.

Run:  uv run python -u m2_two_hop.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct \
          --lens lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import transformers

from fitter import SubjectModel
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import (
    checked_swap_edits,
    output_logits,
    swap_vectors,
    token_forms,
    word_rank,
)
from readability import proportional_band
from stats import newcombe_diff, wilson

EXPERIMENT_PATH = "refs/jacobian-lens/data/experiments/probe-swap.json"
EXPECTED_ITEMS = 90  # oracle-drift guard, shape as extracted 2026-07-16
FIELDS = ("intermediate", "swap_to", "answer", "swap_answer")
GATE_WILSON_FLOOR = 0.5  # frozen would-gate wording; descriptive mode applies
MIN_N = 20
DESCRIPTIVE_KS = (5, 10)
SWAPS = (  # (swap name, source field, target field)
    ("intermediate", "intermediate", "swap_to"),
    ("answer", "answer", "swap_answer"),  # D11 comparison arm
)
ARMS = (("jlens", True), ("identity", False))


def load_items(path: str = EXPERIMENT_PATH) -> list[dict]:
    """The 90 shipped two-hop items, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    items = json.load(open(path))["items"]
    if len(items) != EXPECTED_ITEMS or any(
        not all(field in item for field in (*FIELDS, "prompt", "category", "name"))
        for item in items
    ):
        raise ValueError(
            f"probe-swap.json drifted from the extracted {EXPECTED_ITEMS} items "
            "with intermediate/swap_to/answer/swap_answer fields — re-extract "
            "before running"
        )
    return items


def plan_item(item: dict, tokenizer) -> tuple[dict[str, list[int]], list[str]]:
    """Single-token forms for all four graded/swapped fields. Returns
    (forms by field, missing fields) — any missing field drops the item."""
    forms = {field: token_forms(item[field], tokenizer) for field in FIELDS}
    missing = [field for field in FIELDS if not forms[field]]
    return forms, missing


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
    try:
        load_items()
    except (FileNotFoundError, ValueError) as exc:
        fail_invalid(str(exc))
    return band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lens", required=True, help="fitted lens artifact (*.pt)")
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and stop")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cpu"),
        help="cpu keeps a smoke run off the GPU while something else owns it",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="grade only the first N items — SMOKE ONLY, never a result",
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
    print(
        f"validated: {args.model_id} n_layers={subject.n_layers}, lens n_prompts="
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]}, {EXPECTED_ITEMS} items OK"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    items = load_items()
    if args.limit is not None:
        items = items[: args.limit]

    records, dropped = [], []
    for item in items:
        start = time.perf_counter()
        forms, missing = plan_item(item, subject.tokenizer)
        if missing:
            dropped.append({"name": item["name"], "missing": missing})
            continue
        input_ids = subject.encode(item["prompt"], max_length=128)
        base_logits = output_logits(subject, input_ids)
        greedy = int(base_logits.argmax())
        baseline_correct = greedy in forms["answer"]

        record = {
            "name": item["name"],
            "category": item["category"],
            "baseline_correct": baseline_correct,
            "baseline_greedy": subject.tokenizer.decode([greedy]),
            "swaps": {},
        }
        for swap_name, src, dst in SWAPS:
            u_s = unembed_rows[forms[src][0]]  # bare forms — M1 convention
            u_t = unembed_rows[forms[dst][0]]
            for arm, use_jacobian in ARMS:
                vectors = swap_vectors(
                    jacobians, band, u_s, u_t, use_jacobian=use_jacobian
                )
                logits = output_logits(subject, input_ids, checked_swap_edits(vectors))
                swapped_greedy = int(logits.argmax())
                record["swaps"][f"{swap_name}_{arm}"] = {
                    "success": swapped_greedy in forms["swap_answer"],
                    "displaced": swapped_greedy not in forms["answer"],
                    "swap_answer_rank": word_rank(logits, forms["swap_answer"]),
                    "greedy": subject.tokenizer.decode([swapped_greedy]),
                }
        records.append(record)
        marks = " ".join(
            f"{k}={'HIT' if v['success'] else 'miss'}"
            for k, v in record["swaps"].items()
            if k.startswith("intermediate")
        )
        print(
            f"{item['name']}: baseline={'OK' if baseline_correct else 'WRONG'}"
            f" ({record['baseline_greedy']!r})  {marks}"
            f"  ({time.perf_counter() - start:.1f}s)"
        )

    def cell(swap_arm: str, subset: list[dict]) -> dict:
        n = len(subset)
        k = sum(r["swaps"][swap_arm]["success"] for r in subset)
        lb, ub = wilson(k, n) if n else (0.0, 1.0)
        return {
            "n": n,
            "successes": k,
            "rate": k / n if n else None,
            "wilson_95": [lb, ub],
            "displaced": sum(r["swaps"][swap_arm]["displaced"] for r in subset),
            "top_k": {
                str(kk): sum(r["swaps"][swap_arm]["swap_answer_rank"] <= kk for r in subset)
                for kk in DESCRIPTIVE_KS
            },
        }

    primary = [r for r in records if r["baseline_correct"]]
    n_base, n_all = len(primary), len(records)
    pooled = {
        "baseline_accuracy": {
            "n": n_all,
            "correct": n_base,
            "wilson_95": list(wilson(n_base, n_all)) if n_all else [0.0, 1.0],
        },
        "primary": {sa: cell(sa, primary) for sa in [f"{s}_{a}" for s, _, _ in SWAPS for a, _ in ARMS]},
        "unconditioned": {sa: cell(sa, records) for sa in [f"{s}_{a}" for s, _, _ in SWAPS for a, _ in ARMS]},
        "underpowered": n_base < MIN_N,
    }
    k_j = pooled["primary"]["intermediate_jlens"]["successes"]
    k_i = pooled["primary"]["intermediate_identity"]["successes"]
    diff, dlo, dhi = (
        newcombe_diff(k_i, n_base, k_j, n_base) if n_base else (0.0, -1.0, 1.0)
    )
    pooled["newcombe_intermediate_j_minus_identity"] = [diff, dlo, dhi]
    per_category = {}
    for r in primary:
        c = per_category.setdefault(r["category"], {"n": 0, "hits": 0})
        c["n"] += 1
        c["hits"] += r["swaps"]["intermediate_jlens"]["success"]

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "grading": "top-1 flip (anchor's grading, D10)",
            "primary_cell": "baseline-correct items (D9)",
            "wilson_floor_frozen_wording": GATE_WILSON_FLOOR,
            "min_n": MIN_N,
        },
        "dropped_single_token_prefilter": dropped,
        "items": records,
        "per_category_primary_jlens": per_category,
        "pooled": pooled,
    }

    lb, ub = pooled["primary"]["intermediate_jlens"]["wilson_95"]
    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    print(
        f"\nDESCRIPTIVE VERDICT: {smoke}baseline {n_base}/{n_all} correct | "
        f"intermediate-swap flips (primary): J-lens {k_j}/{n_base} "
        f"Wilson[{lb:.3f},{ub:.3f}]{' UNDERPOWERED' if pooled['underpowered'] else ''} | "
        f"identity {k_i}/{n_base} | J−identity diff={diff:+.3f} CI[{dlo:+.3f},{dhi:+.3f}] | "
        f"answer-swap (D11): J-lens "
        f"{pooled['primary']['answer_jlens']['successes']}/{n_base}, identity "
        f"{pooled['primary']['answer_identity']['successes']}/{n_base} | "
        f"no property claim (triple readability NULL; descriptive mode)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/two-hop-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
