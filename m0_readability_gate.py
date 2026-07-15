"""M0 readability gate — is the workspace readable in this subject? (frozen D4)

Two arms, decided per distribution at k=10 over the six lens-eval distributions,
each intermediate one Bernoulli trial (hit iff min-over-fitted-layers rank at the
readout position ≤ k):

  Arm 1 — absolute readability (GATES): the subject READS iff J-lens pass@10 has
      Wilson 95% lower bound ≥ 0.5 on ≥ 3 of 6 distributions; otherwise NULL.
      A distribution with n < 20 trials is pre-declared UNDERPOWERED and cannot
      pass. A pre-committed NULL is a reportable result, not a failure.
  Arm 2 — J-advantage (reported per distribution, never gates the READS call):
      Newcombe 95% CI on (J-lens − logit-lens) hit-rate difference; the Jacobian
      correction "adds value" where the CI excludes 0. The logit lens is this
      same code with J = identity (`use_jacobian` off in the grading).

Also reported descriptively (never gating): k ∈ {1, 5, 25}, and the
band-restricted min-rank over the frozen D2 proportional band.

INVALID (exit 2) on any wrong-arm input — lens/model mismatch, drifted eval
data, or a band that disagrees with the frozen table — before any grading runs.
`--dry-run` performs exactly this validation and stops.

Run:  uv run python -u m0_readability_gate.py \
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

import readability
from fitter import SubjectModel
from stats import excludes_zero, newcombe_diff, wilson

GATE_K = 10
DESCRIPTIVE_KS = (1, 5, 25)
WILSON_FLOOR = 0.5
MIN_N = 20
NEED_DISTRIBUTIONS = 3
#: Frozen D2 bands (M0-BRIEF): proportional 38–92% of depth.
#: 36-layer entry (Qwen2.5-3B-Instruct) pre-registered 2026-07-15, before the
#: 3B fit or any 3B readout existed — the escalation trigger (both subjects
#: NULL) fired and Kyle chose to escalate; see DECISIONS.md.
FROZEN_BANDS = {
    24: list(range(9, 22)),
    28: list(range(11, 25)),
    36: list(range(14, 33)),
}


def fail_invalid(reason: str) -> None:
    print(f"VERDICT: INVALID — {reason}")
    raise SystemExit(2)


def validate(args, artifact: dict, subject: SubjectModel) -> list[int]:
    """Wrong-arm input checks. Returns the frozen band for this subject."""
    if artifact.get("model_id") != args.model_id:
        fail_invalid(
            f"lens artifact was fitted on {artifact.get('model_id')!r}, "
            f"not {args.model_id!r}"
        )
    if artifact.get("d_model") != subject.d_model:
        fail_invalid(
            f"lens d_model={artifact.get('d_model')} != subject d_model={subject.d_model}"
        )
    expected_layers = list(range(subject.n_layers - 1))
    if sorted(artifact["J"]) != expected_layers:
        fail_invalid(
            f"lens layers {sorted(artifact['J'])[:3]}..{sorted(artifact['J'])[-1]} "
            f"!= expected 0..{subject.n_layers - 2}"
        )
    band = readability.proportional_band(subject.n_layers)
    if band != FROZEN_BANDS.get(subject.n_layers):
        fail_invalid(
            f"proportional band {band} disagrees with frozen table "
            f"{FROZEN_BANDS.get(subject.n_layers)} for {subject.n_layers} layers"
        )
    for slug in readability.SLUGS:
        readability.load_distribution(slug)  # raises on drift / missing files
    return band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--lens", required=True, help="fitted lens artifact (*.pt)")
    parser.add_argument("--out", default=None, help="results JSON path")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and stop")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "mps", "cpu"),
        help="cpu keeps a smoke run off the GPU while a fit owns it",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="grade only the first N items per distribution — SMOKE ONLY, "
        "never a gate result",
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
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]}, six distributions OK"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no grading performed")
        raise SystemExit(0)

    jacobians = {l: J.to(device) for l, J in artifact["J"].items()}
    all_layers = sorted(jacobians)

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "gate": {
            "k": GATE_K, "wilson_floor": WILSON_FLOOR, "min_n": MIN_N,
            "need_distributions": NEED_DISTRIBUTIONS,
        },
        "distributions": {},
    }
    arm1_passes = 0
    for slug in readability.SLUGS:
        items = readability.load_distribution(slug)
        if args.limit is not None:
            items = items[: args.limit]
        start = time.perf_counter()
        graded, dropped = [], []
        for item in items:
            g, d = readability.grade_item(subject, jacobians, slug, item)
            graded.extend(g)
            dropped.extend(d)
        n = len(graded)

        cells: dict[str, dict] = {}
        for arm in ("jlens", "logitlens"):
            for layerset_name, layerset in (("all", all_layers), ("band", band)):
                hits = {
                    k: sum(1 for g in graded if g.min_rank(arm, layerset) <= k)
                    for k in (GATE_K, *DESCRIPTIVE_KS)
                }
                cells[f"{arm}_{layerset_name}"] = {"n": n, "hits": hits}

        k_j = cells["jlens_all"]["hits"][GATE_K]
        k_l = cells["logitlens_all"]["hits"][GATE_K]
        lb, ub = wilson(k_j, n)
        diff, dlo, dhi = newcombe_diff(k_l, n, k_j, n)
        underpowered = n < MIN_N
        arm1_pass = (not underpowered) and lb >= WILSON_FLOOR
        arm1_passes += arm1_pass
        j_advantage = excludes_zero(dlo, dhi) and diff > 0

        results["distributions"][slug] = {
            "n_trials": n,
            "dropped_single_token_prefilter": dropped,
            "cells": cells,
            "arm1": {
                "pass_at_10": k_j / n if n else None,
                "wilson_95": [lb, ub],
                "underpowered": underpowered,
                "reads": arm1_pass,
            },
            "arm2": {
                "logitlens_pass_at_10": k_l / n if n else None,
                "newcombe_diff": [diff, dlo, dhi],
                "j_advantage": j_advantage,
            },
        }
        print(
            f"{slug}: n={n} (dropped {len(dropped)})  "
            f"J-lens pass@10={k_j}/{n} Wilson[{lb:.3f},{ub:.3f}] "
            f"{'READS' if arm1_pass else ('UNDERPOWERED' if underpowered else 'below floor')}"
            f"  |  logit-lens {k_l}/{n}, J−logit diff={diff:+.3f} "
            f"CI[{dlo:+.3f},{dhi:+.3f}] {'J-ADVANTAGE' if j_advantage else 'no clear gap'}"
            f"  ({time.perf_counter() - start:.0f}s)"
        )

    verdict = "READS" if arm1_passes >= NEED_DISTRIBUTIONS else "NULL"
    results["verdict"] = {
        "arm1_distributions_passing": arm1_passes,
        "subject_reads": verdict == "READS",
    }
    smoke = f"SMOKE (limit={args.limit}) — not a gate result — " if args.limit else ""
    print(
        f"\nVERDICT: {smoke}{verdict} — {arm1_passes}/6 distributions with J-lens "
        f"pass@10 Wilson LB >= {WILSON_FLOOR} (gate needs >= {NEED_DISTRIBUTIONS}); "
        f"J-advantage reported per distribution above"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/readability-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
