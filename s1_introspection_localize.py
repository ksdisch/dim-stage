"""s1_introspection_localize.py — S1 stretch: harden + localize the 1.5B introspection dose–response.

M1 found the project's one live signal: at 1.5B a *steered-in* thought becomes
reportable, report rate rising 0 → 30/101 with strength against an exactly-0/101
control. S1 (Bundle 2, D15) deepens that flagship finding without any new model,
fit, band, or intervention primitive — it reuses the frozen M0/M1 lens and the M1
steering operator, and adds three things:

  A. **Saturation** (D17): extend the α grid past M1's top (still-rising) 8 with
     {12, 16, 24}, so the paper's Figure-7 rise-then-saturate shape is visible.
  B. **Falsification arm** (D16): rerun every steer with `J = I` — steer along the
     unit-normalized *raw unembedding row* u_t (the logit-lens direction, no
     Jacobian transport) at the same layers/positions/scale. Newcombe 95% CI on
     (J-lens − J=I) at each α. This is the standing Arm-2 contrast every other
     milestone carried; the introspection curve is the only headline finding that
     never got one. The paper claims the J-space component is what drives the
     report (§specificity control).
  C. **Layer localization** (D18, 1.5B only): steer three contiguous sub-bands of
     L11–L24 separately — early L11–15, middle L16–20, late L21–24 — at the
     best-reporting α (both arms), to see *where* in the band the reportability
     lives. Anchored to the paper's mid-layer "middle block" workspace claim.

**Degeneracy guard** (D17): at each α we record the *attractor share* — the largest
fraction of concepts whose greedy top-1 at the open quote is a single shared token.
Under genuine per-concept reporting no token wins a majority (each concept reports
its own word). A **collapse** is flagged when attractor share ≥ 0.5 AND the
attractor token differs from the control (α=0) token — i.e. a *new* fixed point
dominates, not the model's natural default. This separates real saturation from the
model breaking under over-steering, deterministically. Localization's best-α is
chosen among non-collapsed strengths.

Everything else is frozen from M1: steering direction (unit-normalized J-lens vector
scaled by the layer's mean residual norm over the D3 corpus × α), applied at every
band layer over the user-question turn; readout = rank-1 of the concept's surface at
the open quote; `default` prefill (S1 drops the `word` prefill — M1 showed it tracks
`default` within a few hits; owned deviation, keeps the sweep legible).

**Mode: descriptive** (triple readability NULL — the pre-registered re-scope holds).
S1 characterizes the flagship finding; it does not reclassify it as reproduction.

INVALID (exit 2) fires before any measurement on: lens/model mismatch, a band off the
frozen table, drifted experiment data, a missing/short prompts file, or a tokenizer
without a chat template. `--dry-run` performs exactly that validation and stops.

Run:  uv run python -u s1_introspection_localize.py \
          --model-id Qwen/Qwen2.5-1.5B-Instruct \
          --lens lenses/qwen2.5-1.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter

import torch
import transformers

from fitter import SubjectModel
from intervention import positional, steer, steer_edits
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_introspection import (
    build_prompt,
    load_introspection,
    mean_residual_norms,
    median_reciprocal_rank,
)
from m1_verbal_report import output_logits, token_forms, word_rank
from readability import proportional_band
from stats import newcombe_diff, wilson

PREFILL = "default"  # S1 primary; `word` dropped (M1: tracks within a few hits) — owned
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)  # D8 grid + D17 extension
ARMS = ("jlens", "identity")  # identity = J=I falsification arm (D16)
COLLAPSE_SHARE = 0.5  # D17 degeneracy guard threshold
LOCALIZATION_N_LAYERS = 28  # D18: sub-band localization runs on 1.5B only


def identity_steer_edits(band, u_t, mean_residual_norms, alpha, *, positions):
    """The J = I falsification arm (D16): the M1 steering operator with the
    Jacobian transport removed — direction = unit-normalized *raw* unembedding row
    u_t (the logit-lens direction), scaled by the layer's mean residual norm × α,
    at the same layers and question-turn positions. The only difference from
    `steer_edits` is v_t = u_t instead of v_t = J_lᵀu_t; reuses the same `steer`
    and `positional` primitives, so nothing about application changes."""
    norm = u_t.norm()
    if norm == 0:
        raise ValueError("u_t is the zero vector — no steering direction")
    edits = {}
    for layer in band:
        direction = u_t / norm * mean_residual_norms[layer]

        def edit(h, direction=direction):
            return steer(h, direction.to(device=h.device, dtype=h.dtype), alpha)

        edits[layer] = positional(edit, positions)
    return edits


def arm_edits(arm, jacobians, at, u_t, norms, alpha, *, positions):
    """Dispatch to the J-lens or J=I steering edits for one arm."""
    if arm == "jlens":
        return steer_edits(jacobians, at, u_t, norms, alpha, positions=positions)
    return identity_steer_edits(at, u_t, norms, alpha, positions=positions)


def summarize(ranks, top1_tokens, control_token):
    """Per-(arm, α) cell: report rate (rank-1) with Wilson CI, MRR, and the
    degeneracy diagnostics (attractor share/token + collapse flag)."""
    n = len(ranks)
    k = sum(r == 1 for r in ranks)
    lb, ub = wilson(k, n) if n else (0.0, 1.0)
    attr_token, attr_count = Counter(top1_tokens).most_common(1)[0]
    share = attr_count / n if n else 0.0
    collapsed = share >= COLLAPSE_SHARE and attr_token != control_token
    return {
        "n": n,
        "report_hits": k,
        "report_rate": k / n if n else None,
        "wilson_95": [lb, ub],
        "mrr": median_reciprocal_rank(ranks) if n else None,
        "attractor_token": int(attr_token),
        "attractor_share": share,
        "collapsed": collapsed,
    }


def validate(args, artifact, subject):
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
    if not getattr(subject.tokenizer, "chat_template", None):
        fail_invalid("tokenizer has no chat template — the intro prompt needs one")
    try:
        load_introspection()
    except (FileNotFoundError, ValueError) as exc:
        fail_invalid(str(exc))
    if not os.path.exists(args.prompts_file):
        fail_invalid(
            f"{args.prompts_file} missing — the frozen D5 steering scale needs "
            "the D3 prompts dump (see fitter.py --prompts-file)"
        )
    prompts = json.load(open(args.prompts_file))
    if len(prompts) != 100:
        fail_invalid(f"{args.prompts_file} holds {len(prompts)} prompts, not 100")
    return band


def sub_bands(band):
    """D18 thirds of the 1.5B band L11–L24: early L11–15, middle L16–20, late L21–24."""
    return {
        f"L{band[0]}-{band[4]}": band[0:5],
        f"L{band[5]}-{band[9]}": band[5:10],
        f"L{band[10]}-{band[-1]}": band[10:],
    }


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
        help="steer only the first N concepts — SMOKE ONLY, never a result",
    )
    parser.add_argument(
        "--prompts-file", default="wikitext-n100-prompts.json",
        help="the frozen D3 corpus dump (steering-scale convention)",
    )
    args = parser.parse_args()

    device = (
        ("mps" if torch.backends.mps.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
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
    localize = subject.n_layers == LOCALIZATION_N_LAYERS  # D18: 1.5B only
    print(
        f"validated: {args.model_id} n_layers={subject.n_layers}, band "
        f"L{band[0]}–L{band[-1]}, arms={ARMS}, α={ALPHAS}, "
        f"localization={'ON (1.5B)' if localize else 'off (D18: 1.5B only)'}"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no measurement performed")
        raise SystemExit(0)

    prompts = json.load(open(args.prompts_file))
    t0 = time.perf_counter()
    norms = mean_residual_norms(subject, prompts, band)
    print(
        f"mean residual norms over the D3 corpus: L{band[0]}={norms[band[0]]:.2f} "
        f".. L{band[-1]}={norms[band[-1]]:.2f}  ({time.perf_counter() - t0:.0f}s)"
    )

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    data = load_introspection()
    concepts = data["concepts"][: args.limit] if args.limit else data["concepts"]
    messages = data["intro_prompt"][:3]

    input_ids, positions = build_prompt(subject, messages, data["prefills"][PREFILL])
    control_logits = output_logits(subject, input_ids)  # α=0, concept-independent
    control_token = int(control_logits.argmax())

    # --- A + B: all-band sweep, both arms, extended α grid ---
    per_concept, dropped = [], []
    for i, concept in enumerate(concepts):
        surface = concept["surface"]
        forms = token_forms(surface, subject.tokenizer)
        if not forms:
            dropped.append(surface)
            continue
        u_t = unembed_rows[forms[0]]
        control_rank = word_rank(control_logits, forms)
        row = {"name": concept["name"], "surface": surface, "token": forms[0], "sweep": {}}
        t0 = time.perf_counter()
        for arm in ARMS:
            row["sweep"][arm] = {"0.0": {"rank": control_rank, "top1": control_token}}
            for alpha in ALPHAS[1:]:
                edits = arm_edits(arm, jacobians, band, u_t, norms, alpha, positions=positions)
                logits = output_logits(subject, input_ids, edits)
                row["sweep"][arm][str(alpha)] = {
                    "rank": word_rank(logits, forms), "top1": int(logits.argmax())
                }
        per_concept.append(row)
        jl = [row["sweep"]["jlens"][str(a)]["rank"] for a in ALPHAS]
        print(f"{i + 1}/{len(concepts)} {surface}: jlens ranks {jl}  "
              f"({time.perf_counter() - t0:.1f}s)")

    n = len(per_concept)
    sweep = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for alpha in ALPHAS:
            a = str(alpha)
            ranks = [c["sweep"][arm][a]["rank"] for c in per_concept]
            top1 = [c["sweep"][arm][a]["top1"] for c in per_concept]
            sweep[arm][a] = summarize(ranks, top1, control_token)
    # Arm B: J-lens − J=I difference per α (Newcombe)
    sweep_diff = {}
    for alpha in ALPHAS:
        a = str(alpha)
        d, lo, hi = newcombe_diff(
            sweep["identity"][a]["report_hits"], n,
            sweep["jlens"][a]["report_hits"], n,
        )
        sweep_diff[a] = {"jlens_minus_identity": d, "newcombe_95": [lo, hi]}

    # --- C: layer localization (1.5B only) at the best non-collapsed α ---
    localization = None
    if localize and n:
        eligible = [
            a for a in ALPHAS[1:]
            if not sweep["jlens"][str(a)]["collapsed"]
        ] or list(ALPHAS[1:])
        best_alpha = max(eligible, key=lambda a: sweep["jlens"][str(a)]["report_hits"])
        bands = sub_bands(band)
        print(f"\nlocalization: best non-collapsed α={best_alpha}; "
              f"sub-bands {list(bands)} × {ARMS}")
        localization = {"best_alpha": best_alpha, "sub_bands": {}}
        for label, layers in bands.items():
            cell = {}
            for arm in ARMS:
                ranks, top1 = [], []
                for c in per_concept:
                    u_t = unembed_rows[c["token"]]
                    edits = arm_edits(arm, jacobians, layers, u_t, norms, best_alpha,
                                      positions=positions)
                    logits = output_logits(subject, input_ids, edits)
                    ranks.append(word_rank(logits, token_forms(c["surface"], subject.tokenizer)))
                    top1.append(int(logits.argmax()))
                cell[arm] = summarize(ranks, top1, control_token)
            localization["sub_bands"][label] = cell
            print(f"  {label}: jlens {cell['jlens']['report_hits']}/{n}, "
                  f"identity {cell['identity']['report_hits']}/{n}")
        # full band at best α, for the comparison baseline
        localization["full_band"] = {
            arm: sweep[arm][str(best_alpha)] for arm in ARMS
        }

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "alphas": list(ALPHAS),
            "prefill": PREFILL,
            "arms": list(ARMS),
            "steering_scale": "mean L2 residual norm, D3 corpus, fit valid positions",
            "collapse_share_threshold": COLLAPSE_SHARE,
            "control_token": control_token,
        },
        "mean_residual_norms": {str(l): norms[l] for l in band},
        "dropped_single_token_prefilter": dropped,
        "n": n,
        "sweep": sweep,
        "sweep_diff": sweep_diff,
        "localization": localization,
        "concepts": per_concept,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    print(f"\nDESCRIPTIVE SUMMARY: {smoke}n={n} concepts (dropped {len(dropped)})")
    for arm in ARMS:
        print(f"  arm={arm}:")
        for alpha in ALPHAS:
            cell = sweep[arm][str(alpha)]
            flag = "  <collapse>" if cell["collapsed"] else ""
            print(
                f"    α={alpha:<5}: report {cell['report_hits']}/{n} "
                f"Wilson[{cell['wilson_95'][0]:.3f},{cell['wilson_95'][1]:.3f}] "
                f"MRR={cell['mrr']:.3f}  attractor={cell['attractor_share']:.2f}{flag}"
            )
    print("  J-lens − J=I (Newcombe 95%):")
    for alpha in ALPHAS:
        d = sweep_diff[str(alpha)]["jlens_minus_identity"]
        lo, hi = sweep_diff[str(alpha)]["newcombe_95"]
        print(f"    α={alpha:<5}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print(
        "no property claim (triple readability NULL; descriptive mode; "
        "introspection never gates — D8/D15)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/s1-introspection-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
