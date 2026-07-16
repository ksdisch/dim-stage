"""m3_directed_modulation.py — M3: can the model steer its own workspace on command? (frozen D12–D14)

The setup, in plain terms: the model is instructed about a target X ("Think
about X while you write." / "X is irrelevant — ignore it." / a bare mention)
and then **teacher-forced** through an unrelated carrier sentence. The J-lens
reads the workspace band over the carrier's token positions — text that gives
no reason for X to appear. A trial is a **hit iff a tracked token is the
lens's top-1 readout at any (layer, position)** — Figure 10's grading,
verbatim. The paper's text-stated anchors: the no-instruction baseline is
approximately zero, and the ordering focus > suppress > baseline (with the
human "white bear" effect: suppression instructions still activate the
concept above baseline).

This is a **reading** milestone (the modulation is the instruction; no
activation edits), so the J = I logit-lens falsification arm returns: every
trial is graded under both the J-lens and the plain logit lens from the same
forward pass.

Frozen conventions (D12–D14 + owned details, all pre-declared):

- **Families** (D12): the two shipped ones — category-instance (22 targets,
  every `member` tracked) and math-expression (24 targets, the `answer`
  tracked) — plus the constructed no-instruction baseline. Line-width is out
  (stimuli unshipped; M3-BRIEF deviations row 3).
- **Trial grid** (D13): every phrasing × every target; carrier assigned
  deterministically as `carriers[(i_phrasing + j_target) mod 20]`; baseline =
  every target × `carriers[j_target mod 20]`. No RNG anywhere.
- **Prompt frame** (D13, owned — the reference ships none): user turn = the
  filled phrasing, a space, then `Copy this sentence exactly: "<carrier>"`;
  baseline turn = the copy frame alone. Chat template with generation prompt
  (M1 precedent); the assistant reply is teacher-forced to the carrier
  verbatim; **readout span = the carrier's token positions**.
- **Tracked tokens**: each tracked string's single-token {`w`, `␣w`} forms
  (M0's convention); a target with no tracked form drops and is counted.
- **Conditions**: `group_kind` collapses the four shipped phrasing groups to
  focus / control (mention) / suppress (dismissal + negated-think); the
  dismissal vs negated-think split is reported separately (the appendix
  Figure 65 finding the shipped groups preserve).
- **Verdict** (D14, wording frozen; descriptive mode per the triple
  readability NULL): cells per family × condition × arm with Wilson 95% CIs;
  the would-gate reads "modulates" iff the focus − suppress Newcombe 95% CI
  excludes 0 (positive) on BOTH families AND the pooled no-instruction
  baseline is clean (Wilson UB ≤ 0.10 — the pre-declared reading of
  "approximately zero"; pooled across families, n = 46, because a zero-hit
  per-family cell of n = 22–24 cannot get its Wilson UB under 0.10 —
  per-family baselines are reported alongside). Also reported:
  suppress − baseline (white bear), focus − control (instruction vs mere
  mention), and J − logit per condition.

INVALID (exit 2) fires before any measurement on wrong-arm inputs (lens/model
mismatch, off-table band, drifted experiment data, missing chat template).
`--dry-run` validates and stops.

Run:  uv run python -u m3_directed_modulation.py \
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

from fitter import SubjectModel, _record_residuals
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import token_forms
from readability import proportional_band
from stats import newcombe_diff, wilson

EXPERIMENT_PATH = "refs/jacobian-lens/data/experiments/directed-modulation.json"
#: Oracle-drift guard: shape as extracted 2026-07-16.
EXPECTED = {"phrasings": 24, "carrier_sentences": 20, "math_problems": 24, "topic_categories": 22}
FRAME = 'Copy this sentence exactly: "{carrier}"'
CONDITIONS = ("focus", "control", "suppress", "baseline")
ARMS = ("jlens", "logitlens")
BASELINE_UB = 0.10  # pre-declared reading of the paper's "approximately zero"
FAMILIES = ("category", "math")


def load_experiment(path: str = EXPERIMENT_PATH) -> dict:
    """The shipped protocol data, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    data = json.load(open(path))
    if (
        any(len(data[k]) != n for k, n in EXPECTED.items())
        or set(data["group_kind"].values()) != {"focus", "control", "suppress"}
        or {p["group"] for p in data["phrasings"]} != set(data["group_kind"])
    ):
        raise ValueError(
            "directed-modulation.json drifted from the extracted shape "
            "(24 phrasings / 20 carriers / 24 math / 22 categories, four "
            "groups mapping to focus/control/suppress) — re-extract"
        )
    return data


def build_targets(data: dict) -> list[dict]:
    """The 46 targets: {x} fill + the tracked strings, per family."""
    targets = [
        {"family": "math", "name": m["expr"], "x": m["expr"], "tracked": [m["answer"]]}
        for m in data["math_problems"]
    ]
    targets += [
        {"family": "category", "name": t["name"], "x": t["name"], "tracked": t["members"]}
        for t in data["topic_categories"]
    ]
    return targets


def build_grid(n_phrasings: int, n_targets: int, n_carriers: int):
    """The frozen deterministic pairing (D13): trials (i, j, (i+j) mod C) and
    baselines (None, j, j mod C). No RNG anywhere."""
    trials = [
        (i, j, (i + j) % n_carriers)
        for i in range(n_phrasings)
        for j in range(n_targets)
    ]
    baselines = [(None, j, j % n_carriers) for j in range(n_targets)]
    return trials, baselines


def build_prompt(subject: SubjectModel, user_text: str, carrier: str):
    """Chat-templated user turn + the carrier teacher-forced as the assistant
    reply. Returns (input_ids, span) where span covers exactly the carrier's
    token positions — the readout span."""
    tok = subject.tokenizer
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        add_generation_prompt=True,
        return_dict=False,
    )
    carrier_ids = tok(carrier, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix + carrier_ids]).to(subject._input_device)
    span = list(range(len(prefix), len(prefix) + len(carrier_ids)))
    return input_ids, span


def trial_hits(
    subject: SubjectModel,
    jacobians: dict[int, torch.Tensor],
    band: list[int],
    input_ids: torch.Tensor,
    span: list[int],
    tracked: set[int],
) -> dict[str, bool]:
    """One forward; both arms. Hit iff a tracked token is the top-1 readout at
    any (band layer, span position) — Figure 10's grading, verbatim."""
    hits = {"jlens": False, "logitlens": False}
    with torch.no_grad():
        with _record_residuals(subject.layers, band, graph_root=None) as res:
            subject.forward(input_ids)
        for layer in band:
            h = res[layer][0, span].float()
            for arm in ARMS:
                if hits[arm]:
                    continue
                z = h @ jacobians[layer].T.to(h.device) if arm == "jlens" else h
                top1 = subject.unembed(z).argmax(dim=-1)
                if any(int(t) in tracked for t in top1):
                    hits[arm] = True
            if all(hits.values()):
                break
    return hits


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
    if not getattr(subject.tokenizer, "chat_template", None):
        fail_invalid("tokenizer has no chat template — the D13 prompt frame needs one")
    try:
        load_experiment()
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
        help="first N targets per family — SMOKE ONLY, never a result",
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
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]}, experiment data OK"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    data = load_experiment()
    phrasings = data["phrasings"]
    group_kind = data["group_kind"]
    carriers = data["carrier_sentences"]

    targets, dropped = [], []
    for target in build_targets(data):
        forms: set[int] = set()
        for word in target["tracked"]:
            forms.update(token_forms(word, subject.tokenizer))
        if not forms:
            dropped.append(target["name"])
            continue
        targets.append({**target, "forms": forms})
    if args.limit is not None:
        kept = []
        for family in FAMILIES:
            kept += [t for t in targets if t["family"] == family][: args.limit]
        targets = kept

    trials, baselines = build_grid(len(phrasings), len(targets), len(carriers))
    records = []
    start_all = time.perf_counter()
    for n_done, (i, j, c) in enumerate([*trials, *baselines]):
        target = targets[j]
        if i is None:
            condition, group = "baseline", "baseline"
            user_text = FRAME.format(carrier=carriers[c])
        else:
            phrasing = phrasings[i]
            condition, group = group_kind[phrasing["group"]], phrasing["group"]
            user_text = (
                phrasing["text"].replace("{x}", target["x"])
                + " "
                + FRAME.format(carrier=carriers[c])
            )
        input_ids, span = build_prompt(subject, user_text, carriers[c])
        hits = trial_hits(subject, jacobians, band, input_ids, span, target["forms"])
        records.append(
            {
                "family": target["family"],
                "target": target["name"],
                "condition": condition,
                "group": group,
                "phrasing": None if i is None else phrasings[i]["name"],
                "carrier": c,
                "hit_jlens": hits["jlens"],
                "hit_logitlens": hits["logitlens"],
            }
        )
        if (n_done + 1) % 100 == 0:
            print(
                f"{n_done + 1}/{len(trials) + len(baselines)} trials "
                f"({time.perf_counter() - start_all:.0f}s)"
            )

    def cell(family: str | None, condition: str, arm: str) -> dict:
        subset = [
            r
            for r in records
            if r["condition"] == condition and (family is None or r["family"] == family)
        ]
        n = len(subset)
        k = sum(r[f"hit_{arm}"] for r in subset)
        lb, ub = wilson(k, n) if n else (0.0, 1.0)
        return {"n": n, "hits": k, "rate": k / n if n else None, "wilson_95": [lb, ub]}

    cells = {
        family: {cond: {arm: cell(family, cond, arm) for arm in ARMS} for cond in CONDITIONS}
        for family in FAMILIES
    }
    group_cells: dict = {}  # the Figure 65 split within suppress (jlens arm)
    for family in FAMILIES:
        group_cells[family] = {}
        for group in ("dismissal", "negated-think"):
            sub = [r for r in records if r["family"] == family and r["group"] == group]
            k = sum(r["hit_jlens"] for r in sub)
            lb, ub = wilson(k, len(sub)) if sub else (0.0, 1.0)
            group_cells[family][group] = {
                "n": len(sub), "hits": k, "wilson_95": [lb, ub],
            }
    pooled_baseline = {arm: cell(None, "baseline", arm) for arm in ARMS}

    def contrast(family: str, a: str, b: str, arm: str = "jlens") -> list[float]:
        ca, cb = cells[family][a][arm], cells[family][b][arm]
        return list(newcombe_diff(cb["hits"], cb["n"], ca["hits"], ca["n"]))

    contrasts = {
        family: {
            "focus_minus_suppress": contrast(family, "focus", "suppress"),
            "suppress_minus_baseline": contrast(family, "suppress", "baseline"),
            "focus_minus_control": contrast(family, "focus", "control"),
            "jlens_minus_logitlens_focus": list(
                newcombe_diff(
                    cells[family]["focus"]["logitlens"]["hits"],
                    cells[family]["focus"]["logitlens"]["n"],
                    cells[family]["focus"]["jlens"]["hits"],
                    cells[family]["focus"]["jlens"]["n"],
                )
            ),
        }
        for family in FAMILIES
    }

    baseline_clean = pooled_baseline["jlens"]["wilson_95"][1] <= BASELINE_UB
    contrast_positive = all(
        contrasts[f]["focus_minus_suppress"][1] > 0 for f in FAMILIES
    )
    would_modulate = baseline_clean and contrast_positive

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "frame": FRAME,
            "grading": "tracked token is top-1 at any (layer, position) — Fig 10",
            "pairing": "carriers[(i + j) mod 20]; baseline carriers[j mod 20]",
            "baseline_ub": BASELINE_UB,
        },
        "dropped_targets": dropped,
        "n_trials": len(records),
        "cells": cells,
        "suppress_group_split_jlens": group_cells,
        "pooled_baseline": pooled_baseline,
        "contrasts_jlens": contrasts,
        "would_gate": {
            "baseline_clean": baseline_clean,
            "focus_minus_suppress_positive_both_families": contrast_positive,
            "modulates_wording": would_modulate,
        },
        "trials": records,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    print(f"\nDESCRIPTIVE SUMMARY: {smoke}{len(records)} trials, dropped targets: {dropped or 'none'}")
    for family in FAMILIES:
        line = f"  {family:9s} jlens: "
        line += "  ".join(
            f"{cond} {cells[family][cond]['jlens']['hits']}/{cells[family][cond]['jlens']['n']}"
            f"[{cells[family][cond]['jlens']['wilson_95'][0]:.2f},{cells[family][cond]['jlens']['wilson_95'][1]:.2f}]"
            for cond in CONDITIONS
        )
        print(line)
        fs = contrasts[family]["focus_minus_suppress"]
        fc = contrasts[family]["focus_minus_control"]
        sb = contrasts[family]["suppress_minus_baseline"]
        print(
            f"            contrasts: focus−suppress {fs[0]:+.3f}[{fs[1]:+.3f},{fs[2]:+.3f}]"
            f"  suppress−baseline {sb[0]:+.3f}[{sb[1]:+.3f},{sb[2]:+.3f}]"
            f"  focus−control {fc[0]:+.3f}[{fc[1]:+.3f},{fc[2]:+.3f}]"
        )
    pb = pooled_baseline["jlens"]
    print(
        f"  pooled baseline (jlens): {pb['hits']}/{pb['n']} "
        f"Wilson UB {pb['wilson_95'][1]:.3f} {'CLEAN' if baseline_clean else 'NOT CLEAN'} (bar {BASELINE_UB})"
    )
    print(
        f"  would-gate wording: {'modulates' if would_modulate else 'does not modulate'}"
        " — descriptive framing applies (triple readability NULL); no property claim"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/directed-modulation-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
