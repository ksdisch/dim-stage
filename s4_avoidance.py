"""s4_avoidance.py — S4: naming a concept vs deliberately avoiding it (D27–D30).

The question, in plain terms: S3 showed the workspace band is load-bearing for
flexible tasks. The paper's Figure-69 appendix experiment makes the sharper
within-task claim that **not saying something takes workspace machinery that
saying it does not** (the inclusion/exclusion paradigm: avoiding a primed
response is the mark of deliberate processing). A clue sentence implies a
concept without naming it; the model is asked either to *name* the implied
concept or to *name something of the same kind that is NOT it*. Deleting just
that concept's lens direction (k = 1, S3's projection-removal operator) at the
**early** third of the band should leave naming intact but make the model
blurt the concept it was told to avoid (~5× at Claude scale); at the **late**
third it should suppress the concept under both instructions.

Frozen conventions (D27–D30, all pre-declared; descriptive mode throughout):

- **Items** (D27c): `s4-avoidance-items.json`, constructed under frozen rules
  from measured vocabularies (the reference ships no set), 20 concepts × 3
  clues; concept/control single-token filter drops are counted; test-guarded
  against clue/concept leakage.
- **Conditions** (D28): clean, primed-concept ablation at early / middle /
  late sub-band thirds (S1's convention), and the same-category control
  concept at early — every prompt position, k = 1, D6-style read-back on
  every edit.
- **Readout** (D29): chat-template user turn; binary = concept's single-token
  form is the greedy next token; concept softmax mass as texture. Competence
  gate: clean naming produces AND clean avoidance does not (greedy-based;
  verbatim P ≥ .85 / ≤ .15 rates reported beside it). Gated set = primary.
- **Would-gate** (D30): "avoidance-dissociation-consistent" iff (1) early
  primed ablation raises avoidance failure CI-cleanly over clean; (2) naming
  under early primed ablation shows NO CI-clean drop (pre-declared null leg);
  (3) early primed beats early control CI-cleanly on avoidance failure.
  Late-layer suppression is texture. INVALID (exit 2) on wrong-arm inputs.

Run:  uv run python -u s4_avoidance.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct \
          --lens lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import torch
import transformers

from fitter import SubjectModel, _record_residuals
from intervention import Edit, ablate, edit_residuals, jlens_vector
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import encode_chat, output_logits, token_forms
from readability import proportional_band
from s3_selectivity import READBACK_TOL, degeneracy, rate_cell
from stats import excludes_zero, newcombe_diff, wilson

ITEMS_PATH = "s4-avoidance-items.json"
EXPECTED_ITEMS = 60  # 20 concepts x 3 clues, frozen 2026-07-17
NAMING_Q = (
    'Consider this sentence: "{clue}"\n\n'
    "Which {noun} is the sentence describing? Answer with just the name."
)
AVOIDANCE_Q = (
    'Consider this sentence: "{clue}"\n\n'
    "Name a {noun} that the sentence is NOT describing. Answer with just the name."
)
INSTRUCTIONS = (("naming", NAMING_Q), ("avoidance", AVOIDANCE_Q))
CONDITIONS = (  # D28 cells + the D31 control tiers (S4b)
    "clean",
    "primed_early", "primed_middle", "primed_late",
    "control_early", "control_middle", "control_late",
)
P_NAME_FLOOR = 0.85  # paper's verbatim competence thresholds — reported, not gating
P_AVOID_CEIL = 0.15
MIN_N = 20
COLLAPSE_SHARE = 0.5


def load_items(path: str = ITEMS_PATH) -> list[dict]:
    """The frozen constructed set, guarded against drift and clue leakage."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — the frozen D27 item set")
    items = json.load(open(path))["items"]
    fields = ("name", "category", "noun", "concept", "control", "clue")
    if len(items) != EXPECTED_ITEMS or any(
        not all(f in item for f in fields) for item in items
    ):
        raise ValueError(
            f"{path} drifted from the frozen {EXPECTED_ITEMS}-item shape — "
            "re-freeze before running"
        )
    for item in items:
        clue = item["clue"].lower()
        for word in (item["concept"], item["control"]):
            if word.lower() in clue:
                raise ValueError(
                    f"item {item['name']}: clue contains {word!r} — violates "
                    "the frozen construction rules"
                )
        if item["concept"] == item["control"]:
            raise ValueError(f"item {item['name']}: control equals concept")
    return items


def sub_band_thirds(band: list[int]) -> dict[str, list[int]]:
    """S1's early/middle/late thirds of the frozen band (D28)."""
    n = len(band)
    third = max(1, n // 3)
    return {
        "early": band[:third],
        "middle": band[third : 2 * third],
        "late": band[2 * third :],
    }


def concept_ablation_edits(
    jacobians: dict[int, torch.Tensor], layers: list[int], u_row: torch.Tensor
) -> dict[int, Edit]:
    """k = 1 edits: project the one concept direction (layer's own J_l, M1 raw-
    row convention) out of every position, with the D6-style read-back."""
    edits: dict[int, Edit] = {}
    for layer in layers:
        v = jlens_vector(jacobians[layer], u_row.float())

        def edit(h: torch.Tensor, layer=layer, v=v) -> torch.Tensor:
            hs = h[0]  # [seq, d_model]
            direction = v.to(device=hs.device, dtype=torch.float32)
            directions = direction.expand(hs.shape[0], 1, -1)
            out = ablate(hs.float(), directions).to(hs.dtype)
            leftover = (out.float() @ direction).abs()
            scale = direction.norm() * hs.float().norm(dim=-1)
            worst = float((leftover / scale.clamp_min(1e-30)).max())
            if worst > READBACK_TOL:
                fail_invalid(
                    f"S4 read-back failed at layer {layer}: surviving "
                    f"projection {worst:.2e} > {READBACK_TOL:.0e}"
                )
            return out.unsqueeze(0)

        edits[layer] = edit
    return edits


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


def concept_mass(logits: torch.Tensor, forms: list[int]) -> float:
    """Softmax probability mass on the concept's single-token forms — the
    paper-comparable texture readout (their gate thresholds are probabilities)."""
    probs = torch.softmax(logits.float(), dim=-1)
    return float(sum(probs[t] for t in forms))


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
    thirds = sub_band_thirds(band)

    items = load_items()
    planned, dropped = [], []
    for item in items:
        forms = {
            "concept": token_forms(item["concept"], tok),
            "control": token_forms(item["control"], tok),
        }
        missing = [k for k, v in forms.items() if not v]
        if missing:
            dropped.append({"name": item["name"], "missing": missing})
        else:
            planned.append((item, forms))

    print(
        f"validated: {args.model_id} n_layers={subject.n_layers}, lens n_prompts="
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]} | thirds "
        + ", ".join(f"{t} L{v[0]}–L{v[-1]}" for t, v in thirds.items())
        + f" | items {len(planned)} gradable / {len(dropped)} dropped"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    if args.limit is not None:
        planned = planned[: args.limit]

    records = []
    for i, (item, forms) in enumerate(planned):
        start = time.perf_counter()
        u_concept = unembed_rows[forms["concept"][0]]  # bare form — M1 convention
        u_control = unembed_rows[forms["control"][0]]
        record = {
            "name": item["name"], "category": item["category"],
            "concept": item["concept"], "control": item["control"],
            "instructions": {},
        }
        for inst_name, template in INSTRUCTIONS:
            prompt = template.format(clue=item["clue"], noun=item["noun"])
            input_ids = encode_chat(subject, prompt)
            cells = {}
            for condition in CONDITIONS:
                if condition == "clean":
                    edits = None
                else:
                    kind, tier = condition.split("_")
                    u_row = u_control if kind == "control" else u_concept
                    edits = concept_ablation_edits(jacobians, thirds[tier], u_row)
                logits = output_logits(subject, input_ids, edits)
                greedy = int(logits.argmax())
                cells[condition] = {
                    "produced": greedy in forms["concept"],
                    "greedy": tok.decode([greedy]),
                    "greedy_id": greedy,
                    "concept_mass": concept_mass(logits, forms["concept"]),
                }
            record["instructions"][inst_name] = cells
        naming, avoid = (record["instructions"][k]["clean"] for k in ("naming", "avoidance"))
        record["gate_greedy"] = naming["produced"] and not avoid["produced"]
        record["gate_verbatim_p"] = (
            naming["concept_mass"] >= P_NAME_FLOOR
            and avoid["concept_mass"] <= P_AVOID_CEIL
        )
        records.append(record)
        print(
            f"[{i + 1}/{len(planned)}] {item['name']}: "
            f"name={'OK' if naming['produced'] else 'x'}({naming['greedy']!r}) "
            f"avoid={'OK' if not avoid['produced'] else 'FAIL'}({avoid['greedy']!r}) "
            f"gate={'IN' if record['gate_greedy'] else 'out'} "
            f"({time.perf_counter() - start:.1f}s)"
        )

    gated = [r for r in records if r["gate_greedy"]]
    n = len(gated)

    def cell(inst: str, condition: str, produced: bool = True) -> dict:
        k = sum(
            r["instructions"][inst][condition]["produced"] is produced for r in gated
        )
        return rate_cell(k, n)

    naming_success = {c: cell("naming", c, True) for c in CONDITIONS}
    avoid_failure = {c: cell("avoidance", c, True) for c in CONDITIONS}

    k_clean_fail = avoid_failure["clean"]["hits"]  # 0 by gate construction
    k_early_fail = avoid_failure["primed_early"]["hits"]
    k_ctrl_fail = avoid_failure["control_early"]["hits"]
    k_early_name = naming_success["primed_early"]["hits"]
    leg1 = newcombe_diff(k_clean_fail, n, k_early_fail, n) if n else (0.0, -1.0, 1.0)
    leg2 = newcombe_diff(k_early_name, n, n, n) if n else (0.0, -1.0, 1.0)
    leg3 = newcombe_diff(k_ctrl_fail, n, k_early_fail, n) if n else (0.0, -1.0, 1.0)
    legs = {
        "1_early_breaks_avoidance": {
            "newcombe_early_minus_clean_failure": list(leg1),
            "holds": excludes_zero(leg1[1], leg1[2]) and leg1[0] > 0,
        },
        "2_early_spares_naming_null_leg": {
            "newcombe_clean_minus_early_naming": list(leg2),
            "holds": not (excludes_zero(leg2[1], leg2[2]) and leg2[0] > 0),
        },
        "3_primed_beats_control": {
            "newcombe_primed_minus_control_failure": list(leg3),
            "holds": excludes_zero(leg3[1], leg3[2]) and leg3[0] > 0,
        },
    }
    consistent = all(leg["holds"] for leg in legs.values()) and n >= MIN_N

    # D31 (S4b): is the late-tier switch concept-specific?
    k_primed_late = naming_success["primed_late"]["hits"]
    k_ctrl_late = naming_success["control_late"]["hits"]
    s4b = newcombe_diff(k_primed_late, n, k_ctrl_late, n) if n else (0.0, -1.0, 1.0)
    late_switch = {
        "newcombe_control_minus_primed_late_naming": list(s4b),
        "holds": excludes_zero(s4b[1], s4b[2]) and s4b[0] > 0,
        "underpowered": n < MIN_N,
    }

    guard = {}
    for inst_name, _ in INSTRUCTIONS:
        for condition in CONDITIONS:
            g = degeneracy(
                [r["instructions"][inst_name][condition]["greedy_id"] for r in records],
                tok,
            )
            guard[f"{inst_name}_{condition}"] = g
    for r in records:
        for inst in r["instructions"].values():
            for c in inst.values():
                del c["greedy_id"]

    mass = {
        inst: {
            c: (
                sum(r["instructions"][inst][c]["concept_mass"] for r in gated) / n
                if n
                else None
            )
            for c in CONDITIONS
        }
        for inst, _ in INSTRUCTIONS
    }

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "thirds": thirds,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "instructions": {k: v for k, v in INSTRUCTIONS},
            "conditions": list(CONDITIONS),
            "gate": "greedy-based (D29); verbatim P .85/.15 reported alongside",
            "readback_tol": READBACK_TOL,
            "min_n": MIN_N,
            "collapse_share": COLLAPSE_SHARE,
        },
        "dropped_single_token_prefilter": dropped,
        "competence": {
            "items_run": len(records),
            "gate_greedy": n,
            "gate_verbatim_p": sum(r["gate_verbatim_p"] for r in records),
            "naming_clean_all_items": sum(
                r["instructions"]["naming"]["clean"]["produced"] for r in records
            ),
            "avoid_clean_fail_all_items": sum(
                r["instructions"]["avoidance"]["clean"]["produced"] for r in records
            ),
        },
        "items": records,
        "naming_success_gated": naming_success,
        "avoidance_failure_gated": avoid_failure,
        "mean_concept_mass_gated": mass,
        "would_gate": {
            "legs": legs,
            "underpowered": n < MIN_N,
            "avoidance_dissociation_consistent": consistent,
        },
        "late_switch_specificity": late_switch,  # D31 (S4b)
        "degeneracy_guard": guard,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    collapsed = [k for k, g in guard.items() if g["collapsed"]]
    print(
        f"\nDESCRIPTIVE VERDICT: {smoke}gate {n}/{len(records)} items"
        f"{' UNDERPOWERED' if n < MIN_N else ''} (verbatim-P would keep "
        f"{results['competence']['gate_verbatim_p']}) | avoidance failure "
        f"clean→early→ctrl: {k_clean_fail}/{n}, {k_early_fail}/{n}, {k_ctrl_fail}/{n} | "
        f"naming early {k_early_name}/{n} | middle/late failure "
        f"{avoid_failure['primed_middle']['hits']}/{n}, "
        f"{avoid_failure['primed_late']['hits']}/{n}; late naming "
        f"{naming_success['primed_late']['hits']}/{n} | legs "
        f"1={'HOLDS' if legs['1_early_breaks_avoidance']['holds'] else 'fails'} "
        f"2={'HOLDS' if legs['2_early_spares_naming_null_leg']['holds'] else 'fails'} "
        f"3={'HOLDS' if legs['3_primed_beats_control']['holds'] else 'fails'} → "
        f"{'avoidance-dissociation-consistent' if consistent else 'NOT shown'} "
        f"(descriptive) | D31 late-switch naming primed→ctrl: "
        f"{k_primed_late}/{n} → {k_ctrl_late}/{n}, "
        f"{s4b[0]:+.3f} [{s4b[1]:+.3f},{s4b[2]:+.3f}] → "
        f"{'concept-SPECIFIC' if late_switch['holds'] else 'not shown'}"
        f"{' UNDERPOWERED' if late_switch['underpowered'] else ''} | degeneracy: "
        f"{', '.join(collapsed) if collapsed else 'none'} | no property claim "
        f"(triple readability NULL; descriptive mode)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/s4-avoidance-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
