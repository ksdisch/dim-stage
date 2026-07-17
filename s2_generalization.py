"""s2_generalization.py — S2: one swapped thought, sixteen functions (frozen D19–D22).

The question, in plain terms: the workspace story says a written thought is
**broadcast** — consumable by many downstream circuits, not just one. The
paper's test: sixteen one-line templates apply different functions to the same
argument ("The capital of {arg} is the city of", "Most people in {arg}
speak", …; 4 categories × 4 functions). Swap the argument's J-lens coordinates
for another argument's — the identical swap clamped at every prompt position
and band layer, regardless of template — and check whether each template's
greedy answer becomes the right answer *for the swapped-in argument*. The
paper's anchors on this exact shipped item set (Sonnet-scale): **76/192**
top-1 at α = 1, **101/192** at the "double strength" α = 2; category order
countries ≫ months > animals > numbers (Figures 18/19/68).

Frozen conventions (D19–D22 + inherited M1/M2 conventions, all pre-declared):

- **Raw-text prompts** (no chat template); readout = greedy next token at the
  final prompt position; success = the target-appropriate answer's single-token
  form is top-1 (rank recorded otherwise — Figure 68's grey numbers).
- **D19**: verbatim reference item set + the standing M0 single-token
  pre-filter — trials whose target answer has no single-token form are skipped
  and counted (12 expected, all animals): the pooled cell is **180 trials**.
  The frozen plan (180 gradable / 12 filtered on the shared Qwen tokenizer) is
  itself a wrong-arm check: a different plan → INVALID.
- **D20**: every trial in both arms — **J-lens** and the standing **J = I**
  falsification (raw unembedding rows through the same coordinate exchange);
  Newcombe 95% CI on (J − I) at each α.
- **D21**: α ∈ {1, 2, 4, 8} — anchored {1, 2} + two owned doublings; the
  α-swap is `h ← h + α·V(σ(c) − c)` (the involution argument pins "double
  strength" as scaling the correction; invariants in test_intervention.py).
  Degeneracy guard: an α × arm cell whose most common greedy token covers
  ≥ half its trials is flagged collapsed (S1's COLLAPSE_SHARE convention —
  with ~60 distinct expected answers, no honest cell is half one token).
- **D22**: three readouts from one run — unconditional (anchor frame),
  baseline-conditioned (source prompt AND the target's own diagonal correct
  unswapped), and per-argument **workspace loading** (band-layer mean cosine
  between the unmodified residual and the argument's J-lens vector at the
  argument + readout positions) against per-argument swap success (the
  paper's Fig-19-right predictor).
- **Swap rows take the prompt-position token form** (the argument appears in
  these prompts, unlike M2's latent intermediates; owned departure from M2's
  bare-form default — deviations row 3). The target row matches the source's
  variant, falling back to its other single-token form (counted, expected 0).
- Would-gate wording (M2's frozen floor, descriptive mode): "routes" iff the
  α = 2 pooled J-lens Wilson LB ≥ 0.5. Per-template cells (N = 12) are
  pre-declared UNDERPOWERED texture. The D6 runtime read-back runs on every
  swap, generalized to α: coordinates must land at (1 − α)·c + α·σ(c).
  INVALID (exit 2) fires on any wrong-arm input before grading; `--dry-run`
  validates + plans and stops; `--limit N` is smoke, never a result.

Run:  uv run python -u s2_generalization.py \
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
from intervention import Edit, jlens_vector, lens_coordinates, swap
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import output_logits, swap_vectors, token_forms, word_rank
from readability import proportional_band
from stats import newcombe_diff, wilson

EXPERIMENT_PATH = "refs/jacobian-lens/data/experiments/flexible-generalization.json"
FROZEN_ALPHAS = (1.0, 2.0, 4.0, 8.0)  # D21: {1,2} anchored, {4,8} owned convention
ARMS = (("jlens", True), ("identity", False))  # D20: standing falsification arm
ANCHORS = {"1": [76, 192], "2": [101, 192]}  # paper Figure 19 left, Sonnet-scale
FROZEN_PLAN = {"trials": 180, "filtered": 12}  # D19, measured 2026-07-16
GATE_WILSON_FLOOR = 0.5  # frozen would-gate wording (α=2 pooled J-lens LB); descriptive
MIN_N = 20
COLLAPSE_SHARE = 0.5  # D21 degeneracy guard threshold (S1 convention)
DESCRIPTIVE_KS = (5, 10)


def load_categories(path: str = EXPERIMENT_PATH) -> list[dict]:
    """The shipped 4×4×4 item set, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    cats = json.load(open(path))["categories"]
    if len(cats) != 4 or any(
        len(c.get("args", [])) != 4
        or len(c.get("funcs", [])) != 4
        or any(
            "{arg}" not in f.get("template", "")
            or set(f.get("answers", {})) != set(c["args"])
            for f in c.get("funcs", [])
        )
        for c in cats
    ):
        raise ValueError(
            "flexible-generalization.json drifted from the extracted 4 categories "
            "× 4 args × 4 funcs shape with per-arg answers — re-extract before running"
        )
    return cats


def variant_forms(word: str, tokenizer) -> dict[str, int]:
    """Single-token ids keyed by variant ('bare', 'space') — the swap-row
    convention needs to know WHICH form sits in the prompt, so this keeps the
    two forms separate where `token_forms` pools them."""
    out: dict[str, int] = {}
    for key, text in (("bare", word), ("space", " " + word)):
        enc = tokenizer(text, add_special_tokens=False).input_ids
        if len(enc) == 1:
            out[key] = enc[0]
    return out


def prompt_form(
    prompt_ids: list[int], forms: dict[str, int]
) -> tuple[str, int, int] | None:
    """(variant, token id, position) of the argument as it actually appears in
    the tokenized prompt; None if neither single-token form is present (the
    argument merged with a neighbor — counted, expected 0)."""
    for variant in ("space", "bare"):  # mid-sentence templates dominate
        tid = forms.get(variant)
        if tid is not None and tid in prompt_ids:
            return variant, tid, prompt_ids.index(tid)
    return None


def plan_trials(cats: list[dict], tokenizer, encode_ids) -> tuple[list, list, list]:
    """Apply D19 to the shipped set. Returns (diagonals, trials, filtered):
    64 diagonal baselines, the gradable ordered swap trials, and the skipped
    trials with their reasons. `encode_ids` maps a prompt string to its token
    id list (the runner passes the subject's own encoder; tests stub it)."""
    diagonals, trials, filtered = [], [], []
    for cat in cats:
        arg_forms = {a: variant_forms(a, tokenizer) for a in cat["args"]}
        for func in cat["funcs"]:
            fid = f"{cat['name']}/{func['name']}"
            per_arg: dict[str, tuple[str, list[int], tuple | None]] = {}
            for arg in cat["args"]:
                prompt = func["template"].format(arg=arg)
                ids = encode_ids(prompt)
                pf = prompt_form(ids, arg_forms[arg])
                per_arg[arg] = (prompt, ids, pf)
                answer = func["answers"][arg]
                diagonals.append(
                    {
                        "func": fid,
                        "category": cat["name"],
                        "arg": arg,
                        "prompt": prompt,
                        "prompt_ids": ids,
                        "arg_variant": pf[0] if pf else None,
                        "arg_token": pf[1] if pf else None,
                        "arg_pos": pf[2] if pf else None,
                        "answer": answer,
                        "answer_forms": token_forms(answer, tokenizer),
                    }
                )
            for source in cat["args"]:
                for target in cat["args"]:
                    if source == target:
                        continue
                    answer = func["answers"][target]
                    answer_forms = token_forms(answer, tokenizer)
                    if not answer_forms:
                        filtered.append(
                            {
                                "func": fid,
                                "source": source,
                                "target": target,
                                "answer": answer,
                                "reason": "answer_no_single_token",
                            }
                        )
                        continue
                    if not arg_forms[target]:
                        filtered.append(
                            {
                                "func": fid,
                                "source": source,
                                "target": target,
                                "answer": answer,
                                "reason": "target_arg_no_single_token",
                            }
                        )
                        continue
                    prompt, ids, pf = per_arg[source]
                    if pf is None:  # merged tokenization — fall back, counted
                        variant, u_s_token = None, token_forms(source, tokenizer)[0]
                        fallback = True
                    else:
                        variant, u_s_token, _ = pf
                        fallback = False
                    tf = arg_forms[target]
                    if variant in tf:
                        u_t_token = tf[variant]
                    else:
                        u_t_token = next(iter(tf.values()))
                        fallback = True
                    trials.append(
                        {
                            "func": fid,
                            "category": cat["name"],
                            "source": source,
                            "target": target,
                            "prompt": prompt,
                            "prompt_ids": ids,
                            "u_s_token": u_s_token,
                            "u_t_token": u_t_token,
                            "variant": variant,
                            "fallback": fallback,
                            "answer": answer,
                            "answer_forms": answer_forms,
                            "source_answer_forms": token_forms(
                                func["answers"][source], tokenizer
                            ),
                        }
                    )
    return diagonals, trials, filtered


def checked_alpha_swap_edits(
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]], alpha: float
) -> dict[int, Edit]:
    """Per-layer α-swap edits carrying the D6 runtime read-back, generalized:
    after `h + α·V(σ(c) − c)` the lens coordinates on the real residual must
    land at (1 − α)·c + α·σ(c) — at α = 1 exactly M1's exchanged-coordinates
    check — else INVALID. Local to S2 (the shared M1 wrapper stays α = 1)."""
    edits: dict[int, Edit] = {}
    for layer, (v_s, v_t) in vectors.items():

        def edit(h: torch.Tensor, layer=layer, v_s=v_s, v_t=v_t) -> torch.Tensor:
            vs = v_s.to(device=h.device, dtype=h.dtype)
            vt = v_t.to(device=h.device, dtype=h.dtype)
            patched = swap(h, vs, vt, alpha)
            c_before = lens_coordinates(h, vs, vt)
            c_after = lens_coordinates(patched, vs, vt)
            expected = (1.0 - alpha) * c_before + alpha * c_before.flip(-1)
            scale = float(c_before.abs().max())
            if not torch.allclose(
                c_after, expected, rtol=1e-2, atol=1e-3 * max(scale, 1.0)
            ):
                fail_invalid(
                    f"read-back failed at layer {layer}: coordinates did not land "
                    f"at (1-α)c + ασ(c) (D6 runtime self-check, α={alpha:g})"
                )
            return patched

        edits[layer] = edit
    return edits


def baseline_readout(
    subject: SubjectModel,
    input_ids: torch.Tensor,
    band: list[int],
    jacobians: dict[int, torch.Tensor],
    u_row: torch.Tensor | None,
    arg_pos: int | None,
) -> tuple[torch.Tensor, float | None]:
    """One unmodified forward: final-position output logits (baseline grading)
    plus the D22 workspace-loading readout — mean over band layers of
    cosine(residual, the argument's J-lens vector) at the argument and readout
    positions ("averaged over the argument and readout positions in the
    unmodified forward pass", Fig. 19 right; band-layer mean is our owned
    aggregation detail, S2-BRIEF assumptions)."""
    final = subject.n_layers - 1
    with torch.no_grad():
        with _record_residuals(
            subject.layers, sorted({*band, final}), graph_root=None
        ) as res:
            subject.forward(input_ids)
        logits = subject.unembed(res[final][0, -1].float()).float().cpu()
        loading = None
        if u_row is not None:
            cos = []
            readout_pos = input_ids.shape[1] - 1
            positions = [p for p in (arg_pos, readout_pos) if p is not None]
            for layer in band:
                v = jlens_vector(jacobians[layer], u_row)
                v = v / v.norm()
                for pos in positions:
                    h = res[layer][0, pos]
                    cos.append(float(h @ v.to(h.dtype) / h.norm()))
            loading = sum(cos) / len(cos) if cos else None
    return logits, loading


def degeneracy(greedy_ids: list[int]) -> dict:
    """D21 guard for one α × arm cell: the most common greedy token's share."""
    if not greedy_ids:
        return {"attractor_token": None, "share": 0.0, "collapsed": False}
    token, count = Counter(greedy_ids).most_common(1)[0]
    share = count / len(greedy_ids)
    return {
        "attractor_token": token,
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
    try:
        load_categories()
    except (FileNotFoundError, ValueError) as exc:
        fail_invalid(str(exc))
    return band


def cell(subset: list[dict], key: str) -> dict:
    """One α × arm cell over a trial subset: successes with Wilson 95%, rank
    texture, and the displaced rate (None-displaced trials excluded there)."""
    n = len(subset)
    k = sum(t["swaps"][key]["success"] for t in subset)
    lb, ub = wilson(k, n) if n else (0.0, 1.0)
    displaced = [t["swaps"][key]["displaced"] for t in subset]
    displaced = [d for d in displaced if d is not None]
    return {
        "n": n,
        "successes": k,
        "rate": k / n if n else None,
        "wilson_95": [lb, ub],
        "underpowered": n < MIN_N,
        "displaced": sum(displaced),
        "displaced_n": len(displaced),
        "top_k": {
            str(kk): sum(t["swaps"][key]["rank"] <= kk for t in subset)
            for kk in DESCRIPTIVE_KS
        },
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
        help="grade only the first N trials — SMOKE ONLY, never a result",
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

    def encode_ids(prompt: str) -> list[int]:
        return subject.encode(prompt, max_length=64)[0].tolist()

    diagonals, trials, filtered = plan_trials(load_categories(), tok, encode_ids)
    fallbacks = sum(t["fallback"] for t in trials)
    plan = {"trials": len(trials), "filtered": len(filtered)}
    if plan != FROZEN_PLAN:
        fail_invalid(
            f"trial plan {plan} != frozen D19 plan {FROZEN_PLAN} — the tokenizer "
            "or data file is not the one the freeze was measured on"
        )
    print(
        f"validated: {args.model_id} n_layers={subject.n_layers}, lens n_prompts="
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]} | plan: "
        f"{plan['trials']} trials gradable, {plan['filtered']} filtered "
        f"(single-token pre-filter), {fallbacks} variant fallbacks"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    if args.limit is not None:
        trials = trials[: args.limit]

    # --- baselines + loading (64 unmodified forwards) --------------------------
    diag_ok: dict[tuple[str, str], bool] = {}
    for diag in diagonals:
        ids = torch.tensor([diag["prompt_ids"]], device=subject._input_device)
        u_row = (
            unembed_rows[diag["arg_token"]] if diag["arg_token"] is not None else None
        )
        logits, loading = baseline_readout(
            subject, ids, band, jacobians, u_row, diag["arg_pos"]
        )
        greedy = int(logits.argmax())
        diag["greedy"] = tok.decode([greedy])
        diag["correct"] = bool(diag["answer_forms"]) and greedy in diag["answer_forms"]
        diag["loading"] = loading
        diag["gradable"] = bool(diag["answer_forms"])
        del diag["prompt_ids"]
        diag_ok[(diag["func"], diag["arg"])] = diag["correct"]
    n_diag_ok = sum(d["correct"] for d in diagonals)
    n_diag_gradable = sum(d["gradable"] for d in diagonals)
    print(f"baselines: {n_diag_ok}/{n_diag_gradable} diagonals correct")

    # --- swap trials (|trials| × 4 α × 2 arms forwards) -------------------------
    keys = [f"{arm}_a{alpha:g}" for arm, _ in ARMS for alpha in FROZEN_ALPHAS]
    greedy_by_key: dict[str, list[int]] = {key: [] for key in keys}
    for i, trial in enumerate(trials):
        start = time.perf_counter()
        ids = torch.tensor([trial["prompt_ids"]], device=subject._input_device)
        u_s = unembed_rows[trial["u_s_token"]]
        u_t = unembed_rows[trial["u_t_token"]]
        trial["swaps"] = {}
        for arm, use_jacobian in ARMS:
            vectors = swap_vectors(
                jacobians, band, u_s, u_t, use_jacobian=use_jacobian
            )
            for alpha in FROZEN_ALPHAS:
                logits = output_logits(
                    subject, ids, checked_alpha_swap_edits(vectors, alpha)
                )
                greedy = int(logits.argmax())
                key = f"{arm}_a{alpha:g}"
                greedy_by_key[key].append(greedy)
                trial["swaps"][key] = {
                    "success": greedy in trial["answer_forms"],
                    "rank": word_rank(logits, trial["answer_forms"]),
                    "displaced": (
                        greedy not in trial["source_answer_forms"]
                        if trial["source_answer_forms"]
                        else None
                    ),
                    "greedy": tok.decode([greedy]),
                }
        del trial["prompt_ids"]
        marks = " ".join(
            f"a{alpha:g}={'HIT' if trial['swaps'][f'jlens_a{alpha:g}']['success'] else 'miss'}"
            for alpha in FROZEN_ALPHAS
        )
        print(
            f"[{i + 1}/{len(trials)}] {trial['func']} {trial['source']}->"
            f"{trial['target']}: J {marks} | I "
            + " ".join(
                f"a{alpha:g}={'HIT' if trial['swaps'][f'identity_a{alpha:g}']['success'] else 'miss'}"
                for alpha in FROZEN_ALPHAS
            )
            + f" ({time.perf_counter() - start:.1f}s)"
        )

    # --- cells (D22's three readouts) -------------------------------------------
    conditioned = [
        t
        for t in trials
        if diag_ok[(t["func"], t["source"])] and diag_ok[(t["func"], t["target"])]
    ]
    categories = sorted({t["category"] for t in trials})
    pooled = {
        "baselines": {
            "n": n_diag_gradable,
            "correct": n_diag_ok,
            "wilson_95": list(wilson(n_diag_ok, n_diag_gradable)),
        },
        "unconditional": {key: cell(trials, key) for key in keys},
        "conditioned": {
            "n": len(conditioned),
            **{key: cell(conditioned, key) for key in keys},
        },
        "per_category_unconditional": {
            c: {
                key: cell([t for t in trials if t["category"] == c], key)
                for key in keys
            }
            for c in categories
        },
    }
    newcombe = {}
    for frame, subset in (("unconditional", trials), ("conditioned", conditioned)):
        n = len(subset)
        for alpha in FROZEN_ALPHAS:
            k_j = sum(t["swaps"][f"jlens_a{alpha:g}"]["success"] for t in subset)
            k_i = sum(t["swaps"][f"identity_a{alpha:g}"]["success"] for t in subset)
            d, lo, hi = newcombe_diff(k_i, n, k_j, n) if n else (0.0, -1.0, 1.0)
            newcombe[f"{frame}_j_minus_identity_a{alpha:g}"] = [d, lo, hi]

    guard = {}
    for key in keys:
        g = degeneracy(greedy_by_key[key])
        if g["attractor_token"] is not None:
            g["attractor_token"] = tok.decode([g["attractor_token"]])
        guard[key] = g
    collapsed_cells = [key for key in keys if guard[key]["collapsed"]]

    loading_by_arg = []
    for c in categories:
        for arg in sorted(
            {d["arg"] for d in diagonals if d["category"] == c}
        ):
            loads = [
                d["loading"]
                for d in diagonals
                if d["category"] == c and d["arg"] == arg and d["loading"] is not None
            ]
            as_source = [
                t for t in trials if t["category"] == c and t["source"] == arg
            ]
            loading_by_arg.append(
                {
                    "category": c,
                    "arg": arg,
                    "mean_loading": sum(loads) / len(loads) if loads else None,
                    "swap_success_a1_jlens": [
                        sum(t["swaps"]["jlens_a1"]["success"] for t in as_source),
                        len(as_source),
                    ],
                }
            )

    per_template = {  # texture — N = 12 per template, pre-declared UNDERPOWERED
        fid: {
            f"a{alpha:g}": [
                sum(
                    t["swaps"][f"jlens_a{alpha:g}"]["success"]
                    for t in trials
                    if t["func"] == fid
                ),
                sum(t["func"] == fid for t in trials),
            ]
            for alpha in (1.0, 2.0)
        }
        for fid in sorted({t["func"] for t in trials})
    }

    a2 = pooled["unconditional"]["jlens_a2"]
    routes = a2["wilson_95"][0] >= GATE_WILSON_FLOOR
    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "alphas": list(FROZEN_ALPHAS),
            "arms": [a for a, _ in ARMS],
            "grading": "top-1 target-appropriate answer (anchor's grading, D19/D22)",
            "wilson_floor_frozen_wording": GATE_WILSON_FLOOR,
            "min_n": MIN_N,
            "collapse_share": COLLAPSE_SHARE,
            "anchors_192": ANCHORS,
        },
        "plan": {**plan, "fallbacks": fallbacks},
        "filtered": filtered,
        "diagonals": diagonals,
        "trials": trials,
        "pooled": pooled,
        "newcombe": newcombe,
        "degeneracy_guard": guard,
        "loading_by_arg": loading_by_arg,
        "per_template_jlens": per_template,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    n = len(trials)
    a1 = pooled["unconditional"]["jlens_a1"]
    cn = pooled["conditioned"]["n"]
    ca2 = pooled["conditioned"]["jlens_a2"]
    print(
        f"\nDESCRIPTIVE VERDICT: {smoke}baselines {n_diag_ok}/{n_diag_gradable} | "
        f"α=1 J-lens {a1['successes']}/{n} Wilson[{a1['wilson_95'][0]:.3f},"
        f"{a1['wilson_95'][1]:.3f}] vs anchor 76/192 | "
        f"α=2 J-lens {a2['successes']}/{n} Wilson[{a2['wilson_95'][0]:.3f},"
        f"{a2['wilson_95'][1]:.3f}] vs anchor 101/192 | "
        f"would-gate (α=2 LB ≥ {GATE_WILSON_FLOOR}): "
        f"{'routes' if routes else 'does not route'} (descriptive) | "
        f"J−I α=1 {newcombe['unconditional_j_minus_identity_a1'][0]:+.3f} "
        f"[{newcombe['unconditional_j_minus_identity_a1'][1]:+.3f},"
        f"{newcombe['unconditional_j_minus_identity_a1'][2]:+.3f}] | "
        f"conditioned n={cn}{' UNDERPOWERED' if cn < MIN_N else ''}: α=2 J "
        f"{ca2['successes']}/{cn} | degeneracy: "
        f"{', '.join(collapsed_cells) if collapsed_cells else 'none'} | "
        f"no property claim (triple readability NULL; descriptive mode)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/s2-generalization-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
