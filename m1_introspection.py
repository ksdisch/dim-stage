"""m1_introspection.py — M1 protocol 2: steer a thought in, ask the model to name it (frozen D5/D8).

The setup, in plain terms: the model is told (reference `intro_prompt`, shipped
verbatim) that a researcher may have injected a "thought" into its activations
and is asked to identify it. We **teacher-force** a reply (a *prefill* — we
write the model's answer for it) ending in an open quote, so the very next
token is the reported word — deterministic, no sampling. For each of the 101
shipped concepts, that concept's **steering direction** is added to the
residual stream while the model reads the question; the score is the rank of
the concept's surface word in the next-token distribution at the open quote.

Frozen conventions (D5/D8 + owned details, pre-declared here):

- **Steering direction** (reference README, verbatim): the unit-normalized
  J-lens vector for the surface token, scaled by the layer's **mean residual
  norm** — computed per layer as the mean L2 norm over the frozen D3 fit
  corpus (N=100 WikiText prompts) at the fit's own valid positions
  (`skip_first=16`, final position excluded). Strength α multiplies on top;
  the D8 grid is α ∈ {0, 0.5, 1, 2, 4, 8}, α = 0 the README's control.
- **Where it is applied**: every frozen-band layer (each with its own `J_l`),
  at **every token of the user's question turn** — the third `intro_prompt`
  message's whole templated block. The turn's position range is located by the
  tokenized-prefix property of the chat template, asserted at runtime
  (INVALID if the template doesn't concatenate turns).
- **Prompt construction**: chat template over the three non-empty messages
  with the generation prompt appended, then the prefill's tokens verbatim
  (leading space and all, as shipped). `default` prefill is primary; `word`
  reported alongside (D8).
- **Scoring**: 1-based competition rank, min over the surface's single-token
  forms ({`w`, `␣w`} — M0's convention); the steered token is the bare form
  (first form). Concepts with no single-token form are dropped and counted
  (M1-BRIEF deviations row 5).
- **α = 0 is computed once per prefill** (steering by zero is an exact no-op —
  a D6-tested invariant — so every concept's control rank reads off the same
  unsteered distribution).
- **Metrics per strength**: report rate (rank-1 at the open quote) with a
  Wilson 95% CI, and **median reciprocal rank** (MRR — the paper's Figure 7
  metric; reciprocal rank = 1/rank, so rank 1 → 1.0, rank 10 → 0.1).

**Mode: descriptive** (triple readability NULL, D7): numbers reported in full,
characterization framing, no property claim either direction. Never gates M1.

INVALID (exit 2) fires before any measurement on: lens/model mismatch, a band
off the frozen table, drifted experiment data, a missing/short prompts file,
a tokenizer without a chat template, or a non-concatenating template.
`--dry-run` performs exactly this validation and stops.

Run:  uv run python -u m1_introspection.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct \
          --lens lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import transformers

from fitter import SubjectModel, _record_residuals, valid_position_mask
from intervention import steer_edits
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from m1_verbal_report import output_logits, token_forms, word_rank
from readability import proportional_band
from stats import wilson

EXPERIMENT_PATH = "refs/jacobian-lens/data/experiments/verbal-introspection.json"
EXPECTED_CONCEPTS = 101
PREFILLS = ("default", "word")  # default is primary (D8)
ALPHAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)  # frozen D8 grid; 0 = control
MAX_SEQ_LEN = 128  # the fit's own truncation, for the mean-norm pass


def load_introspection(path: str = EXPERIMENT_PATH) -> dict:
    """The shipped protocol data, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    data = json.load(open(path))
    roles = [m["role"] for m in data["intro_prompt"]]
    if (
        roles != ["user", "assistant", "user", "assistant"]
        or data["intro_prompt"][-1]["content"] != ""
        or sorted(data["prefills"]) != sorted(PREFILLS)
        or len(data["concepts"]) != EXPECTED_CONCEPTS
        or any("surface" not in c for c in data["concepts"])
    ):
        raise ValueError(
            "verbal-introspection.json drifted from the extracted shape "
            "(4 messages ending in an empty assistant turn, 2 prefills, "
            f"{EXPECTED_CONCEPTS} concepts) — re-extract before running"
        )
    return data


def median_reciprocal_rank(ranks: list[int]) -> float:
    """The paper's introspection metric: median over trials of 1/rank."""
    return statistics.median(1.0 / r for r in ranks)


def mean_residual_norms(
    subject: SubjectModel, prompts: list[str], band: list[int]
) -> dict[int, float]:
    """Per-band-layer mean L2 residual norm over the frozen D3 fit corpus, at
    the fit's own valid positions (skip_first=16, final excluded) — the frozen
    D5 steering-scale convention. Prompts too short to have a valid position
    are skipped, matching the fitter."""
    sums = {layer: 0.0 for layer in band}
    count = 0
    with torch.no_grad():
        for prompt in prompts:
            input_ids = subject.encode(prompt, max_length=MAX_SEQ_LEN)
            try:
                mask = valid_position_mask(input_ids.shape[1])
            except ValueError:
                continue
            with _record_residuals(subject.layers, band, graph_root=None) as res:
                subject.forward(input_ids)
            for layer in band:
                norms = res[layer][0, mask].float().norm(dim=-1)
                sums[layer] += float(norms.sum())
            count += int(mask.sum())
    if count == 0:
        raise ValueError("no prompt was long enough for the mean-norm pass")
    return {layer: sums[layer] / count for layer in band}


def build_prompt(
    subject: SubjectModel, messages: list[dict], prefill: str
) -> tuple[torch.Tensor, list[int]]:
    """The teacher-forced sequence and the steering positions.

    input_ids = chat template over the three non-empty messages (generation
    prompt appended) + the prefill's tokens verbatim. Steering positions =
    the third message's whole templated block, located via the template's
    tokenized-prefix property (asserted; INVALID if the template rewrites
    earlier turns instead of concatenating)."""
    tok = subject.tokenizer

    def ids(msgs: list[dict], generation_prompt: bool) -> list[int]:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=generation_prompt, return_dict=False
        )

    two = ids(messages[:2], False)
    three = ids(messages[:3], False)
    full = ids(messages[:3], True)
    if three[: len(two)] != two or full[: len(three)] != three:
        fail_invalid(
            "chat template does not concatenate turns — the question-turn "
            "position range cannot be located"
        )
    prefill_ids = tok(prefill, add_special_tokens=False).input_ids
    input_ids = torch.tensor([full + prefill_ids]).to(subject._input_device)
    positions = list(range(len(two), len(three)))
    return input_ids, positions


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
        print("DRY-RUN: inputs valid; no measurement performed")
        raise SystemExit(0)

    prompts = json.load(open(args.prompts_file))
    start = time.perf_counter()
    norms = mean_residual_norms(subject, prompts, band)
    print(
        f"mean residual norms over the D3 corpus: L{band[0]}={norms[band[0]]:.2f} "
        f".. L{band[-1]}={norms[band[-1]]:.2f}  ({time.perf_counter() - start:.0f}s)"
    )

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()
    data = load_introspection()
    concepts = data["concepts"]
    if args.limit is not None:
        concepts = concepts[: args.limit]
    messages = data["intro_prompt"][:3]

    prompt_by_prefill = {
        name: build_prompt(subject, messages, data["prefills"][name])
        for name in PREFILLS
    }
    control_logits = {
        name: output_logits(subject, input_ids)
        for name, (input_ids, _) in prompt_by_prefill.items()
    }

    per_concept, dropped = [], []
    for i, concept in enumerate(concepts):
        surface = concept["surface"]
        forms = token_forms(surface, subject.tokenizer)
        if not forms:
            dropped.append(surface)
            continue
        u_t = unembed_rows[forms[0]]  # bare form — the steered token
        ranks: dict[str, dict[str, int]] = {}
        start = time.perf_counter()
        for name in PREFILLS:
            input_ids, positions = prompt_by_prefill[name]
            ranks[name] = {"0.0": word_rank(control_logits[name], forms)}
            for alpha in ALPHAS[1:]:
                edits = steer_edits(
                    jacobians, band, u_t, norms, alpha, positions=positions
                )
                logits = output_logits(subject, input_ids, edits)
                ranks[name][str(alpha)] = word_rank(logits, forms)
        per_concept.append(
            {"name": concept["name"], "surface": surface, "token": forms[0], "ranks": ranks}
        )
        default_ranks = [ranks["default"][str(a)] for a in ALPHAS]
        print(
            f"{i + 1}/{len(concepts)} {surface}: default ranks "
            f"{default_ranks}  ({time.perf_counter() - start:.1f}s)"
        )

    n = len(per_concept)
    summary: dict[str, dict[str, dict]] = {}
    for name in PREFILLS:
        summary[name] = {}
        for alpha in ALPHAS:
            ranks = [c["ranks"][name][str(alpha)] for c in per_concept]
            k = sum(r == 1 for r in ranks)
            lb, ub = wilson(k, n) if n else (0.0, 1.0)
            summary[name][str(alpha)] = {
                "n": n,
                "report_rate": k / n if n else None,
                "report_hits": k,
                "wilson_95": [lb, ub],
                "mrr": median_reciprocal_rank(ranks) if n else None,
            }

    results = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "alphas": list(ALPHAS),
            "prefills": list(PREFILLS),
            "steering_scale": "mean L2 residual norm, D3 corpus, fit valid positions",
        },
        "mean_residual_norms": {str(l): norms[l] for l in band},
        "dropped_single_token_prefilter": dropped,
        "concepts": per_concept,
        "summary": summary,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    print(f"\nDESCRIPTIVE SUMMARY: {smoke}n={n} concepts (dropped {len(dropped)})")
    for name in PREFILLS:
        for alpha in ALPHAS:
            cell = summary[name][str(alpha)]
            print(
                f"  {name:8s} α={alpha:<4}: report {cell['report_hits']}/{n} "
                f"Wilson[{cell['wilson_95'][0]:.3f},{cell['wilson_95'][1]:.3f}]  "
                f"MRR={cell['mrr']:.3f}"
            )
    print(
        "no property claim (triple readability NULL; D7 descriptive mode; "
        "introspection never gates — D8)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/introspection-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
