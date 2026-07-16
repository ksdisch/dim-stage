"""m1_verbal_report.py — M1 protocol 1: rewrite the workspace, watch the report (frozen D5/D7).

The question, in plain terms: the model is asked `Think of a {category}. Answer
in one word.`; it has an answer ready at the final prompt token. We **swap** that
answer for a different candidate inside the residual stream at every frozen-band
layer (patching in lens coordinates — `intervention.py`), let the forward pass
continue, and read where the swapped-in candidate lands in the model's actual
next-token distribution. If the verbal report tracks the workspace, the swap
drags the answer with it.

Trial construction (D5, reference-faithful — every choice cited or M0-inherited):

- Prompts built with the tokenizer's **chat template** (`apply_chat_template`,
  generation prompt appended — including Qwen's default system message, verbatim
  template behavior). Readout position = the **final prompt token**, the analog
  of the paper's `Assistant:` colon (M1-BRIEF deviations row 2).
- Swap-out `s` = the model's **greedy next token** at that position.
- Swap-in trials = the **first 10 listed candidates** per category, skipping the
  spontaneous answer itself; a candidate with no single-token form ({`w`, `␣w`})
  is dropped and counted (deviations row 3); a candidate whose baseline rank is
  already ≤ 10 is excluded (the paper grades "candidates starting at rank ≥ 11").
  The swap token prefers the **bare form** — the shape an answer takes right
  after the chat template's trailing newline; grading ranks are min over both
  single-token forms (M0's convention).
- The swap is applied at **every band layer and every token position** (paper:
  "at all token positions"), each layer using its own `J_l`.

Two arms, mirroring D4 (frozen D7):

- **Arm 1** — hit iff the swapped-in candidate reaches **top-5** (the anchor's k)
  at the readout; Wilson 95% CI on the pooled rate. Rank-1 and top-10 reported
  descriptively. n < 20 is pre-declared UNDERPOWERED.
- **Arm 2** — every swap repeated with **raw unembedding rows** (J = I) in place
  of J-lens vectors; Newcombe 95% CI on the (J − identity) difference — the
  standing falsification arm: does J-transport help *writing*, not just reading?

**Mode: descriptive.** All three subjects are NULL on M0 readability, so the
pre-registered re-scope applies (KICKOFF risk #1): identical numbers, Wilson
floor wording frozen but never gating, characterization framing, no property
claim either direction.

Runtime self-check (D6): every swap application verifies on the real residual
that the lens coordinates come back exchanged (c' = σ(c)); a failure exits
INVALID, like every other wrong-arm input. INVALID (exit 2) also fires on
lens/model mismatch, a band that disagrees with the frozen table, drifted
experiment data, or a tokenizer without a chat template — before any grading.
`--dry-run` performs exactly this validation and stops.

Run:  uv run python -u m1_verbal_report.py \
          --model-id Qwen/Qwen2.5-0.5B-Instruct \
          --lens lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import transformers

from fitter import SubjectModel, _record_residuals
from intervention import Edit, edit_residuals, jlens_vector, lens_coordinates, swap
from m0_readability_gate import FROZEN_BANDS, fail_invalid
from readability import proportional_band
from stats import newcombe_diff, wilson

EXPERIMENT_PATH = "refs/jacobian-lens/data/experiments/verbal-report.json"
#: Oracle-drift guard, M0 pattern: shape as extracted 2026-07-16.
EXPECTED_CATEGORIES = 14
EXPECTED_WORDS = 14

PROMPT_TEMPLATE = "Think of a {category}. Answer in one word."
N_CANDIDATES = 10  # first 10 listed per category (reference README)
MIN_BASELINE_RANK = 11  # paper: grade only candidates starting at rank >= 11
GATE_K = 5  # the anchor's top-5 grading (D7 Arm 1)
DESCRIPTIVE_KS = (1, 10)
MIN_N = 20  # pooled n below this is pre-declared UNDERPOWERED


def load_candidates(path: str = EXPERIMENT_PATH) -> dict[str, list[str]]:
    """The 14x14 candidates table, guarded against oracle drift."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    candidates = json.load(open(path))["candidates"]
    if len(candidates) != EXPECTED_CATEGORIES or any(
        len(words) != EXPECTED_WORDS for words in candidates.values()
    ):
        raise ValueError(
            "verbal-report.json drifted from the extracted 14 categories x 14 "
            "words — re-extract before running"
        )
    return candidates


def token_forms(word: str, tokenizer) -> list[int]:
    """Single-token ids for `word` — M0's {`w`, `␣w`} convention, kept *ordered*
    (bare form first) so the swap token can prefer the bare form, the shape an
    answer takes right after the chat template's trailing newline. Grading uses
    the min over all forms, so order never changes a rank."""
    ids: list[int] = []
    for variant in (word, " " + word):
        enc = tokenizer(variant, add_special_tokens=False).input_ids
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def word_rank(logits: torch.Tensor, forms: list[int]) -> int:
    """1-based competition rank of a word: min over its single-token forms of
    1 + (number of vocab tokens with strictly greater logit) — readability.py's
    convention, unchanged."""
    return min(int((logits > logits[t]).sum()) for t in forms) + 1


@dataclass
class Trial:
    """One planned swap-in candidate and why it is (in)eligible."""

    word: str
    status: str  # eligible | no_single_token | spontaneous | baseline_top10
    token: int | None  # the swap-in token id (eligible trials only)
    baseline_rank: int | None


def plan_category(
    words: list[str],
    answer_token: int,
    forms_by_word: dict[str, list[int]],
    rank_by_word: dict[str, int | None],
) -> list[Trial]:
    """Apply D5's trial rules to one category's first N_CANDIDATES words."""
    trials = []
    for word in words[:N_CANDIDATES]:
        forms = forms_by_word[word]
        if not forms:
            trials.append(Trial(word, "no_single_token", None, None))
        elif answer_token in forms:
            trials.append(Trial(word, "spontaneous", None, rank_by_word[word]))
        elif rank_by_word[word] < MIN_BASELINE_RANK:
            trials.append(Trial(word, "baseline_top10", None, rank_by_word[word]))
        else:
            trials.append(Trial(word, "eligible", forms[0], rank_by_word[word]))
    return trials


def encode_chat(subject: SubjectModel, text: str) -> torch.Tensor:
    """One user turn through the tokenizer's own chat template, generation
    prompt appended — the frozen D5 prompt construction."""
    input_ids = subject.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,  # the bare [1, seq] tensor, not a BatchEncoding
    )
    return input_ids.to(subject._input_device)


def checked_swap_edits(
    vectors: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> dict[int, Edit]:
    """Per-layer swap edits carrying the D6 runtime read-back: on the real
    residual, the lens coordinates after the patch must equal the exchanged
    coordinates before it (σ(c), fp32 tolerance) — else INVALID. The check is
    two extra [seq]·[d_model] dot products per layer; effectively free."""
    edits: dict[int, Edit] = {}
    for layer, (v_s, v_t) in vectors.items():

        def edit(h: torch.Tensor, layer=layer, v_s=v_s, v_t=v_t) -> torch.Tensor:
            vs = v_s.to(device=h.device, dtype=h.dtype)
            vt = v_t.to(device=h.device, dtype=h.dtype)
            patched = swap(h, vs, vt)
            c_before = lens_coordinates(h, vs, vt)
            c_after = lens_coordinates(patched, vs, vt)
            scale = float(c_before.abs().max())
            if not torch.allclose(
                c_after, c_before.flip(-1), rtol=1e-2, atol=1e-3 * max(scale, 1.0)
            ):
                fail_invalid(
                    f"read-back failed at layer {layer}: coordinates did not "
                    "exchange on the real residual (D6 runtime self-check)"
                )
            return patched

        edits[layer] = edit
    return edits


def swap_vectors(
    jacobians: dict[int, torch.Tensor],
    band: list[int],
    u_s: torch.Tensor,
    u_t: torch.Tensor,
    *,
    use_jacobian: bool,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per-band-layer (v_s, v_t): J_lᵀu with each layer's own J_l for the J-lens
    arm, or the raw unembedding rows themselves for the J = I arm."""
    if not use_jacobian:
        return {layer: (u_s, u_t) for layer in band}
    return {
        layer: (jlens_vector(jacobians[layer], u_s), jlens_vector(jacobians[layer], u_t))
        for layer in band
    }


def output_logits(
    subject: SubjectModel, input_ids: torch.Tensor, edits: dict[int, Edit] | None = None
) -> torch.Tensor:
    """The model's next-token logits at the final prompt position — the actual
    output distribution (final-layer residual, unembedded), not a lens readout —
    with the band edits active if given."""
    final_layer = subject.n_layers - 1
    with torch.no_grad(), (
        edit_residuals(subject.layers, edits) if edits else nullcontext()
    ):
        with _record_residuals(subject.layers, [final_layer], graph_root=None) as res:
            subject.forward(input_ids)
        return subject.unembed(res[final_layer][0, -1].float()).float().cpu()


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
    expected_layers = list(range(subject.n_layers - 1))
    if sorted(artifact["J"]) != expected_layers:
        fail_invalid(
            f"lens layers != expected 0..{subject.n_layers - 2}"
        )
    band = proportional_band(subject.n_layers)
    if band != FROZEN_BANDS.get(subject.n_layers):
        fail_invalid(
            f"proportional band {band} disagrees with frozen table "
            f"{FROZEN_BANDS.get(subject.n_layers)} for {subject.n_layers} layers"
        )
    if not getattr(subject.tokenizer, "chat_template", None):
        fail_invalid("tokenizer has no chat template — D5 prompts need one")
    try:
        load_candidates()
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
        help="grade only the first N categories — SMOKE ONLY, never a result",
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
        f"{artifact['n_prompts']}, band L{band[0]}–L{band[-1]}, candidates 14x14 OK"
    )
    if args.dry_run:
        print("DRY-RUN: inputs valid; no trials performed")
        raise SystemExit(0)

    jacobians = {l: artifact["J"][l].to(device) for l in band}
    unembed_rows = hf.lm_head.weight.detach()  # raw rows — deviations row 7
    candidates = load_candidates()
    if args.limit is not None:
        candidates = dict(list(candidates.items())[: args.limit])

    results: dict = {
        "model_id": args.model_id,
        "lens": args.lens,
        "lens_n_prompts": artifact["n_prompts"],
        "band": band,
        "mode": "descriptive",  # triple readability NULL — pre-registered re-scope
        "protocol": {
            "prompt_template": PROMPT_TEMPLATE,
            "n_candidates": N_CANDIDATES,
            "min_baseline_rank": MIN_BASELINE_RANK,
            "gate_k": GATE_K,
            "wilson_floor_frozen_wording": 0.5,
            "min_n": MIN_N,
        },
        "categories": {},
    }
    pooled = {
        "jlens": {k: 0 for k in (GATE_K, *DESCRIPTIVE_KS)},
        "identity": {k: 0 for k in (GATE_K, *DESCRIPTIVE_KS)},
    }
    n_eligible = 0
    status_counts = {"no_single_token": 0, "spontaneous": 0, "baseline_top10": 0}

    for category, words in candidates.items():
        start = time.perf_counter()
        input_ids = encode_chat(subject, PROMPT_TEMPLATE.format(category=category))
        base_logits = output_logits(subject, input_ids)
        answer_token = int(base_logits.argmax())
        answer_text = subject.tokenizer.decode([answer_token])

        forms_by_word = {w: token_forms(w, subject.tokenizer) for w in words[:N_CANDIDATES]}
        rank_by_word = {
            w: (word_rank(base_logits, f) if f else None)
            for w, f in forms_by_word.items()
        }
        trials = plan_category(words, answer_token, forms_by_word, rank_by_word)
        u_s = unembed_rows[answer_token]

        recorded = []
        for trial in trials:
            record = asdict(trial)
            if trial.status != "eligible":
                status_counts[trial.status] += 1
                recorded.append(record)
                continue
            n_eligible += 1
            u_t = unembed_rows[trial.token]
            for arm, use_jacobian in (("jlens", True), ("identity", False)):
                vectors = swap_vectors(
                    jacobians, band, u_s, u_t, use_jacobian=use_jacobian
                )
                logits = output_logits(subject, input_ids, checked_swap_edits(vectors))
                rank = word_rank(logits, forms_by_word[trial.word])
                record[f"{arm}_rank"] = rank
                for k in (GATE_K, *DESCRIPTIVE_KS):
                    pooled[arm][k] += rank <= k
            recorded.append(record)

        results["categories"][category] = {
            "prompt_tokens": int(input_ids.shape[1]),
            "answer": {"token": answer_token, "text": answer_text},
            "trials": recorded,
        }
        eligible_here = sum(1 for t in trials if t.status == "eligible")
        print(
            f"{category}: answer={answer_text!r} eligible={eligible_here}/"
            f"{len(trials)}  ({time.perf_counter() - start:.0f}s)"
        )

    n = n_eligible
    k_j, k_i = pooled["jlens"][GATE_K], pooled["identity"][GATE_K]
    lb, ub = wilson(k_j, n) if n else (0.0, 1.0)
    diff, dlo, dhi = newcombe_diff(k_i, n, k_j, n) if n else (0.0, -1.0, 1.0)
    underpowered = n < MIN_N
    results["pooled"] = {
        "n_eligible": n,
        "excluded": status_counts,
        "jlens": {
            "hits": pooled["jlens"],
            "top5_rate": k_j / n if n else None,
            "wilson_95": [lb, ub],
        },
        "identity": {
            "hits": pooled["identity"],
            "top5_rate": k_i / n if n else None,
        },
        "newcombe_top5_j_minus_identity": [diff, dlo, dhi],
        "underpowered": underpowered,
    }

    smoke = f"SMOKE (limit={args.limit}) — not a result — " if args.limit else ""
    print(
        f"\nDESCRIPTIVE VERDICT: {smoke}swap→report top-5 rate "
        f"{k_j}/{n} Wilson[{lb:.3f},{ub:.3f}]"
        f"{' UNDERPOWERED' if underpowered else ''} | identity arm {k_i}/{n} | "
        f"J−identity diff={diff:+.3f} CI[{dlo:+.3f},{dhi:+.3f}] | "
        f"rank-1 {pooled['jlens'][1]}/{n}, top-10 {pooled['jlens'][10]}/{n} | "
        f"no property claim (triple readability NULL; D7 descriptive mode)"
    )

    if args.limit and not args.out:
        print("smoke run: results not written (pass --out to keep them)")
        raise SystemExit(0)
    out = args.out or (
        "results/verbal-report-" + args.model_id.split("/")[-1].lower() + ".json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results written to {out}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
