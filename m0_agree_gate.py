"""M0 AGREE gate — does the independent fitter match the reference implementation?

Frozen spec: docs/M0-BRIEF.md D1. Two comparisons, both on Qwen2.5-0.5B-Instruct
(fp32, MPS), both implementations fitting on the byte-identical prompt list:

  1. DECIDES — per-layer relative Frobenius distance (Frobenius norm = a matrix's
     overall size: sqrt of the sum of squared entries):
         ||J_mine - J_ref||_F / ||J_ref||_F   per fitted layer.
     Gate: max over layers <= tolerance, where tolerance = 10x the reference's own
     run-to-run noise floor (the reference refit twice on the same prompts; their
     self-distance is the floor). If the floor is exactly 0 (fully deterministic
     MPS), 1e-4 stands in — recorded in the output.
  2. CONFIRMS — top-1 readout agreement: on held-out prompts, both lenses decode
     every fitted layer at sampled positions; the fraction of (layer, position)
     cells where their rank-1 token matches must have Wilson 95% lower bound >= 0.95.

AGREE = (1) passes AND (2) passes. Exit code 0 on AGREE, 1 otherwise.

The reference jlens is imported HERE ONLY (cross-check oracle pattern) — the
independent fitter (fitter.py) and all measurement harness code stay torch-only.

Run: uv run --group crosscheck python -u m0_agree_gate.py
"""

import logging
import time

import torch
import transformers

import jlens
import jlens.examples
import jlens.fitting

import fitter
from stats import wilson

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_FIT = 16          # AGREE fit subset (identical prompts for all three fits)
N_HELDOUT = 5       # readout-confirmation prompts, disjoint from the fit subset
DIM_BATCH = 8       # frozen on MPS (M0-BRIEF hour-one gate result)
MAX_SEQ_LEN = 128
NOISE_FLOOR_STANDIN = 1e-4
READOUT_WILSON_LB = 0.95
POSITION_STRIDE = 4  # sample every 4th valid position for the readout comparison


def rel_frobenius(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]) -> dict[int, float]:
    """Per-layer ||a - b||_F / ||b||_F (b is the reference arm)."""
    assert sorted(a) == sorted(b), f"layer sets differ: {sorted(a)} vs {sorted(b)}"
    return {l: ((a[l] - b[l]).norm() / b[l].norm()).item() for l in sorted(a)}


def main() -> None:
    if not torch.backends.mps.is_available():
        print("VERDICT: INVALID — MPS not available")
        raise SystemExit(1)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"torch {torch.__version__}, transformers {transformers.__version__}")

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    ref_model = jlens.from_hf(hf, tok)
    subject = fitter.SubjectModel(hf, tok)
    print(f"{MODEL_ID}: n_layers={subject.n_layers}, d_model={subject.d_model}")

    # D3 sanity: the reimplemented loader must be byte-identical to the reference's.
    prompts = fitter.load_wikitext_prompts(N_FIT + N_HELDOUT)
    ref_prompts = jlens.examples.load_wikitext_prompts(N_FIT + N_HELDOUT)
    if prompts != ref_prompts:
        print("VERDICT: INVALID — reimplemented WikiText loader diverges from reference")
        raise SystemExit(1)
    print(f"prompt loaders byte-identical over {len(prompts)} prompts")
    fit_prompts, heldout = prompts[:N_FIT], prompts[N_FIT:]

    # --- The independent fit first (fail-fast if the new code chokes on MPS) ---
    start = time.perf_counter()
    my_J = fitter.fit(
        subject, fit_prompts, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN,
        log=lambda msg: print(f"  {msg}"),
    )
    print(f"independent fit: {time.perf_counter() - start:.0f}s")

    def ref_fit(tag: str) -> jlens.JacobianLens:
        start = time.perf_counter()
        lens = jlens.fitting.fit(
            ref_model, fit_prompts, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN
        )
        print(f"reference fit {tag}: {time.perf_counter() - start:.0f}s")
        return lens

    # --- Noise floor: the reference against itself, identical prompts ---
    ref_a = ref_fit("A")
    ref_b = ref_fit("B (noise-floor refit)")
    floor_by_layer = rel_frobenius(ref_b.jacobians, ref_a.jacobians)
    floor = max(floor_by_layer.values())
    floor_used = floor if floor > 0 else NOISE_FLOOR_STANDIN
    standin_note = "" if floor > 0 else f" (exactly 0 -> stand-in {NOISE_FLOOR_STANDIN:g})"
    tolerance = 10 * floor_used
    print(f"noise floor: max rel-Frobenius {floor:.3e}{standin_note}")
    print(f"tolerance (10x floor): {tolerance:.3e}")

    # --- Gate 1: per-layer relative Frobenius distance ---
    dist_by_layer = rel_frobenius(my_J, ref_a.jacobians)
    print("\nlayer  rel_frobenius(mine, ref)   ref_noise_floor")
    for layer, dist in dist_by_layer.items():
        flag = "" if dist <= tolerance else "   <-- EXCEEDS TOLERANCE"
        print(f"  L{layer:<4d} {dist:.3e}                {floor_by_layer[layer]:.3e}{flag}")
    worst = max(dist_by_layer.values())
    frobenius_pass = worst <= tolerance
    print(
        f"matrix gate: max {worst:.3e} vs tolerance {tolerance:.3e} -> "
        f"{'PASS' if frobenius_pass else 'FAIL'}"
    )

    # --- Gate 2 (confirmation): top-1 readout agreement on held-out prompts ---
    agree = total = 0
    for prompt in heldout:
        input_ids = subject.encode(prompt, max_length=MAX_SEQ_LEN)
        seq_len = input_ids.shape[1]
        positions = list(range(fitter.SKIP_FIRST, seq_len - 1, POSITION_STRIDE))
        ref_logits, _, _ = ref_a.apply(
            ref_model, prompt, positions=positions, max_seq_len=MAX_SEQ_LEN
        )
        my_logits, _ = fitter.lens_logits(
            subject, my_J, prompt, positions=positions, max_seq_len=MAX_SEQ_LEN
        )
        for layer in sorted(my_J):
            matches = ref_logits[layer].argmax(-1) == my_logits[layer].argmax(-1)
            agree += int(matches.sum())
            total += matches.numel()
    lb, ub = wilson(agree, total)
    readout_pass = lb >= READOUT_WILSON_LB
    print(
        f"readout confirmation: {agree}/{total} top-1 agreement, "
        f"Wilson 95% [{lb:.4f}, {ub:.4f}] vs LB >= {READOUT_WILSON_LB} -> "
        f"{'PASS' if readout_pass else 'FAIL'}"
    )

    verdict = "AGREE" if (frobenius_pass and readout_pass) else "DISAGREE"
    print(
        f"\nVERDICT: {verdict} — matrix gate "
        f"{'PASS' if frobenius_pass else 'FAIL'}, readout {'PASS' if readout_pass else 'FAIL'} "
        f"(N_fit={N_FIT}, N_heldout={N_HELDOUT}, floor={floor:.3e}{standin_note})"
    )
    raise SystemExit(0 if verdict == "AGREE" else 1)


if __name__ == "__main__":
    main()
