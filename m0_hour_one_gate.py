"""M0 hour-one gate — does the backward pass run on MPS, and how fast?

Frozen spec: docs/M0-BRIEF.md, "Hour-one gate spec". Runs the reference
implementation's per-prompt Jacobian estimator on Qwen2.5-0.5B-Instruct
(fp32, MPS) for 2 WikiText prompts, then extrapolates a full N=100 fit for
both subjects. PASS iff the extrapolated total is <= 12 h.

The reference jlens is used here as a hardware probe only (cross-check
oracle pattern); measurement harness code never imports it.

Run: uv run --group crosscheck m0_hour_one_gate.py
"""

import math
import time

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts
from jlens.fitting import jacobian_for_prompt

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FIT_N = 100          # frozen D3 corpus size
BUDGET_HOURS = 12.0  # frozen PASS threshold: both subjects' fits, combined
DIM_BATCHES = (8, 32)  # 8 = reference default; 32 = math-neutral batching probe

# Extrapolation to the second subject (Qwen2.5-1.5B-Instruct): per-pass cost
# scales ~ with parameter count, pass count with ceil(d_model / dim_batch).
PARAMS_05B, PARAMS_15B = 0.494e9, 1.544e9
D_MODEL_05B, D_MODEL_15B = 896, 1536


def main() -> None:
    if not torch.backends.mps.is_available():
        print("VERDICT: FAIL — MPS not available in this torch build")
        raise SystemExit(1)

    print(f"torch {torch.__version__}, transformers {transformers.__version__}")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to("mps")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf, tok)
    assert model.d_model == D_MODEL_05B, f"unexpected d_model={model.d_model}"
    print(f"{MODEL_ID}: n_layers={model.n_layers}, d_model={model.d_model}")

    # The frozen D3 corpus convention: first records >= 600 chars, in order.
    prompts = load_wikitext_prompts(2)
    source_layers = list(range(model.n_layers - 1))

    best: tuple[float, int] | None = None  # (warm per-prompt seconds, dim_batch)
    for dim_batch in DIM_BATCHES:
        n_passes = math.ceil(model.d_model / dim_batch)
        for i, prompt in enumerate(prompts):
            start = time.perf_counter()
            _, seq_len, n_valid = jacobian_for_prompt(
                model, prompt, source_layers, dim_batch=dim_batch
            )
            elapsed = time.perf_counter() - start
            print(
                f"dim_batch={dim_batch} ({n_passes} backwards): prompt {i} "
                f"seq_len={seq_len} n_valid={n_valid} -> {elapsed:.1f}s"
            )
        # Prompt 1 is past MPS warmup: the honest steady-state rate.
        if best is None or elapsed < best[0]:
            best = (elapsed, dim_batch)

    warm, dim_batch = best
    pass_ratio = math.ceil(D_MODEL_15B / dim_batch) / math.ceil(D_MODEL_05B / dim_batch)
    per_prompt_15b = warm * (PARAMS_15B / PARAMS_05B) * pass_ratio
    hours_05b = warm * FIT_N / 3600
    hours_15b = per_prompt_15b * FIT_N / 3600
    total_hours = hours_05b + hours_15b

    print(f"\nbest config: dim_batch={dim_batch}, {warm:.1f}s/prompt (0.5B, measured)")
    print(f"1.5B extrapolation (FLOPs ratio): {per_prompt_15b:.1f}s/prompt")
    print(
        f"N={FIT_N} fit: 0.5B {hours_05b:.2f}h (measured rate) + "
        f"1.5B {hours_15b:.2f}h (extrapolated) = {total_hours:.2f}h"
    )
    verdict = "PASS" if total_hours <= BUDGET_HOURS else "FAIL"
    print(f"VERDICT: {verdict} — rule: combined fits <= {BUDGET_HOURS:.0f}h")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
