"""fitter.py — the independent Jacobian-lens fitter (torch-only; never imports jlens).

This is dim-stage's own implementation of the fitting procedure extracted in
docs/M0-BRIEF.md ("Design extraction"). It is validated against the reference
implementation only through the AGREE gate (m0_agree_gate.py) — the lossy-wall
pattern: the reference is an oracle we compare against, never a library we call.

What the lens is, in plain terms: the **residual stream** is the running vector a
transformer carries from layer to layer (one vector per token position). The
**unembedding** is the model's final matrix, turning the last-layer vector into one
score (**logit**) per vocabulary token. The **J-lens** reads a *middle* layer by first
multiplying its vector by `J_l` — the average **Jacobian** (matrix of sensitivities:
how much each final-layer coordinate moves per nudge of each layer-l coordinate) —
which transports it into the final layer's coordinate system before unembedding:

    lens_l(h) = unembed( J_l @ h ),    J_l = E[ d h_final / d h_l ]

How `J_l` is estimated (the reduction used in the paper, per the reference spec):
for each output dimension, inject a **one-hot cotangent** — backpropagation's
"direction of interest": a 1 in that single output coordinate — at *every valid
target position at once*, and backprop (a **VJP**, vector–Jacobian product: one
backward pass computes cotangent-times-Jacobian). The gradient landing at source
position p is then `sum over p' >= p of d h_final[p'] / d h_l[p]` (only later
positions can depend on p — the model is causal); the mean over source positions p
gives one row of `J_l`. Batching: the prompt is replicated `dim_batch` times so each
backward pass computes `dim_batch` rows at once — a math-neutral knob (frozen at 8
on MPS; see M0-BRIEF, "dim_batch=32 is ~25x slower").

Frozen parameters (M0-BRIEF "Frozen parameters" table): skip_first=16 (early
positions are attention sinks), final position excluded (no next-token target),
max_seq_len=128, target = final layer, J accumulated fp32 as a running mean over
prompts.

CLI (the N=100 production fit, D3 corpus convention):

    uv run python -u fitter.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
        --n-prompts 100 --out lenses/qwen2.5-0.5b-instruct-n100.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Sequence
from contextlib import contextmanager

import torch
from torch import nn

SKIP_FIRST = 16  # leading positions excluded from the Jacobian average


class SubjectModel:
    """A HuggingFace causal LM wrapped down to what the fitter needs.

    Expects the modern Llama/Qwen layout: `hf.model.layers` (the residual blocks),
    `hf.model.norm` (final pre-unembed norm), `hf.model.embed_tokens`, `hf.lm_head`.
    Both dim-stage subjects (Qwen2.5-0.5B/1.5B-Instruct) use it; anything else
    fails loudly rather than guessing.

    Mutates the model in place: eval mode, every parameter frozen
    (`requires_grad=False`) — the fit differentiates with respect to *activations*,
    never weights, and frozen weights let one captured activation root the whole
    autograd graph (see `_record_residuals`).
    """

    def __init__(self, hf_model: nn.Module, tokenizer) -> None:
        decoder = getattr(hf_model, "model", None)
        for attr in ("layers", "norm", "embed_tokens"):
            if decoder is None or not hasattr(decoder, attr):
                raise ValueError(
                    f"{type(hf_model).__name__} lacks model.{attr}; "
                    "SubjectModel only supports the Llama/Qwen layout"
                )
        if not hasattr(hf_model, "lm_head"):
            raise ValueError(f"{type(hf_model).__name__} lacks lm_head")
        softcap = getattr(hf_model.config.get_text_config(), "final_logit_softcapping", None)
        if softcap is not None:
            raise ValueError("logit softcapping not implemented (not a Qwen2.5 feature)")

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)
        # Reference parity: instruction-tuned checkpoints sometimes ship
        # add_bos_token=False; the reference forces it on when a BOS exists.
        # (Qwen2.5 has no BOS token, so this is a recorded no-op for our subjects.)
        if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(
            tokenizer, "add_bos_token"
        ):
            tokenizer.add_bos_token = True

        self._decoder = decoder
        self.tokenizer = tokenizer
        self.layers: nn.ModuleList = decoder.layers
        self._final_norm = decoder.norm
        self._lm_head = hf_model.lm_head
        self._input_device = decoder.embed_tokens.weight.device
        text_config = hf_model.config.get_text_config()
        self.n_layers: int = text_config.num_hidden_layers
        self.d_model: int = text_config.hidden_size

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        """Tokenize to input_ids [1, seq_len] on the model's input device."""
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self._input_device)

    def forward(self, input_ids: torch.Tensor) -> None:
        """Run the residual stack only (no unembedding — the fit never needs logits)."""
        self._decoder(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Residual [..., d_model] -> logits [..., vocab]: final norm + LM head."""
        weight = self._lm_head.weight
        return self._lm_head(self._final_norm(residual.to(weight.dtype).to(weight.device)))


def valid_position_mask(seq_len: int, *, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    """Boolean [seq_len] mask of positions included in the Jacobian average.

    The first `skip_first` positions are excluded (attention sinks with atypical
    residual statistics) and so is the final position (no next-token target).
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if not mask.any():
        raise ValueError(f"prompt too short: seq_len={seq_len} <= skip_first+1={skip_first + 1}")
    return mask


@contextmanager
def _record_residuals(layers: Sequence[nn.Module], at: Sequence[int], *, graph_root: int | None):
    """Capture each listed block's output residual on the next forward pass.

    Yields a dict that fills with {block index: residual tensor} as the forward
    runs. Tensors are NOT detached, so torch.autograd.grad can differentiate
    through them. If `graph_root` is given, that block's output is flipped to
    requires_grad=True as it is produced — with all weights frozen it is the
    only graph leaf, so autograd retains the graph from that block onward only
    (the memory trick that makes MPS fitting feasible).
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        def hook(module, inputs, output):
            residual = output if torch.is_tensor(output) else output[0]
            if index == graph_root:
                residual.requires_grad_(True)
            captured[index] = residual

        return hook

    indices = sorted(set(at) | ({graph_root} if graph_root is not None else set()))
    try:
        for index in indices:
            handles.append(layers[index].register_forward_hook(make_hook(index)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def jacobian_for_prompt(
    subject: SubjectModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """The per-prompt estimator: one forward, ceil(d_model/dim_batch) backwards.

    Returns (jacobians, seq_len, n_valid): jacobians maps each source layer to a
    [d_model, d_model] fp32 CPU tensor whose row r is the mean-over-source-positions,
    sum-over-later-target-positions gradient of final-layer coordinate r.
    """
    if target_layer is None:
        target_layer = subject.n_layers - 1
    source_layers = sorted(set(source_layers))
    if not source_layers or source_layers[0] < 0 or source_layers[-1] >= target_layer:
        raise ValueError(
            f"source_layers must be within [0, {target_layer - 1}]; got {source_layers}"
        )
    d_model = subject.d_model

    input_ids = subject.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid = int(mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        _record_residuals(
            subject.layers, [*source_layers, target_layer], graph_root=source_layers[0]
        ) as residuals,
        torch.enable_grad(),
    ):
        subject.forward(input_ids.expand(dim_batch, -1))
        target = residuals[target_layer]  # [dim_batch, seq_len, d_model]
        sources = [residuals[layer] for layer in source_layers]

        valid_positions = mask.nonzero(as_tuple=True)[0].to(target.device)
        batch_indices = torch.arange(dim_batch, device=target.device)
        cotangent = torch.zeros_like(target)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            # Batch element b asks about output dimension dim_start+b, injected
            # at every valid target position at once.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target,
                inputs=sources,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                rows = grad[:n_dims, valid_positions, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid


def fit(
    subject: SubjectModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST,
    checkpoint_path: str | None = None,
    log=print,
) -> dict[int, torch.Tensor]:
    """Fit J_l over prompts: the running mean of per-prompt estimates.

    Prompts too short to have any valid position are skipped, matching the
    reference. If `checkpoint_path` is set, the running sum is written after every
    prompt (atomically) and resumed from on restart — the N=100 MPS fit is an
    hour-plus, and a crash should not restart it from zero.

    Returns {layer: J_l} as fp32 CPU tensors.
    """
    if target_layer is None:
        target_layer = subject.n_layers - 1
    if source_layers is None:
        source_layers = list(range(target_layer))
    source_layers = sorted(set(source_layers))

    jacobian_sum = {
        layer: torch.zeros(subject.d_model, subject.d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_done, next_idx = 0, 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state["source_layers"] != source_layers or state["skip_first"] != skip_first:
            raise ValueError(f"checkpoint {checkpoint_path} was fitted with other settings")
        jacobian_sum, n_done, next_idx = state["jacobian_sum"], state["n_done"], state["next_idx"]
        log(f"resuming from {checkpoint_path}: {next_idx}/{len(prompts)} prompts done")

    def write_checkpoint() -> None:
        if checkpoint_path:
            tmp = f"{checkpoint_path}.tmp"
            torch.save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "skip_first": skip_first,
                },
                tmp,
            )
            os.replace(tmp, checkpoint_path)

    for idx, prompt in enumerate(prompts):
        if idx < next_idx:
            continue
        start = time.perf_counter()
        try:
            per_prompt, seq_len, n_valid = jacobian_for_prompt(
                subject,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            log(f"prompt {idx + 1}/{len(prompts)}: skipped ({exc})")
            next_idx = idx + 1
            continue
        for layer in source_layers:
            jacobian_sum[layer] += per_prompt[layer]
        n_done += 1
        next_idx = idx + 1
        log(
            f"prompt {idx + 1}/{len(prompts)}: seq_len={seq_len} n_valid={n_valid} "
            f"{time.perf_counter() - start:.1f}s"
        )
        write_checkpoint()

    if n_done == 0:
        raise ValueError("no prompt was long enough to fit on")
    return {layer: jacobian_sum[layer] / n_done for layer in source_layers}


def lens_logits(
    subject: SubjectModel,
    jacobians: dict[int, torch.Tensor],
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 128,
    use_jacobian: bool = True,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Read the lens out: {layer: [n_positions, vocab] logits}, plus input_ids.

    One forward pass; each requested layer's residual at the requested positions
    is transported with J_l (skipped when `use_jacobian=False` — that is the
    logit-lens baseline, literally J = identity) and unembedded.
    """
    if layers is None:
        layers = sorted(jacobians)
    with torch.no_grad():
        input_ids = subject.encode(prompt, max_length=max_seq_len)
        with _record_residuals(subject.layers, list(layers), graph_root=None) as residuals:
            subject.forward(input_ids)
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = residuals[layer][0]  # [seq_len, d_model]
            if positions is not None:
                residual = residual[list(positions)]
            residual = residual.float()
            if use_jacobian:
                residual = residual @ jacobians[layer].T.to(residual.device)
            out[layer] = subject.unembed(residual).float().cpu()
    return out, input_ids


def load_wikitext_prompts(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """The frozen D3 corpus convention, reimplemented so harness code never
    imports jlens: the first `n_prompts` WikiText-103 train records whose
    stripped text is >= `min_chars` characters, streamed in order (deterministic,
    no seed). Records are returned unstripped, matching the reference."""
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    for record in dataset:
        if len(record["text"].strip()) >= min_chars:
            prompts.append(record["text"])
            if len(prompts) == n_prompts:
                break
    return prompts


def main() -> None:
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--out", required=True, help="lens artifact path (*.pt, gitignored)")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list[str] holding the D3 prompts (same records the WikiText "
        "streaming loader yields); bypasses HuggingFace streaming, which can wedge "
        "in rate-limit backoff on long days. Transport only — never a different corpus.",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {args.model_id} (fp32, {device})")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.float32
    ).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model_id)
    subject = SubjectModel(hf, tok)
    print(f"n_layers={subject.n_layers} d_model={subject.d_model}")

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        if len(prompts) != args.n_prompts:
            raise ValueError(
                f"{args.prompts_file} holds {len(prompts)} prompts, not {args.n_prompts}"
            )
    else:
        prompts = load_wikitext_prompts(args.n_prompts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jacobians = fit(
        subject, prompts, dim_batch=args.dim_batch, checkpoint_path=f"{args.out}.ckpt"
    )
    torch.save(
        {
            "J": jacobians,
            "n_prompts": args.n_prompts,
            "d_model": subject.d_model,
            "model_id": args.model_id,
            "corpus": f"wikitext-103 first {args.n_prompts} records >=600 chars (D3)",
        },
        args.out,
    )
    print(f"saved lens to {args.out}")


if __name__ == "__main__":
    main()
