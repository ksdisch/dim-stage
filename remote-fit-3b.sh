#!/usr/bin/env bash
# Remote 3B lens fit — dim-stage's pre-declared rented-GPU fallback
# (KICKOFF amended bar entry 2; decision recorded in DECISIONS.md 2026-07-15).
#
# Why this exists: the 24 GB Mac cannot fit Qwen2.5-3B in fp32 — the 12.4 GB
# of weights sit at the Metal working-set edge and the backward pass lands in
# the ~25x memory-pressure cliff at any dim_batch. CUDA has no such cliff.
#
# Procedure is byte-identical to the local fits: same fitter.py, same frozen
# D3 prompts (wikitext-n100-prompts.json), dim_batch=8, fp32. Only the device
# changes — an owned deviation row, never a silent move.
#
# HOW TO RUN (once, on a rented CUDA box):
#   1) Rent any single-GPU instance with a PyTorch image (RunPod / Lambda /
#      Vast; an RTX 4090 or A10 at ~$0.3-0.8/h is plenty — expect ~1-2 h).
#   2) From the laptop, copy the three files up:
#        scp fitter.py wikitext-n100-prompts.json remote-fit-3b.sh <user>@<box>:~/
#   3) On the box:
#        bash remote-fit-3b.sh
#   4) When it finishes, pull the lens back — run ON THE LAPTOP, from the
#      repo root, so the queued readability gate fires automatically:
#        scp <user>@<box>:~/lenses/qwen2.5-3b-instruct-n100.pt lenses/
#   5) Shut the instance down. Keep fit-3b-cuda.log if you want the per-prompt
#      record (scp it back too; *.log is gitignored).
set -euo pipefail

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device — pick a GPU instance/image"
print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
EOF

pip install --quiet "transformers>=5.13,<6"

python -u fitter.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --n-prompts 100 \
  --prompts-file wikitext-n100-prompts.json \
  --out lenses/qwen2.5-3b-instruct-n100.pt 2>&1 | tee fit-3b-cuda.log

echo "DONE — now scp lenses/qwen2.5-3b-instruct-n100.pt back to the laptop's lenses/ dir"
