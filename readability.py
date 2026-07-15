"""readability.py — grading machinery for the M0 readability gate (torch-only).

Grades the six `lens-eval-*` distributions from the reference repo's data
(`refs/jacobian-lens/data/evaluations/` — data files only; no jlens code is
imported). Conventions extracted verbatim from that directory's README:

- Each item has a `prompt` and `intermediates` (concept words the workspace is
  expected to hold). The **readout position** is a single token position:
  the final prompt token for association / typo / multihop / multilingual /
  order-ops (the `target` field in the last three only marks where the answer
  *would* be generated — the prompt ends right before it), and the **last
  newline token** (end of couplet line 1) for poetry.
- An intermediate's **rank** at a layer is where its token lands in the lens's
  ranked vocabulary readout at that position (rank 1 = the lens's top pick).
  The item-level metric is the min over layers; pass@k counts rank ≤ k.
- **order-ops synonym sets**: each key expands to digit+word forms (numbers)
  or word+symbol forms (operations); rank is the min over the set.

Frozen owned conventions (decided before any result was seen — forking-paths
guard; each is a deviations-table row where it departs from the paper):

- **Token variants**: a word is looked up as both `w` and ` w` (with leading
  space — how words appear mid-text under BPE); rank = min over the variants
  that encode to a *single* token.
- **Single-token pre-filter**: an intermediate with no single-token variant is
  dropped from grading (counted and reported). Qwen digit-splits multi-digit
  numbers, so e.g. `11` drops.
- **Synonym table** (order-ops): numbers get their digit and lowercase-word
  forms; operations get word, symbol, and common spoken forms, fixed below.
- Ranks use "competition" counting: rank = 1 + (number of vocab tokens with
  strictly greater logit). Float logits make exact ties vanishingly rare.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from fitter import SubjectModel, _record_residuals

EVAL_DIR = "refs/jacobian-lens/data/evaluations"
SLUGS = ("association", "multihop", "multilingual", "order-ops", "poetry", "typo")

#: Oracle-drift guard: item counts as extracted 2026-07-15. A refetched clone
#: that disagrees means the oracle moved — INVALID, not a silent re-grade.
EXPECTED_COUNTS = {
    "association": 102,
    "multihop": 93,
    "multilingual": 107,
    "order-ops": 55,
    "poetry": 98,
    "typo": 96,
}

NUMBER_WORDS = {
    "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
    "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve",
    "13": "thirteen", "15": "fifteen", "16": "sixteen", "20": "twenty",
    "24": "twenty-four",
}
OPERATION_FORMS = {
    "addition": ["addition", "+", "plus"],
    "subtraction": ["subtraction", "-", "minus"],
    "multiplication": ["multiplication", "*", "times"],
    "division": ["division", "/", "divided"],
    "mod": ["mod", "%", "modulo"],
    "squared": ["squared", "^", "²"],
}


def load_distribution(slug: str, eval_dir: str = EVAL_DIR) -> list[dict]:
    """Load one distribution's items, guarding against oracle drift."""
    path = os.path.join(eval_dir, f"lens-eval-{slug}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — refetch the reference clone: "
            "git clone https://github.com/anthropics/jacobian-lens refs/jacobian-lens"
        )
    items = json.load(open(path))["items"]
    if len(items) != EXPECTED_COUNTS[slug]:
        raise ValueError(
            f"{slug}: {len(items)} items but {EXPECTED_COUNTS[slug]} expected — "
            "reference data drifted; re-extract before grading"
        )
    return items


def synonym_forms(key: str, slug: str) -> list[str]:
    """The string forms an intermediate key may appear as (before tokenizing)."""
    if slug == "order-ops":
        if key in NUMBER_WORDS:
            return [key, NUMBER_WORDS[key]]
        if key in OPERATION_FORMS:
            return OPERATION_FORMS[key]
    return [key]


def candidate_token_ids(key: str, slug: str, tokenizer) -> list[int]:
    """Single-token ids for an intermediate: each synonym form, bare and with
    a leading space; forms that don't encode to exactly one token drop out.
    Empty list ⇒ the intermediate fails the single-token pre-filter."""
    ids = []
    for form in synonym_forms(key, slug):
        for variant in (form, " " + form):
            enc = tokenizer(variant, add_special_tokens=False).input_ids
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


def readout_position(slug: str, input_ids: torch.Tensor, tokenizer) -> int:
    """The single scored position for this distribution (see module docstring)."""
    seq_len = input_ids.shape[1]
    if slug != "poetry":
        return seq_len - 1
    newline_positions = [
        p for p in range(seq_len)
        if "\n" in tokenizer.decode([int(input_ids[0, p])])
    ]
    if not newline_positions:
        raise ValueError("poetry prompt has no newline token")
    return max(newline_positions)


@dataclass
class GradedIntermediate:
    """Per-layer ranks (1-based; index = fitted-layer index) for one
    intermediate under both arms. min_rank(...) is the pass@k input."""

    key: str
    ranks_jlens: dict[int, int]
    ranks_logitlens: dict[int, int]

    def min_rank(self, arm: str, layers: list[int] | None = None) -> int:
        ranks = self.ranks_jlens if arm == "jlens" else self.ranks_logitlens
        layers = ranks.keys() if layers is None else layers
        return min(ranks[l] for l in layers)


def grade_item(
    subject: SubjectModel,
    jacobians: dict[int, torch.Tensor],
    slug: str,
    item: dict,
    *,
    max_seq_len: int = 128,
) -> tuple[list[GradedIntermediate], list[str]]:
    """Grade one item: one forward pass, both arms, every fitted layer.

    Returns (graded, dropped): per-intermediate 1-based ranks at the readout
    position for the J-lens arm and the logit-lens (J = identity) arm, plus
    the intermediates dropped by the single-token pre-filter.
    """
    layers = sorted(jacobians)
    input_ids = subject.encode(item["prompt"], max_length=max_seq_len)
    position = readout_position(slug, input_ids, subject.tokenizer)

    with torch.no_grad():
        with _record_residuals(subject.layers, layers, graph_root=None) as residuals:
            subject.forward(input_ids)
        # Residual at the readout position, per layer: [d_model]
        h = {l: residuals[l][0, position].float() for l in layers}
        logits = {}
        for l in layers:
            transported = h[l] @ jacobians[l].T.to(h[l].device)
            logits[l] = {
                "jlens": subject.unembed(transported).float(),
                "logitlens": subject.unembed(h[l]).float(),
            }

    graded, dropped = [], []
    for key in item["intermediates"]:
        token_ids = candidate_token_ids(key, slug, subject.tokenizer)
        if not token_ids:
            dropped.append(key)
            continue
        ranks = {"jlens": {}, "logitlens": {}}
        for l in layers:
            for arm in ("jlens", "logitlens"):
                arm_logits = logits[l][arm]
                best = min(
                    int((arm_logits > arm_logits[t]).sum()) for t in token_ids
                )
                ranks[arm][l] = best + 1  # 1-based: rank 1 = top
        graded.append(
            GradedIntermediate(
                key=key, ranks_jlens=ranks["jlens"], ranks_logitlens=ranks["logitlens"]
            )
        )
    return graded, dropped


def proportional_band(n_layers: int) -> list[int]:
    """The frozen D2 band: layer l is in the workspace band iff
    0.38 <= l/(n_layers-1) <= 0.92 (the paper's percentage depth transplanted)."""
    return [
        l for l in range(n_layers) if 0.38 <= l / (n_layers - 1) <= 0.92
    ]
