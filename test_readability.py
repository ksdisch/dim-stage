"""test_readability.py — analytic checks of the readability grading (no downloads).

Same philosophy as test_fitter.py: tiny rigged models where every rank is known
by construction, so the grading conventions (1-based competition rank, min over
candidate tokens, min over layers, readout positions, single-token pre-filter,
frozen band) are verified in milliseconds before any real run.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

import readability

VOCAB = 11
D_MODEL = 4


class CharTokenizer:
    """One token per character (id = ord(c) % VOCAB) — multi-char strings are
    multi-token, so they exercise the single-token pre-filter."""

    def __call__(self, text, add_special_tokens=False):
        class Enc:
            input_ids = [ord(c) % VOCAB for c in text]

        return Enc()

    def decode(self, ids):
        # Reconstruct printable chars; '\n' (10) round-trips exactly.
        return "".join(chr(i) if i == 10 else chr(ord("a") + i) for i in ids)


class ConstantBlock(nn.Module):
    """Ignores its input and emits fixed residuals: position p gets row p of
    `table` — so downstream logits are hand-computable."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = nn.Parameter(table, requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.table[: h.shape[1]].expand(h.shape[0], -1, -1)


class RiggedSubject:
    """Duck-typed SubjectModel whose layer-l residual at every position is a
    chosen constant vector, and whose unembedding is a fixed known matrix."""

    def __init__(self, layer_tables: list[torch.Tensor], unembed: torch.Tensor):
        self.layers = nn.ModuleList(ConstantBlock(t) for t in layer_tables)
        self.n_layers = len(layer_tables)
        self.d_model = D_MODEL
        self.tokenizer = CharTokenizer()
        self._U = unembed  # [VOCAB, D_MODEL]

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        return torch.tensor([self.tokenizer(text).input_ids[:max_length]])

    def forward(self, input_ids: torch.Tensor) -> None:
        h = torch.zeros(input_ids.shape[0], input_ids.shape[1], D_MODEL)
        for block in self.layers:
            h = block(h)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual @ self._U.T


def test_synonym_forms_and_prefilter():
    assert readability.synonym_forms("5", "order-ops") == ["5", "five"]
    assert readability.synonym_forms("multiplication", "order-ops") == [
        "multiplication", "*", "times",
    ]
    assert readability.synonym_forms("Brazil", "multihop") == ["Brazil"]
    tok = CharTokenizer()
    # Single-char forms survive (bare variant is 1 token; " c" is 2); the
    # multi-char word form contributes nothing.
    assert readability.candidate_token_ids("5", "order-ops", tok) == [ord("5") % VOCAB]
    # Multi-char with no single-token form ⇒ pre-filtered out.
    assert readability.candidate_token_ids("Brazil", "multihop", tok) == []


def test_readout_positions():
    tok = CharTokenizer()
    ids = torch.tensor([[3, 1, 10, 4, 10, 5, 6]])  # newlines at 2 and 4
    assert readability.readout_position("poetry", ids, tok) == 4
    assert readability.readout_position("typo", ids, tok) == 6
    with pytest.raises(ValueError, match="no newline"):
        readability.readout_position("poetry", torch.tensor([[1, 2, 3]]), tok)


def test_grade_item_ranks_are_exact():
    # Layer 0 residual = e0 everywhere; layer 1 residual = e1 everywhere.
    # U row v = v * e_{v % D_MODEL}: logits at layer 0 are nonzero exactly for
    # vocab ids v ≡ 0 (mod 4), i.e. ids {0(=0·e0), 4, 8}, scores {0, 4, 8}.
    e = torch.eye(D_MODEL)
    tables = [e[0].repeat(8, 1), e[1].repeat(8, 1)]
    U = torch.stack([float(v) * e[v % D_MODEL] for v in range(VOCAB)])
    subject = RiggedSubject(tables, U)
    identity_J = {0: torch.eye(D_MODEL)}

    # Intermediates are single chars whose vocab ids are 8 and 4: layer-0
    # logits score id 8 highest (8.0), id 4 second (4.0), the rest 0.
    char_for = {i: chr(c) for c in range(32, 127) if (i := c % VOCAB) in (4, 8)}
    item = {"prompt": "x" * 6, "intermediates": [char_for[8], char_for[4]]}

    graded, dropped = readability.grade_item(subject, identity_J, "typo", item)
    assert dropped == []
    by_key = {g.key: g for g in graded}
    assert by_key[char_for[8]].ranks_jlens[0] == 1  # top-1
    assert by_key[char_for[4]].ranks_jlens[0] == 2  # second
    # J = identity ⇒ both arms identical everywhere.
    for g in graded:
        assert g.ranks_jlens == g.ranks_logitlens


def test_min_rank_over_layers_and_band():
    g = readability.GradedIntermediate(
        key="x", ranks_jlens={0: 7, 1: 2, 2: 9}, ranks_logitlens={0: 1, 1: 5, 2: 5}
    )
    assert g.min_rank("jlens") == 2
    assert g.min_rank("logitlens") == 1
    assert g.min_rank("jlens", layers=[0, 2]) == 7


def test_grade_item_min_over_candidate_tokens():
    # An order-ops number key grades as min over its digit/word single-token
    # forms — with the char tokenizer only the digit survives, and a J that
    # boosts a different direction changes the two arms differently.
    e = torch.eye(D_MODEL)
    tables = [e[2].repeat(8, 1)]
    U = torch.stack([float(v + 1) * e[v % D_MODEL] for v in range(VOCAB)])
    subject = RiggedSubject(tables, U)
    # J maps e2 -> e3: the jlens arm scores ids ≡ 3 (mod 4); logit-lens arm
    # scores ids ≡ 2 (mod 4). id of '7' is (ord('7')=55) % 11 = 0.
    J = {0: torch.zeros(D_MODEL, D_MODEL)}
    J[0][3, 2] = 1.0
    item = {"prompt": "y" * 5, "intermediates": ["7"]}
    graded, _ = readability.grade_item(subject, J, "order-ops", item)
    (g,) = graded
    assert g.ranks_jlens[0] != g.ranks_logitlens[0]


def test_proportional_band_matches_frozen_table():
    assert readability.proportional_band(24) == list(range(9, 22))
    assert readability.proportional_band(28) == list(range(11, 25))
    # 3B escalation, pre-registered 2026-07-15: 36 layers -> L14-L32.
    assert readability.proportional_band(36) == list(range(14, 33))


def test_frozen_bands_table_is_the_d2_rule():
    # Every frozen entry must be exactly what the D2 rule computes — the table
    # exists to catch drift, never to override the rule.
    import m0_readability_gate

    for n_layers, band in m0_readability_gate.FROZEN_BANDS.items():
        assert band == readability.proportional_band(n_layers)


def test_expected_counts_guard(tmp_path):
    import json

    bad = tmp_path / "lens-eval-typo.json"
    bad.write_text(json.dumps({"items": [{"prompt": "x", "intermediates": ["y"]}]}))
    with pytest.raises(ValueError, match="drifted"):
        readability.load_distribution("typo", eval_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="refetch"):
        readability.load_distribution("poetry", eval_dir=str(tmp_path))


def test_gate_validate_rejects_wrong_arm():
    import m0_readability_gate as gate

    class FakeSubject:
        d_model = D_MODEL
        n_layers = 3

    class Args:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    artifact = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "d_model": D_MODEL, "J": {}}
    with pytest.raises(SystemExit) as exc:
        gate.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2
