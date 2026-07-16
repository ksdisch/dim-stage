"""test_verbal_report.py — analytic checks of the D5 trial rules (no downloads).

Same philosophy as test_readability.py: the conventions that decide which swaps
count (first-10 truncation, spontaneous-answer skip, single-token filter,
baseline rank ≥ 11 exclusion, bare-form swap token) are verified on rigged
tokenizers in milliseconds before any real run.
"""
from __future__ import annotations

import pytest
import torch

import m1_verbal_report as vr
from intervention import swap


class VocabTokenizer:
    """Fixed string→id table; anything absent encodes to two tokens (and so
    fails the single-token filter)."""

    def __init__(self, table: dict[str, int]):
        self.table = table

    def __call__(self, text, add_special_tokens=False):
        ids = [self.table[text]] if text in self.table else [0, 0]

        class Enc:
            input_ids = ids

        return Enc()


def test_token_forms_orders_bare_form_first():
    tok = VocabTokenizer({"Blue": 7, " Blue": 8, " Red": 9})
    assert vr.token_forms("Blue", tok) == [7, 8]  # bare first — the swap token
    assert vr.token_forms("Red", tok) == [9]  # only the space form exists
    assert vr.token_forms("Camel", tok) == []  # no single-token form


def test_word_rank_is_competition_min_over_forms():
    logits = torch.tensor([0.0, 5.0, 3.0, 5.0, 1.0])
    assert vr.word_rank(logits, [2]) == 3  # two tokens strictly greater
    assert vr.word_rank(logits, [2, 1]) == 1  # min over forms
    assert vr.word_rank(logits, [1, 3]) == 1  # ties don't inflate the rank


def test_plan_category_applies_every_d5_rule():
    words = [f"w{i}" for i in range(12)]  # 12 listed — only the first 10 count
    forms = {w: [100 + i] for i, w in enumerate(words[:10])}
    forms["w1"] = []  # no single-token form
    forms["w2"] = [42, 142]  # the spontaneous answer's token among its forms
    ranks = {w: 50 for w in words[:10]}
    ranks["w1"] = None
    ranks["w3"] = 10  # already top-10 at baseline
    ranks["w4"] = 11  # exactly at the floor — eligible

    trials = vr.plan_category(words, answer_token=42, forms_by_word=forms, rank_by_word=ranks)
    assert len(trials) == 10  # w10, w11 never considered
    by_word = {t.word: t for t in trials}
    assert by_word["w1"].status == "no_single_token"
    assert by_word["w2"].status == "spontaneous"
    assert by_word["w3"].status == "baseline_top10"
    assert by_word["w4"].status == "eligible" and by_word["w4"].baseline_rank == 11
    assert by_word["w0"].status == "eligible"
    assert by_word["w0"].token == 100  # the bare form, forms[0]
    assert sum(t.status == "eligible" for t in trials) == 7


def test_load_candidates_guards_against_drift(tmp_path):
    import json

    bad = tmp_path / "verbal-report.json"
    bad.write_text(json.dumps({"candidates": {"color": ["Blue"]}}))
    with pytest.raises(ValueError, match="drifted"):
        vr.load_candidates(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        vr.load_candidates(str(tmp_path / "missing.json"))


def test_validate_rejects_a_tokenizer_without_chat_template():
    class NoTemplateTokenizer:
        chat_template = None

    class FakeSubject:
        d_model = 4
        n_layers = 24  # so the frozen-band check passes and we reach the new check
        tokenizer = NoTemplateTokenizer()

    class Args:
        model_id = "m"

    artifact = {"model_id": "m", "d_model": 4, "J": {i: None for i in range(23)}}
    with pytest.raises(SystemExit) as exc:
        vr.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2


def test_checked_swap_edits_is_a_transparent_wrapper():
    gen = torch.Generator().manual_seed(1)
    h = torch.randn(1, 3, 8, generator=gen)
    v_s = torch.randn(8, generator=gen)
    v_t = torch.randn(8, generator=gen)
    edits = vr.checked_swap_edits({0: (v_s, v_t)})
    assert torch.equal(edits[0](h), swap(h, v_s, v_t))  # check passed silently
