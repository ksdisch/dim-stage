"""test_two_hop.py — analytic checks of the M2 trial rules (no downloads)."""
from __future__ import annotations

import json

import pytest

import m2_two_hop as th
from test_verbal_report import VocabTokenizer


def test_load_items_guards_against_drift(tmp_path):
    bad = tmp_path / "probe-swap.json"
    bad.write_text(json.dumps({"items": [{"prompt": "x"}]}))
    with pytest.raises(ValueError, match="drifted"):
        th.load_items(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        th.load_items(str(tmp_path / "missing.json"))


def test_plan_item_requires_single_token_forms_for_all_four_fields():
    tok = VocabTokenizer({"Brazil": 1, " Brazil": 2, "Mexico": 3, "Portuguese": 4})
    item = {
        "intermediate": "Brazil",
        "swap_to": "Mexico",
        "answer": "Portuguese",
        "swap_answer": "Spanish",  # absent from the vocab -> two tokens
    }
    forms, missing = th.plan_item(item, tok)
    assert missing == ["swap_answer"]  # this item would be dropped and counted
    assert forms["intermediate"] == [1, 2]  # bare form first (the swap token)
    assert forms["swap_to"] == [3]

    tok2 = VocabTokenizer({"Brazil": 1, "Mexico": 3, "Portuguese": 4, "Spanish": 5})
    forms2, missing2 = th.plan_item(item, tok2)
    assert missing2 == []
    assert forms2["swap_answer"] == [5]


def test_validate_rejects_a_mismatched_lens():
    class FakeSubject:
        d_model = 4
        n_layers = 24

    class Args:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    artifact = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "d_model": 4, "J": {}}
    with pytest.raises(SystemExit) as exc:
        th.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2
