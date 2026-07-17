"""test_generalization.py — analytic checks of the S2 trial rules (no downloads)."""
from __future__ import annotations

import json

import pytest
import torch

import s2_generalization as gen
from intervention import swap
from test_verbal_report import VocabTokenizer


def test_load_categories_guards_against_drift(tmp_path):
    bad = tmp_path / "flexible-generalization.json"
    bad.write_text(json.dumps({"categories": [{"name": "x", "args": [], "funcs": []}]}))
    with pytest.raises(ValueError, match="drifted"):
        gen.load_categories(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        gen.load_categories(str(tmp_path / "missing.json"))


def test_variant_forms_keeps_variants_separate():
    tok = VocabTokenizer({"France": 2, " France": 1, " China": 3})
    assert gen.variant_forms("France", tok) == {"bare": 2, "space": 1}
    assert gen.variant_forms("China", tok) == {"space": 3}
    assert gen.variant_forms("Camel", tok) == {}


def test_prompt_form_prefers_the_space_variant_and_reports_position():
    assert gen.prompt_form([9, 1, 8], {"bare": 2, "space": 1}) == ("space", 1, 1)
    assert gen.prompt_form([2, 7], {"bare": 2}) == ("bare", 2, 0)
    assert gen.prompt_form([7, 8], {"bare": 2}) is None


def test_plan_applies_the_filter_and_the_prompt_form_convention():
    # One 2-arg category: France→China's answer (Beijing) has no single-token
    # form -> filtered and counted (D19); China→France is gradable, with the
    # swap rows taken from the prompt-position (space) variants.
    cats = [
        {
            "name": "countries",
            "args": ["France", "China"],
            "funcs": [
                {
                    "name": "capital",
                    "template": "The capital of {arg} is",
                    "answers": {"France": "Paris", "China": "Beijing"},
                }
            ],
        }
    ]
    tok = VocabTokenizer({" France": 1, "France": 2, " China": 3, " Paris": 5})
    prompts = {
        "The capital of France is": [9, 1, 8],
        "The capital of China is": [9, 3, 8],
    }
    diagonals, trials, filtered = gen.plan_trials(cats, tok, prompts.__getitem__)

    assert [d["arg"] for d in diagonals] == ["France", "China"]
    assert diagonals[0]["arg_token"] == 1 and diagonals[0]["arg_pos"] == 1
    assert diagonals[0]["answer_forms"] == [5]
    assert diagonals[1]["answer_forms"] == []  # Beijing — ungradable diagonal

    assert len(filtered) == 1 and len(trials) == 1
    assert filtered[0]["reason"] == "answer_no_single_token"
    assert (filtered[0]["source"], filtered[0]["target"]) == ("France", "China")

    trial = trials[0]
    assert (trial["source"], trial["target"]) == ("China", "France")
    assert trial["u_s_token"] == 3  # " China" as it sits in the prompt
    assert trial["u_t_token"] == 1  # " France" — the matching space variant
    assert trial["variant"] == "space" and trial["fallback"] is False
    assert trial["answer_forms"] == [5]
    assert trial["source_answer_forms"] == []  # Beijing -> displaced = None


def test_degeneracy_guard_flags_attractor_share():
    flagged = gen.degeneracy([1, 1, 1, 2])
    assert flagged["share"] == 0.75 and flagged["collapsed"]
    ok = gen.degeneracy([1, 2, 3, 4])
    assert ok["share"] == 0.25 and not ok["collapsed"]
    assert not gen.degeneracy([])["collapsed"]


def test_checked_alpha_swap_edits_apply_alpha_and_read_back():
    # The generalized D6 read-back must accept its own patch at every α (it
    # exits INVALID otherwise) and the patch must equal the α-swap exactly.
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 3, 8, generator=g, dtype=torch.float64)
    v_s = torch.randn(8, generator=g, dtype=torch.float64)
    v_t = torch.randn(8, generator=g, dtype=torch.float64)
    for alpha in (1.0, 2.0, 8.0):
        edits = gen.checked_alpha_swap_edits({0: (v_s, v_t)}, alpha)
        assert torch.allclose(edits[0](h), swap(h, v_s, v_t, alpha), atol=1e-12)


def test_validate_rejects_a_mismatched_lens():
    class FakeSubject:
        d_model = 4
        n_layers = 24

    class Args:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    artifact = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "d_model": 4, "J": {}}
    with pytest.raises(SystemExit) as exc:
        gen.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2
