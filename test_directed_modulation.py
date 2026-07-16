"""test_directed_modulation.py — analytic checks of the M3 conventions (no downloads).

The deterministic grid, the prompt-span math, the drift guard, and the
top-1-anywhere grading are each verified on rigged inputs before any real run.
"""
from __future__ import annotations

import json

import pytest
import torch

import m3_directed_modulation as dm
from test_introspection import ConcatChatTokenizer, FakeSubject
from test_readability import RiggedSubject

D_MODEL = 4
VOCAB = 11


def test_load_experiment_guards_against_drift(tmp_path):
    bad = tmp_path / "directed-modulation.json"
    bad.write_text(
        json.dumps(
            {
                "phrasings": [{"name": "p", "group": "focus", "text": "{x}"}],
                "group_kind": {"focus": "focus"},
                "carrier_sentences": ["c"],
                "math_problems": [],
                "topic_categories": [],
            }
        )
    )
    with pytest.raises(ValueError, match="drifted"):
        dm.load_experiment(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        dm.load_experiment(str(tmp_path / "missing.json"))


def test_build_grid_is_the_frozen_deterministic_pairing():
    trials, baselines = dm.build_grid(n_phrasings=3, n_targets=4, n_carriers=2)
    assert len(trials) == 12 and len(baselines) == 4
    assert trials[0] == (0, 0, 0)
    assert trials[1] == (0, 1, 1)  # carrier = (i + j) mod C
    assert trials[5] == (1, 1, 0)
    assert baselines[3] == (None, 3, 1)  # carrier = j mod C
    # Deterministic: same inputs, same grid.
    assert dm.build_grid(3, 4, 2) == (trials, baselines)


def test_build_targets_fills_x_and_tracks_the_right_strings():
    data = {
        "math_problems": [{"expr": "4 * 2", "answer": "8", "tier": 1}],
        "topic_categories": [{"name": "citrus fruits", "members": ["orange", "lime"]}],
    }
    math, cat = dm.build_targets(data)
    assert math == {"family": "math", "name": "4 * 2", "x": "4 * 2", "tracked": ["8"]}
    assert cat["x"] == "citrus fruits" and cat["tracked"] == ["orange", "lime"]


def test_build_prompt_span_covers_exactly_the_carrier_tokens():
    subject = FakeSubject(ConcatChatTokenizer())
    input_ids, span = dm.build_prompt(subject, "user text", "carrier")
    tok = ConcatChatTokenizer()
    prefix = tok.apply_chat_template(
        [{"role": "user", "content": "user text"}], add_generation_prompt=True, return_dict=False
    )
    carrier_ids = tok("carrier").input_ids
    assert input_ids.shape == (1, len(prefix) + len(carrier_ids))
    assert span == list(range(len(prefix), len(prefix) + len(carrier_ids)))
    assert input_ids[0, span].tolist() == carrier_ids  # the readout span IS the carrier


def test_trial_hits_grades_top1_anywhere_per_arm():
    # Band residual = 3e0 at every position; J maps e0 -> e2. Under the J-lens
    # the top-1 token is the e2 reader (id 7); under the logit lens it is the
    # e0 reader (id 4). The two arms must disagree exactly as predicted.
    e = torch.eye(D_MODEL)
    U = torch.zeros(VOCAB, D_MODEL)
    U[7] = e[2]
    U[4] = 2.0 * e[0]
    subject = RiggedSubject([3.0 * e[0].repeat(8, 1)], U)
    J = torch.zeros(D_MODEL, D_MODEL)
    J[2, 0] = 1.0
    input_ids = torch.tensor([[0, 1, 2, 3]])
    span = [1, 2]

    hits = dm.trial_hits(subject, {0: J}, [0], input_ids, span, tracked={7})
    assert hits == {"jlens": True, "logitlens": False}
    hits = dm.trial_hits(subject, {0: J}, [0], input_ids, span, tracked={4})
    assert hits == {"jlens": False, "logitlens": True}
    hits = dm.trial_hits(subject, {0: J}, [0], input_ids, span, tracked={9})
    assert hits == {"jlens": False, "logitlens": False}
