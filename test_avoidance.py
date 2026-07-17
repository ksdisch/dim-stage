"""test_avoidance.py — the S4 pre-committed gates (no downloads).

The item set is constructed (D27c — the reference ships nothing), so its
frozen construction rules are enforced here as code: shape, no clue leakage,
distinct controls. The k = 1 ablation edit is proven on an analytic rig and
its read-back proven to fire INVALID on a rigged operator.
"""
from __future__ import annotations

import json

import pytest
import torch

import s4_avoidance as s4


def test_load_items_guards_shape_and_leakage(tmp_path):
    bad = tmp_path / "items.json"
    bad.write_text(json.dumps({"items": [{"name": "x"}]}))
    with pytest.raises(ValueError, match="drifted"):
        s4.load_items(str(bad))

    leaky = {
        "items": [
            {
                "name": f"i{k}", "category": "c", "noun": "thing",
                "concept": "France", "control": "Canada",
                "clue": "A sentence about France." if k == 0 else "A clean clue.",
            }
            for k in range(60)
        ]
    }
    bad.write_text(json.dumps(leaky))
    with pytest.raises(ValueError, match="contains 'France'"):
        s4.load_items(str(bad))

    real = s4.load_items()  # the frozen file passes its own guard
    assert len(real) == 60
    assert len({i["concept"] for i in real}) == 20


def test_sub_band_thirds_partition_the_band_in_order():
    band = list(range(9, 22))  # 0.5B: 13 layers
    thirds = s4.sub_band_thirds(band)
    assert thirds["early"] == band[:4]
    assert thirds["middle"] == band[4:8]
    assert thirds["late"] == band[8:]
    assert thirds["early"] + thirds["middle"] + thirds["late"] == band


def test_concept_ablation_edit_zeroes_exactly_the_concept_coordinate():
    # Identity Jacobian + basis-vector unembedding row: the edit must zero
    # coordinate 1 at every position and leave every other coordinate exact.
    d = 8
    u = torch.zeros(d)
    u[1] = 1.0
    edits = s4.concept_ablation_edits({0: torch.eye(d)}, [0], u)
    h = torch.randn(1, 5, d, generator=torch.Generator().manual_seed(0))
    out = edits[0](h.clone())
    assert torch.allclose(out[0, :, 1], torch.zeros(5), atol=1e-6)
    keep = [i for i in range(d) if i != 1]
    assert torch.allclose(out[0][:, keep], h[0][:, keep], atol=1e-6)


def test_concept_ablation_readback_fires_invalid_on_a_rigged_operator(monkeypatch):
    monkeypatch.setattr(s4, "ablate", lambda h, v: h + 1.0)  # broken operator
    d = 8
    u = torch.zeros(d)
    u[1] = 1.0
    edits = s4.concept_ablation_edits({0: torch.eye(d)}, [0], u)
    h = torch.randn(1, 3, d, generator=torch.Generator().manual_seed(1))
    with pytest.raises(SystemExit) as exc:
        edits[0](h)
    assert exc.value.code == 2


def test_concept_mass_sums_softmax_over_forms():
    logits = torch.log(torch.tensor([0.5, 0.25, 0.125, 0.125]))
    assert s4.concept_mass(logits, [1, 2]) == pytest.approx(0.375, abs=1e-6)


def test_validate_rejects_a_mismatched_lens():
    class FakeSubject:
        d_model = 4
        n_layers = 24

    class Args:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    artifact = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "d_model": 4, "J": {}}
    with pytest.raises(SystemExit) as exc:
        s4.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2
