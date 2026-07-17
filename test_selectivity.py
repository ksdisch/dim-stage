"""test_selectivity.py — the S3 pre-committed gates (no downloads).

The reference ships no ablation code, so — exactly as with M1's operators —
these invariants stand in for the AGREE diff (D24): the projection-removal
operator is proven against analytic constructions where the right answer is
known exactly, the selection rule is proven to honor the clean-top-10
exclusion, and the runtime read-back is proven to fire INVALID on a rigged
operator. Wrong-arm inputs must exit INVALID before any trial runs.
"""
from __future__ import annotations

import json

import pytest
import torch

import s3_selectivity as s3
from intervention import ablate
from test_verbal_report import VocabTokenizer


def _directions(k: int, d: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(k, d, generator=gen, dtype=torch.float64)


# --- the operator: exact span-projection removal (D24 invariants) --------------


def test_ablate_recovers_the_orthogonal_component_exactly():
    # h = (known combination of the directions) + w, with w built orthogonal
    # to every direction: ablation must return w — the analytic oracle.
    v = _directions(3, 32)
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(32, generator=gen, dtype=torch.float64)
    coeffs = torch.linalg.lstsq(v.T, w.unsqueeze(-1)).solution
    w = w - (v.T @ coeffs).squeeze(-1)  # now w ⊥ span{v}
    h = w + v.T @ torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64)
    assert torch.allclose(ablate(h, v), w, atol=1e-9)


def test_ablate_zeroes_every_selected_coordinate():
    v = _directions(4, 16)  # non-orthogonal by construction
    h = torch.randn(16, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    out = ablate(h, v)
    assert float((v @ out).abs().max()) < 1e-9


def test_ablate_is_idempotent():
    v = _directions(4, 16)
    h = torch.randn(16, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    once = ablate(h, v)
    assert torch.allclose(ablate(once, v), once, atol=1e-9)


def test_ablate_handles_rank_deficient_direction_sets():
    # A duplicated direction must not break the projection or change the span.
    v = _directions(3, 16)
    stacked = torch.cat([v, v[:1]], dim=0)  # rank 3, k = 4
    h = torch.randn(16, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    assert torch.allclose(ablate(h, stacked), ablate(h, v), atol=1e-8)


def test_ablate_with_no_directions_is_an_exact_noop():
    h = torch.randn(8, generator=torch.Generator().manual_seed(5))
    assert torch.equal(ablate(h, torch.empty(0, 8)), h)


def test_ablate_applies_per_position_independently():
    # Batched call with per-position direction sets == per-row scalar calls.
    gen = torch.Generator().manual_seed(6)
    h = torch.randn(5, 16, generator=gen, dtype=torch.float64)
    v = torch.randn(5, 2, 16, generator=gen, dtype=torch.float64)
    batched = ablate(h, v)
    for pos in range(5):
        assert torch.allclose(batched[pos], ablate(h[pos], v[pos]), atol=1e-9)


# --- tiers, targets, helpers ----------------------------------------------------


def test_tier_bands_are_start_anchored_prefixes():
    band = list(range(9, 22))  # the 0.5B frozen band, 13 layers
    tiers = s3.tier_bands(band)
    assert tiers["light"] == band[:4]  # 13 // 3
    assert tiers["medium"] == band[:8]  # (2 * 13) // 3
    assert tiers["heavy"] == band
    assert tiers["light"] == tiers["medium"][: len(tiers["light"])]  # prefixes


def test_count_word_spells_the_letter_condition_truth():
    assert s3.count_word(10) == "ten"
    assert s3.count_word(15) == "fifteen"
    assert s3.count_word(20) == "twenty"
    assert s3.count_word(46) == "forty-six"
    assert s3.count_word(99) == "ninety-nine"
    with pytest.raises(ValueError):
        s3.count_word(9)


def test_target_ids_pool_case_and_space_variants():
    tok = VocabTokenizer({"forty": 1, " forty": 2, "Forty": 3, " 46": 4})
    assert s3.target_ids(["forty"], tok) == [1, 2, 3]
    assert s3.target_ids(["46"], tok) == [4]


def test_question_positions_cover_exactly_the_post_passage_span():
    # Stub encoder: one "token" per character makes the boundary arithmetic exact.
    encode = lambda t: torch.empty(1, len(t))
    template = "Read:\n\n{text}\n\nName the language."
    positions, seq_len = s3.question_positions(template, "Hola", encode)
    prefix = len("Read:\n\nHola")
    assert positions == list(range(prefix, seq_len))
    assert seq_len == len(template.format(text="Hola"))


# --- item-set drift guards --------------------------------------------------------


def test_load_language_guards_against_drift(tmp_path):
    bad = tmp_path / "selectivity-language.json"
    bad.write_text(json.dumps({"passages": [{"category": "French"}]}))
    with pytest.raises(ValueError, match="drifted"):
        s3.load_language(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        s3.load_language(str(tmp_path / "missing.json"))
    real = s3.load_language()  # the shipped file passes its own guard
    assert len(real["passages"]) == 8


def test_load_linecount_guards_against_drift(tmp_path):
    bad = tmp_path / "selectivity-linecount.json"
    bad.write_text(json.dumps({"passages": []}))
    with pytest.raises(ValueError, match="drifted"):
        s3.load_linecount(str(bad))
    real = s3.load_linecount()
    assert [p["width"] for p in real["passages"]][0] == 40  # gettysburg


# --- selection + read-back through the edit (the D6-replacement gates) -----------


class IdentitySubject:
    """d_model == vocab and unembed == identity matmul: lens logits are the
    residual coordinates themselves, so selection is fully predictable."""

    def __init__(self, d: int):
        self.d = d
        self.weight = torch.eye(d)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual @ self.weight.T


def test_ablation_edit_selects_top_lens_tokens_and_honors_the_exclusion(monkeypatch):
    monkeypatch.setattr(s3, "ABLATION_K", 2)
    d = 8
    subject = IdentitySubject(d)
    h = torch.zeros(1, 1, d)
    h[0, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125])
    clean_top10 = torch.tensor([[0]])  # the top lens token is an intended output
    edits = s3.ablation_edits(
        subject, {0: torch.eye(d)}, [0], torch.eye(d), clean_top10
    )
    out = edits[0](h)[0, 0]
    assert float(out[1]) == pytest.approx(0.0, abs=1e-6)  # selected: next-best two
    assert float(out[2]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[0]) == pytest.approx(5.0, abs=1e-6)  # excluded — untouched
    assert float(out[3]) == pytest.approx(2.0, abs=1e-6)  # unselected — untouched


def test_ablation_edit_readback_fires_invalid_on_a_rigged_operator(monkeypatch):
    monkeypatch.setattr(s3, "ABLATION_K", 2)
    monkeypatch.setattr(s3, "ablate", lambda h, v: h)  # a broken operator
    d = 8
    subject = IdentitySubject(d)
    h = torch.zeros(1, 1, d)
    h[0, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125])
    edits = s3.ablation_edits(
        subject, {0: torch.eye(d)}, [0], torch.eye(d), torch.tensor([[0]])
    )
    with pytest.raises(SystemExit) as exc:
        edits[0](h)
    assert exc.value.code == 2


def test_random_mode_skips_selection_and_is_seed_deterministic(monkeypatch):
    monkeypatch.setattr(s3, "ABLATION_K", 2)
    d = 8

    class NoLensSubject:
        def unembed(self, residual):  # selection must never run in random mode
            raise AssertionError("random mode must not read the lens")

    h = torch.randn(1, 3, d, generator=torch.Generator().manual_seed(7))
    outs = []
    for _ in range(2):
        gen = torch.Generator().manual_seed(s3.RANDOM_SEED)
        edits = s3.ablation_edits(
            NoLensSubject(), {}, [0], torch.eye(d), torch.tensor([[0]]),
            random_directions=gen,
        )
        outs.append(edits[0](h.clone()))
    assert torch.allclose(outs[0], outs[1])  # frozen seed → identical control
    assert not torch.allclose(outs[0], h)  # and it really removed something


# --- wrong-arm inputs exit INVALID ------------------------------------------------


def test_validate_rejects_a_mismatched_lens():
    class FakeSubject:
        d_model = 4
        n_layers = 24

    class Args:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    artifact = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "d_model": 4, "J": {}}
    with pytest.raises(SystemExit) as exc:
        s3.validate(Args(), artifact, FakeSubject())
    assert exc.value.code == 2
