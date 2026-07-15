"""test_fitter.py — analytic checks of the independent fitter (no model download).

The AGREE gate (m0_agree_gate.py) is the real verdict on the fitter; these tests
are the cheap guard that runs in seconds. They build tiny models whose true
Jacobian is known *exactly* — for a linear map the Jacobian IS the matrix — so any
drift in the estimator (a transpose, a wrong position set, a batching bug) fails
here before it burns half an hour of MPS time.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from fitter import SKIP_FIRST, fit, jacobian_for_prompt, lens_logits, valid_position_mask

D_MODEL = 4
VOCAB = 7


class PositionwiseLinear(nn.Module):
    """h_out[p] = h_in[p] @ A — each position transformed independently."""

    def __init__(self, A: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A


class CausalMix(nn.Module):
    """h_out[p] = (sum of h_in[q] for q <= p) @ A — mixes positions causally,
    so every earlier position feeds every later one (like attention does)."""

    def __init__(self, A: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(h, dim=1) @ self.A


class Square(nn.Module):
    """h_out[p] = (h_in[p] * h_in[p]) @ A — input-DEPENDENT Jacobian
    (d out / d in = 2 h_in A), so different prompts give different J."""

    def __init__(self, A: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h * h) @ self.A


class TinySubject:
    """Duck-typed stand-in for fitter.SubjectModel: one token per character."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        g = torch.Generator().manual_seed(0)
        self._embed = torch.randn(VOCAB, D_MODEL, generator=g)
        self._unembed = torch.randn(VOCAB, D_MODEL, generator=g)
        self.layers = nn.ModuleList(blocks)
        self.n_layers = len(blocks)
        self.d_model = D_MODEL

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        return torch.tensor([[ord(c) % VOCAB for c in text[:max_length]]])

    def forward(self, input_ids: torch.Tensor) -> None:
        h = self._embed[input_ids]
        for block in self.layers:
            h = block(h)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual @ self._unembed.T


def _rand(g: torch.Generator) -> torch.Tensor:
    return torch.randn(D_MODEL, D_MODEL, generator=g)


PROMPT = "abcdefg" * 4  # 28 tokens; valid positions 16..26


def test_positionwise_stack_recovers_exact_jacobian():
    # target = source_l @ (product of later A's), so J_l must be that product,
    # transposed (J rows are output dims, matrix columns are output dims).
    g = torch.Generator().manual_seed(1)
    A0, A1, A2 = _rand(g), _rand(g), _rand(g)
    subject = TinySubject([PositionwiseLinear(A) for A in (A0, A1, A2)])
    jacobians, seq_len, n_valid = jacobian_for_prompt(subject, PROMPT, [0, 1], dim_batch=3)
    assert (seq_len, n_valid) == (28, 11)
    torch.testing.assert_close(jacobians[0], (A1 @ A2).T)
    torch.testing.assert_close(jacobians[1], A2.T)


def test_causal_mix_sums_targets_and_means_sources():
    # With one CausalMix block, d target[p'] / d source[p] = A for every p' >= p.
    # Summing over valid targets p' >= p then averaging over valid sources p gives
    # J = mean_p |{valid p' >= p}| * A^T. For 11 valid positions the counts are
    # 11..1, whose mean is 6.
    g = torch.Generator().manual_seed(2)
    A0, A1 = _rand(g), _rand(g)
    subject = TinySubject([PositionwiseLinear(A0), CausalMix(A1)])
    jacobians, _, n_valid = jacobian_for_prompt(subject, PROMPT, [0], dim_batch=2)
    assert n_valid == 11
    torch.testing.assert_close(jacobians[0], 6.0 * A1.T)


def test_dim_batch_is_math_neutral():
    g = torch.Generator().manual_seed(3)
    subject = TinySubject([PositionwiseLinear(_rand(g)), CausalMix(_rand(g))])
    results = [
        jacobian_for_prompt(subject, PROMPT, [0], dim_batch=b)[0][0] for b in (1, 3, 4)
    ]
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])


def test_fit_is_mean_over_prompts_and_skips_short_ones():
    g = torch.Generator().manual_seed(4)
    subject = TinySubject([PositionwiseLinear(_rand(g)), Square(_rand(g))])
    long_a, long_b, too_short = "abcdefg" * 4, "gfedcba" * 5, "abc"
    fitted = fit(subject, [long_a, too_short, long_b], log=lambda *_: None)
    j_a = jacobian_for_prompt(subject, long_a, [0])[0][0]
    j_b = jacobian_for_prompt(subject, long_b, [0])[0][0]
    assert not torch.allclose(j_a, j_b)  # Square makes J input-dependent
    torch.testing.assert_close(fitted[0], (j_a + j_b) / 2)


def test_lens_readout_matches_model_logits_on_linear_stack():
    # On a purely linear model the transported residual IS the final residual,
    # so the lens logits must equal the model's own logits exactly. This pins
    # the J @ h orientation in lens_logits (a transposed J would fail loudly).
    g = torch.Generator().manual_seed(5)
    A0, A1, A2 = _rand(g), _rand(g), _rand(g)
    subject = TinySubject([PositionwiseLinear(A) for A in (A0, A1, A2)])
    jacobians = fit(subject, [PROMPT], source_layers=[0, 1], log=lambda *_: None)
    logits, input_ids = lens_logits(subject, jacobians, PROMPT, positions=[18, 20])
    h = subject._embed[input_ids]
    for A in (A0, A1, A2):
        h = h @ A
    expected = subject.unembed(h[0, [18, 20]])
    torch.testing.assert_close(logits[0], expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(logits[1], expected, rtol=1e-4, atol=1e-5)


def test_mask_rejects_too_short_prompts():
    with pytest.raises(ValueError, match="too short"):
        valid_position_mask(SKIP_FIRST + 1)
    mask = valid_position_mask(SKIP_FIRST + 2)
    assert mask.sum() == 1 and mask[SKIP_FIRST]
