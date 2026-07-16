"""test_intervention.py — the D6 pre-committed correctness gate (no downloads).

The reference ships no intervention code, so M1 has no AGREE oracle; these
invariants ARE the gate, merged before any real run (DECISIONS.md 2026-07-16).
Three families, per the frozen D6:

1. **Rigged-subject oracle** — a tiny model (ConstantBlock band layer + one
   exact linear block) where the true Jacobian is known analytically, so the
   swap's and steering's effect on the output logits is hand-computable. All
   rig numbers are dyadic (powers of two), so assertions are exact equality,
   not tolerances — a sign, transpose, or normalization bug cannot hide.
2. **Coordinate read-back** — after patching, the lens coordinates come back
   exactly exchanged (c' = σ(c)) and the component of the change orthogonal to
   span{v_s, v_t} is zero. Checked on the rig (exact) and on random tensors
   (fp64 tolerance). The measurement runner repeats the read-back as a runtime
   self-check on real subjects when it lands.
3. **Null-ops** — α = 0 steering and s = t swaps change nothing, exactly.

Plus one guard against misread algebra: the implementation's rank-1 form
(c_t − c_s)(v_s − v_t) must equal the paper's literal matrix form V(σ(c) − c)
with c = V†h computed by torch.linalg.pinv, on random tensors.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from fitter import _record_residuals
from intervention import (
    edit_residuals,
    jlens_vector,
    lens_coordinates,
    positional,
    steer,
    steer_edits,
    steering_direction,
    swap,
    swap_edits,
)
from test_readability import ConstantBlock

D_MODEL = 4
VOCAB = 11
S, T = 2, 3  # swap-out / swap-in token ids in the rig


class LinearBlock(nn.Module):
    """h -> h @ A.T (per position: h ↦ A h) — a fixed linear map, so the true
    Jacobian of this block's output with respect to its input is exactly A."""

    def __init__(self, A: torch.Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A.T


class FlowRig:
    """Duck-typed subject: block 0 (the 'band layer') emits a chosen constant
    residual at every position; block 1 is the exact linear map A. So
    J_0 = dh_final/dh_0 = A analytically, and every post-patch logit is
    hand-computable. (test_readability's RiggedSubject blocks ignore their
    input, so a patch could never propagate — the LinearBlock transforms its
    input, which is the whole point here.)"""

    def __init__(self, h_band: torch.Tensor, A: torch.Tensor, U: torch.Tensor):
        self.layers = nn.ModuleList([ConstantBlock(h_band.repeat(8, 1)), LinearBlock(A)])
        self.n_layers = 2
        self.d_model = h_band.shape[-1]
        self._U = U

    def forward(self, input_ids: torch.Tensor) -> None:
        h = torch.zeros(*input_ids.shape, self.d_model)
        for block in self.layers:
            h = block(h)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return residual @ self._U.T


def make_rig():
    """The shared analytic setup. A = cyclic shift (A e_i = e_{i+1 mod 4}), so
    Aᵀ ≠ A and any J-vs-Jᵀ mixup lands on the wrong basis vector.

      u_s = 2e2, u_t = 4e3   ⇒  v_s = Aᵀu_s = 2e1,  v_t = Aᵀu_t = 4e2
      h_band = 7e0 + 8e1 + 1e2  ⇒  c = (⟨h,v_s⟩/4, ⟨h,v_t⟩/16) = (4, 0.25)
      swap ⇒ h' = h + (0.25 − 4)(v_s − v_t) = 7e0 + 0.5e1 + 16e2
      final = A·h: pre = 7e1+8e2+1e3 (logit_s=16 > logit_t=4)
                  post = 7e1+0.5e2+16e3 (logit_s=1  < logit_t=64)
    """
    A = torch.zeros(D_MODEL, D_MODEL)
    for j in range(D_MODEL):
        A[(j + 1) % D_MODEL, j] = 1.0
    U = torch.zeros(VOCAB, D_MODEL)
    U[S, 2] = 2.0
    U[T, 3] = 4.0
    U[5, 0] = 1.0  # a competitor row so ranks have competition
    h_band = torch.tensor([7.0, 8.0, 1.0, 0.0])
    return FlowRig(h_band, A, U), {0: A}, U


def random_case(d: int = 8):
    """Seeded fp64 h/v_s/v_t for the tolerance-based property tests."""
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(2, 3, d, generator=gen, dtype=torch.float64)
    v_s = torch.randn(d, generator=gen, dtype=torch.float64)
    v_t = torch.randn(d, generator=gen, dtype=torch.float64)
    return h, v_s, v_t


# --- operator algebra ---------------------------------------------------------


def test_jlens_vector_is_jacobian_transpose_times_row():
    _, jacobians, U = make_rig()
    assert torch.equal(jlens_vector(jacobians[0], U[S]), 2.0 * torch.eye(D_MODEL)[1])
    assert torch.equal(jlens_vector(jacobians[0], U[T]), 4.0 * torch.eye(D_MODEL)[2])


def test_lens_coordinates_exact_on_nonorthogonal_directions():
    # V = [e0+e1, e1]: G = [[2,1],[1,1]], det = 1; h = 3e0+5e1 = 3v_s + 2v_t.
    v_s = torch.tensor([1.0, 1.0, 0.0, 0.0])
    v_t = torch.tensor([0.0, 1.0, 0.0, 0.0])
    h = torch.tensor([3.0, 5.0, 0.0, 0.0])
    assert torch.equal(lens_coordinates(h, v_s, v_t), torch.tensor([3.0, 2.0]))


def test_parallel_directions_are_rejected():
    h, v_s, _ = random_case()
    with pytest.raises(ValueError, match="parallel"):
        lens_coordinates(h, v_s, 2.0 * v_s)
    with pytest.raises(ValueError, match="parallel"):
        swap(h, v_s, 2.0 * v_s)  # distinct vectors, same degeneracy


def test_swap_matches_the_literal_paper_formula():
    # Implementation form (c_t − c_s)(v_s − v_t) ≡ literal V(σ(c) − c), c = V†h.
    h, v_s, v_t = random_case()
    V = torch.stack((v_s, v_t), dim=-1)  # [d, 2]
    c = h @ torch.linalg.pinv(V).T  # [..., 2]
    literal = h + (c.flip(-1) - c) @ V.T
    assert torch.allclose(swap(h, v_s, v_t), literal, atol=1e-12)


def test_swap_exchanges_coordinates_and_preserves_orthogonal_component():
    h, v_s, v_t = random_case()
    patched = swap(h, v_s, v_t)
    delta = patched - h
    assert delta.norm() > 0.1  # the swap did something
    # c' = σ(c), through the same coordinate reader the runner will use.
    c_before = lens_coordinates(h, v_s, v_t)
    c_after = lens_coordinates(patched, v_s, v_t)
    assert torch.allclose(c_after, c_before.flip(-1), atol=1e-10)
    # Δh lies entirely in span{v_s, v_t}: its least-squares reconstruction
    # against V is itself, i.e. the orthogonal component is zero.
    p = lens_coordinates(delta, v_s, v_t)
    recon = p[..., 0:1] * v_s + p[..., 1:2] * v_t
    assert torch.allclose(recon, delta, atol=1e-10)


# --- null-ops (D6 family 3) ---------------------------------------------------


def test_swap_with_itself_is_an_exact_noop():
    h, v_s, _ = random_case()
    assert swap(h, v_s, v_s.clone()) is h  # short-circuit: bitwise by construction


def test_steer_at_alpha_zero_is_an_exact_noop():
    h, _, v_t = random_case()
    assert torch.equal(steer(h, v_t, 0.0), h)


def test_steer_adds_exactly_alpha_times_direction():
    h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    d = torch.tensor([0.5, 0.0, -1.0, 0.0])
    assert torch.equal(steer(h, d, 2.0), h + torch.tensor([1.0, 0.0, -2.0, 0.0]))


def test_steering_direction_convention():
    _, jacobians, U = make_rig()
    # v_t = 4e2 ⇒ unit e2, scaled to the mean residual norm.
    direction = steering_direction(jacobians[0], U[T], 2.0)
    assert torch.equal(direction, 2.0 * torch.eye(D_MODEL)[2])
    with pytest.raises(ValueError, match="zero vector"):
        steering_direction(torch.zeros(D_MODEL, D_MODEL), U[T], 1.0)


# --- hook machinery -----------------------------------------------------------


def test_positional_restricts_the_edit_to_listed_positions():
    h = torch.arange(8.0).reshape(1, 4, 2)
    out = positional(lambda x: 2.0 * x, [1, 3])(h)
    assert torch.equal(out[:, [1, 3]], 2.0 * h[:, [1, 3]])
    assert torch.equal(out[:, [0, 2]], h[:, [0, 2]])
    assert torch.equal(h, torch.arange(8.0).reshape(1, 4, 2))  # input unmutated


def test_swap_edits_uses_each_layers_own_jacobian():
    _, rig_jacobians, U = make_rig()
    A = rig_jacobians[0]
    jacobians = {0: A, 1: torch.eye(D_MODEL)}
    edits = swap_edits(jacobians, [0, 1], U[S], U[T])
    probe = torch.tensor([[[7.0, 8.0, 1.0, 0.0]]])
    expected_0 = swap(probe, jlens_vector(A, U[S]), jlens_vector(A, U[T]))
    expected_1 = swap(probe, U[S], U[T])  # J = I ⇒ v is the raw row
    assert torch.equal(edits[0](probe), expected_0)
    assert torch.equal(edits[1](probe), expected_1)
    assert not torch.equal(edits[0](probe), edits[1](probe))  # late binding guard


# --- the rigged-subject oracle (D6 families 1 + 2) ----------------------------


def test_swap_through_the_model_matches_the_analytic_prediction():
    rig, jacobians, U = make_rig()
    input_ids = torch.tensor([[0, 1, 2]])

    # Baseline: s on top by construction (logit_s = 16 vs logit_t = 4).
    with _record_residuals(rig.layers, [1], graph_root=None) as residuals:
        rig.forward(input_ids)
    pre_logits = rig.unembed(residuals[1][0, -1])
    assert torch.equal(pre_logits[S], torch.tensor(16.0))
    assert torch.equal(pre_logits[T], torch.tensor(4.0))
    assert int(pre_logits.argmax()) == S

    edits = swap_edits(jacobians, [0], U[S], U[T])
    with (
        edit_residuals(rig.layers, edits),
        _record_residuals(rig.layers, [0, 1], graph_root=None) as residuals,
    ):
        rig.forward(input_ids)

    # The band residual is exactly the hand-computed patch, at every position…
    expected_band = torch.tensor([7.0, 0.5, 16.0, 0.0])
    assert torch.equal(residuals[0], expected_band.expand(1, 3, -1))
    # …its coordinates read back exactly exchanged: (4, 0.25) → (0.25, 4)…
    v_s = jlens_vector(jacobians[0], U[S])
    v_t = jlens_vector(jacobians[0], U[T])
    c_after = lens_coordinates(residuals[0][0, 0], v_s, v_t)
    assert torch.equal(c_after, torch.tensor([0.25, 4.0]))
    # …and the patch propagates through A to exactly the predicted logits:
    # final = A·h' = 7e1 + 0.5e2 + 16e3 ⇒ logit_s = 1, logit_t = 64, rest 0.
    post_logits = rig.unembed(residuals[1][0, -1])
    expected = torch.zeros(VOCAB)
    expected[S], expected[T] = 1.0, 64.0
    assert torch.equal(post_logits, expected)
    assert int(post_logits.argmax()) == T  # the swap flipped the model's answer


def test_steering_through_the_model_matches_the_analytic_prediction():
    rig, jacobians, U = make_rig()
    input_ids = torch.tensor([[0, 1, 2]])
    # d = unit(v_t)·mean_norm = 2e2; α = 2 ⇒ h' = h + 4e2 ⇒ final = A·h' adds
    # 4e3 ⇒ logit_t: 4 → 20, logit_s untouched at 16.
    edits = steer_edits(jacobians, [0], U[T], {0: 2.0}, 2.0)
    with (
        edit_residuals(rig.layers, edits),
        _record_residuals(rig.layers, [1], graph_root=None) as residuals,
    ):
        rig.forward(input_ids)
    logits = rig.unembed(residuals[1][0, -1])
    assert torch.equal(logits[T], torch.tensor(20.0))
    assert torch.equal(logits[S], torch.tensor(16.0))
    assert int(logits.argmax()) == T


def test_edit_hooks_are_removed_when_the_context_exits():
    rig, jacobians, U = make_rig()
    input_ids = torch.tensor([[0, 1, 2]])
    with edit_residuals(rig.layers, swap_edits(jacobians, [0], U[S], U[T])):
        pass  # enter and exit without a forward
    with _record_residuals(rig.layers, [1], graph_root=None) as residuals:
        rig.forward(input_ids)
    logits = rig.unembed(residuals[1][0, -1])
    assert torch.equal(logits[S], torch.tensor(16.0))  # unedited baseline
    assert int(logits.argmax()) == S
