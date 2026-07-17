"""intervention.py — the M1 swap + steering operators (torch-only; never imports jlens).

M0 only ever *read* residuals; this module *writes* them mid-forward-pass. The
reference repo ships no intervention code (M1-BRIEF, "What the reference does NOT
ship"), so there is no oracle to diff against — correctness is enforced by the
pre-committed D6 invariants in test_intervention.py instead of an AGREE gate.

The operators, verbatim from the paper (M1-BRIEF, "Design extraction"):

- **J-lens vector**: `v_t = J_lᵀ u_t`, token t's direction in the layer-l residual
  stream, where `u_t` is token t's row of the unembedding matrix. The lens logit
  for t is (approximately) `⟨v_t, h⟩`. Frozen convention: `u_t` is the raw
  `lm_head.weight` row — the literal formula reading; the final RMSNorm's
  elementwise scale sits between the residual and that matrix in Qwen and is NOT
  folded in (owned, pre-declared; DECISIONS.md 2026-07-16).
- **Swap** (patching in lens coordinates): with `V = [v_s v_t]` and lens
  coordinates `c = V†h` (`V†` = pseudoinverse — the least-squares way to read
  coordinates against two non-orthogonal directions), set
  `h ← h + V(σ(c) − c)` where σ exchanges the two entries of c. The component of
  h orthogonal to span{v_s, v_t} is unchanged. Implemented through the algebraic
  identity `V(σ(c) − c) = (c_t − c_s)(v_s − v_t)` (σ(c) − c = (c_t − c_s)·[1, −1]),
  which test_intervention.py proves equal to the literal matrix form; it makes
  s = t an exact no-op, since v_s − v_t is exactly zero.
- **Steering**: `h ← h + α·d_t` with direction `d_t` = unit-normalized `v_t`
  scaled by the layer's mean residual norm (reference README convention), so
  strengths are in units of the layer's typical residual size. α = 0 is the
  control and an exact no-op; negative α is an ablation direction.
- **Ablation** (S3, D24): `h ← h − V†-projection of h onto span{v_1..v_k}` —
  remove the residual's component along a set of lens directions entirely,
  leaving every selected direction's lens coordinate at exactly zero and the
  orthogonal complement untouched. Selection (top-k by lens logit, clean
  top-10 output exclusion) lives with the S3 runner; the operator is pure
  geometry.

Application point: edits replace a block's **output** residual — the same hook
point `fitter._record_residuals` captures and the lens reads — so every later
layer (and the final unembedding) consumes the edited stream. Both M1 protocols
apply an edit at every frozen-band layer, each layer using its own `J_l`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager

import torch
from torch import nn

#: Reject (near-)parallel direction pairs: the 2x2 normal-equations solve is
#: degenerate there and the coordinates blow up. Distinct real tokens are
#: nowhere near this (relative Gram determinant ~1 in practice).
PARALLEL_TOLERANCE = 1e-6

#: Relative residual floor for the ablation projection's Gram-Schmidt basis:
#: a direction whose component orthogonal to the already-accepted basis falls
#: below this fraction of its own norm is numerically inside their span and
#: is dropped from the basis (its coordinate is removed by the others).
RANK_TOLERANCE = 1e-10

Edit = Callable[[torch.Tensor], torch.Tensor]


def jlens_vector(jacobian: torch.Tensor, unembed_row: torch.Tensor) -> torch.Tensor:
    """v_t = J_lᵀ u_t: the [d_model] residual-stream direction for one token."""
    return jacobian.T @ unembed_row


def lens_coordinates(
    h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor
) -> torch.Tensor:
    """c = V†h for V = [v_s v_t]: least-squares coordinates of h [..., d_model]
    against the two directions, returned as [..., 2] (order: c_s, c_t).

    Solved via the closed-form 2x2 normal equations (`V†h = (VᵀV)⁻¹Vᵀh`) — exact
    for this two-column case and free of torch.linalg kernels, which MPS does
    not fully cover. Raises on (near-)parallel directions.
    """
    g_ss, g_st, g_tt = v_s @ v_s, v_s @ v_t, v_t @ v_t
    det = g_ss * g_tt - g_st * g_st
    if det <= PARALLEL_TOLERANCE * g_ss * g_tt:
        raise ValueError(
            "v_s and v_t are (near-)parallel — lens coordinates are degenerate"
        )
    b_s, b_t = h @ v_s, h @ v_t
    c_s = (g_tt * b_s - g_st * b_t) / det
    c_t = (g_ss * b_t - g_st * b_s) / det
    return torch.stack((c_s, c_t), dim=-1)


def swap(
    h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """h + α·V(σ(c) − c): exchange h's lens coordinates along v_s and v_t, leaving
    the component orthogonal to span{v_s, v_t} untouched. h is [..., d_model];
    the swap applies independently at every leading position.

    α scales the correction term (S2's frozen D21 reading of the paper's
    "double strength" swap): the exchange is an involution — applying it twice
    restores the original — so α = 2 can only mean twice the correction, never
    apply-twice. α = 1 is the exact exchange, bit-identical to the pre-S2
    operator; α = 0 is an exact no-op (the correction is exactly zero).

    s = t short-circuits to h unchanged — the exact value of the formula there
    (σ swaps two equal coordinates), reached before the degenerate solve.
    """
    if torch.equal(v_s, v_t):
        return h
    c = lens_coordinates(h, v_s, v_t)
    delta = (c[..., 1] - c[..., 0]).unsqueeze(-1)
    return h + alpha * delta * (v_s - v_t)


def steer(h: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """h + α·direction, at every leading position of h. α = 0 is an exact no-op."""
    return h + alpha * direction


def ablate(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """h minus its projection onto span{directions}: the S3 ablation operator
    (D24). h is [..., d_model]; directions is [..., k, d_model] with matching
    leading dims (each position carries its own direction set). Returns the
    unique minimal-norm edit of h whose inner product with every direction is
    zero — "zero out the residual stream's projection onto each" achieved
    simultaneously, which one-at-a-time zeroing of non-orthogonal directions
    would not do (S3-BRIEF, design extraction).

    Computed by modified Gram-Schmidt with re-orthogonalization on CPU in
    float64. Real top-k lens direction sets are brutally ill-conditioned —
    near-duplicate tokens (including Qwen's untrained reserved vocab slots)
    give nearly identical directions, where a least-squares solve blows up
    and LAPACK's iterative SVD refuses to converge; S3's first smoke runs
    tripped the runtime read-back / crashed on exactly those. MGS never
    iterates: each direction is orthogonalized against the accepted basis
    (twice — the classical fix for cancellation) and dropped if nothing new
    survives, so numerically dependent directions land inside the kept span
    and every selected direction's coordinate still ends at ~0. k = 0 is an
    exact no-op.
    """
    if directions.shape[-2] == 0:
        return h
    # Move BEFORE casting: `.to("cpu", float64)` in one step silently corrupts
    # values coming off MPS (float64 is cast device-side, unsupported there —
    # measured 2026-07-17, max abs diff ~5 on unit-scale data).
    v = directions.cpu().to(torch.float64)  # [..., k, d_model]
    b = h.cpu().to(torch.float64)
    basis: list[torch.Tensor] = []
    for i in range(v.shape[-2]):
        vec = v[..., i, :]
        norm0 = vec.norm(dim=-1, keepdim=True)
        for _ in range(2):
            for q in basis:
                vec = vec - (vec * q).sum(-1, keepdim=True) * q
        norm = vec.norm(dim=-1, keepdim=True)
        keep = norm > norm0 * RANK_TOLERANCE
        basis.append(
            torch.where(keep, vec / norm.clamp_min(torch.finfo(vec.dtype).tiny), 0.0)
        )
    for q in basis:
        b = b - (b * q).sum(-1, keepdim=True) * q
    return b.to(device=h.device, dtype=h.dtype)


def steering_direction(
    jacobian: torch.Tensor, unembed_row: torch.Tensor, mean_residual_norm: float
) -> torch.Tensor:
    """The reference README's steering direction for one token at one layer:
    unit-normalized v_t = J_lᵀu_t, scaled by the layer's mean residual norm
    (strength α multiplies at application time)."""
    v = jlens_vector(jacobian, unembed_row)
    norm = v.norm()
    if norm == 0:
        raise ValueError("J_lᵀu_t is the zero vector — no steering direction")
    return v / norm * mean_residual_norm


def positional(edit: Edit, positions: Sequence[int] | None) -> Edit:
    """Restrict an edit to the listed token positions of a [batch, seq, d_model]
    residual (None = every position, the verbal-report convention). Untouched
    positions keep their exact original values."""
    if positions is None:
        return edit

    def restricted(h: torch.Tensor) -> torch.Tensor:
        out = h.clone()
        out[:, list(positions), :] = edit(h[:, list(positions), :])
        return out

    return restricted


@contextmanager
def edit_residuals(layers: Sequence[nn.Module], edits: dict[int, Edit]):
    """Apply each edit to its block's output residual on every forward pass run
    inside the context — the write-side mirror of `fitter._record_residuals`:
    same hook point (block output), so the edited residual is exactly what the
    lens would read there and what every later layer consumes. A recording hook
    registered *inside* this context sees the edited stream (torch runs forward
    hooks in registration order, each receiving the previous one's replacement).
    """
    handles = []

    def make_hook(edit: Edit):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                return edit(output)
            return (edit(output[0]), *output[1:])

        return hook

    try:
        for index, edit in edits.items():
            handles.append(layers[index].register_forward_hook(make_hook(edit)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def swap_edits(
    jacobians: dict[int, torch.Tensor],
    at: Sequence[int],
    u_s: torch.Tensor,
    u_t: torch.Tensor,
    *,
    positions: Sequence[int] | None = None,
    alpha: float = 1.0,
) -> dict[int, Edit]:
    """Per-layer swap edits for `edit_residuals`: each listed layer exchanges
    the u_s ↔ u_t lens coordinates using its own J_l (protocol 1 applies this
    at every band layer and, by default, every token position). α scales the
    correction (see `swap`); the default reproduces the pre-S2 operator."""
    edits: dict[int, Edit] = {}
    for layer in at:
        v_s = jlens_vector(jacobians[layer], u_s)
        v_t = jlens_vector(jacobians[layer], u_t)

        def edit(h: torch.Tensor, v_s=v_s, v_t=v_t) -> torch.Tensor:
            return swap(
                h,
                v_s.to(device=h.device, dtype=h.dtype),
                v_t.to(device=h.device, dtype=h.dtype),
                alpha,
            )

        edits[layer] = positional(edit, positions)
    return edits


def steer_edits(
    jacobians: dict[int, torch.Tensor],
    at: Sequence[int],
    u_t: torch.Tensor,
    mean_residual_norms: dict[int, float],
    alpha: float,
    *,
    positions: Sequence[int] | None = None,
) -> dict[int, Edit]:
    """Per-layer steering edits for `edit_residuals`: each listed layer adds
    α times its own mean-norm-scaled direction (protocol 2 applies this at every
    band layer, restricted to the user-question-turn positions)."""
    edits: dict[int, Edit] = {}
    for layer in at:
        direction = steering_direction(
            jacobians[layer], u_t, mean_residual_norms[layer]
        )

        def edit(h: torch.Tensor, direction=direction) -> torch.Tensor:
            return steer(
                h, direction.to(device=h.device, dtype=h.dtype), alpha
            )

        edits[layer] = positional(edit, positions)
    return edits
