"""stats.py — proportion confidence intervals (ported from lossy-wall, itself from decay-pin).

Every headline number in this project is a **proportion**: k hits out of n trials — a
rank-1 hit rate on a lens-eval distribution, a swap→answer flip rate, a verbal-report
success rate. A proportion is NOT a normal "average," so the familiar mean ± standard-
deviation is the wrong ruler for it (it can hand you an interval that runs below 0% or
above 100%, which is nonsense for a rate). This module gives the right rulers — the
methodology guardrail in code form:

  - `wilson(k, n)` — the **Wilson score interval**: the honest range a single cell's *true*
    rate likely sits in, given only n samples. Well-behaved at the edges (k=0 or k=n) and
    never escapes [0, 1], which is exactly where ±std falls apart. Our cells live at the
    edges too: the no-instruction modulation baseline sits ≈0%, pure-lens-vector report
    sits near ceiling (paper: 88% top-5). k=0 at n=20 reads [0%, ~16%] — "consistent with
    ~0%", never "proved 0%". The paper itself reports Wilson 95% CIs, so comparisons to
    its anchors are like-for-like.

  - `newcombe_diff(...)` — the **Newcombe interval** for the *difference* between two
    cells' rates. This is the number the gates turn on: the M0 readability gate is
    (J-lens − logit-lens baseline) excluding zero; the M2 component contrast is
    (J-space − non-J-space) excluding zero. If the interval straddles 0, we cannot claim
    a gap — `excludes_zero(...)` is that test.

Why these two specifically: measuring a delta between two cells needs intervals that stay
sane with few samples and near 0%/100% — precisely the Wilson/Newcombe regime. Pure
functions, no model, no network: unit-tested in test_stats.py.

Plain-English terms:
  - *proportion* — a fraction of hits, k/n.
  - *confidence interval (CI)* — the honest range the true value likely sits in given only n
    samples; here we use the conventional 95% level (z = 1.96).
  - *z* — how many standard-normal widths the interval spans; z = 1.96 is the 95% two-sided value.
"""
from __future__ import annotations

import math

Z_95 = 1.96  # two-sided 95% standard-normal critical value


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval (lo, hi) for a proportion k/n at confidence level `z`.

    The interval is centred on a slightly shrunk estimate (pulled toward 1/2) and is asymmetric
    near the edges — which is the honest behaviour: 0/20 hits does NOT mean the true rate
    is exactly 0%, so the upper bound sits sensibly above 0.0 rather than collapsing. Always
    inside [0, 1].

    `n = 0` returns (0.0, 1.0): with no data we know nothing, so the interval is the whole range.
    """
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(
    k_base: int, n_base: int, k_mech: int, n_mech: int, z: float = Z_95
) -> tuple[float, float, float]:
    """Newcombe interval for the difference d = p_mech - p_base. Returns (d, lo, hi).

    `d` is the point estimate of how much arm 2 ("mech") exceeds arm 1 ("base"). `(lo, hi)`
    is the honest range on that difference. This is Newcombe's "square-and-add" method (his
    method 10): it combines each arm's own Wilson interval rather than assuming a single
    pooled normal spread, which keeps it trustworthy even when an arm sits near 0% or 100%
    and when n is small.

    dim-stage convention: arm 1 ("base") is the reference arm, arm 2 ("mech") the arm
    expected higher, so a positive `d` whose (lo, hi) stays above 0 is a real advantage
    for arm 2. Call sites name the arms: the readability gate passes base=logit_lens,
    mech=j_lens; the component contrast passes base=non_j_space, mech=j_space.
    """
    p1 = k_base / n_base if n_base else 0.0   # arm 1: reference
    p2 = k_mech / n_mech if n_mech else 0.0   # arm 2: expected higher
    l1, u1 = wilson(k_base, n_base, z)
    l2, u2 = wilson(k_mech, n_mech, z)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (d, lo, hi)


def excludes_zero(lo: float, hi: float) -> bool:
    """True iff the difference interval (lo, hi) does NOT straddle 0 — i.e. a real effect.

    This is the honesty gate: if the Newcombe interval includes 0, the two cells are
    statistically indistinguishable at this n and we report 'no clear effect,' never a win.
    Every milestone gate in docs/KICKOFF.md has an explicit CI condition; a claim that
    doesn't clear its gate doesn't get made.
    """
    return lo > 0.0 or hi < 0.0
