"""test_stats.py — offline unit tests for the proportion CIs (ported from lossy-wall/decay-pin).

No network, no model. One command greens the repo before any real run: `uv run pytest`.
Every check asserts (no print-and-collect traps). Reference values are decay-pin's,
unchanged (hand-computed, 95%, z = 1.96), plus the cell shapes dim-stage's gates lean on:
the ≈0% floor (no-instruction modulation baseline) and the big-contrast shape
(J-space vs non-J-space).

These intervals are the project's *ruler*: if the math is off, every readability /
report / swap / modulation verdict downstream is wrong.
"""
from __future__ import annotations

from stats import excludes_zero, newcombe_diff, wilson


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def test_wilson_known_values():
    # p = 0.5, n = 20 -> symmetric around 0.5, ~(0.299, 0.701)
    lo, hi = wilson(10, 20)
    assert close(lo, 0.2993)
    assert close(hi, 0.7007)
    assert close((lo + hi) / 2, 0.5)

    # p = 0.8, n = 20 -> ~(0.584, 0.919)
    lo, hi = wilson(16, 20)
    assert close(lo, 0.5840)
    assert close(hi, 0.9193)


def test_wilson_edges():
    # k=0 is where the no-instruction modulation baseline lives: upper ~16% at n=20 —
    # "consistent with ~0%", never "proved 0%".
    lo, hi = wilson(0, 20)
    assert lo == 0.0
    assert close(hi, 0.1611)

    lo, hi = wilson(20, 20)
    assert hi == 1.0
    assert close(lo, 0.8389)

    lo, hi = wilson(0, 0)
    assert lo == 0.0 and hi == 1.0

    for k, n in [(0, 5), (3, 5), (5, 5), (7, 13), (40, 40)]:
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_newcombe_overlap_case():
    # A mild gap (16/20 vs 20/20) straddles 0 -> "not a result".
    d, lo, hi = newcombe_diff(16, 20, 20, 20)  # 80% vs 100%
    assert close(d, 0.20)
    assert close(lo, -0.0005)
    assert close(hi, 0.4160)
    assert excludes_zero(lo, hi) is False


def test_newcombe_clear_case():
    # A crisp gap (24/40 vs 38/40) excludes 0 -> a real result.
    d, lo, hi = newcombe_diff(24, 40, 38, 40)  # 60% vs 95%
    assert close(d, 0.35)
    assert close(lo, 0.171, tol=2e-3)
    assert close(hi, 0.508, tol=2e-3)
    assert excludes_zero(lo, hi) is True


def test_newcombe_component_contrast_shape():
    # The shape M2 hopes to see (paper: J-space 59-61% vs non-J-space 5-28%):
    # base=non_j_space 2/40, mech=j_space 24/40.
    d, lo, hi = newcombe_diff(2, 40, 24, 40)  # 5% vs 60%
    assert close(d, 0.55)
    assert lo > 0.0 and excludes_zero(lo, hi) is True
    assert -1.0 <= lo <= hi <= 1.0


def test_excludes_zero_logic():
    assert excludes_zero(0.1, 0.4) is True
    assert excludes_zero(-0.4, -0.1) is True   # a real *negative* effect still excludes 0
    assert excludes_zero(-0.05, 0.30) is False
    assert excludes_zero(0.0, 0.30) is False   # touching 0 is NOT a clear result
