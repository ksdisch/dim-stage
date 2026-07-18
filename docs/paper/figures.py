"""figures.py — deterministic figure rendering for the dim-stage paper.

Run (from the repo root):

    uv run --with matplotlib docs/paper/figures.py

matplotlib is injected for this run only, so pyproject.toml stays untouched.

Reads ONLY the committed per-run result files (results/*.json) — no models, no
lenses, no re-measurement. Every plotted value is a count or rate already recorded
in those files; nothing is smoothed, interpolated, or derived beyond hits/n.
Re-running this script on the committed JSONs regenerates the four PNGs in
docs/paper/figures/ bit-for-bit-equivalent (same data in, same pixels out).

Figures:
  fig-s1-dose-response.png  — M1/S1 introspection report rate vs steering strength α
                              (9-point grid), J-lens vs J = I, one panel per subject.
  fig-s1-localization.png   — S1 sub-band localization at 1.5B: report hits of 101
                              per steered sub-band, J-lens vs J = I, α = 24.
  fig-s3-retention.png      — S3 two-hop retention vs ablation tier per subject,
                              with the matched random-direction control and the
                              unablated baseline marked.
  fig-s2-alpha-cliff.png    — S2 pooled swap successes of 180 at α ∈ {1, 2, 4, 8},
                              J-lens vs J = I, one panel per subject.

The script prints every plotted number so a reviewer can eyeball them against the
paper's tables without opening the PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # docs/paper/
REPO = HERE.parents[1]                          # repo root
RESULTS = REPO / "results"
OUT = HERE / "figures"

SUBJECTS = ["0.5b", "1.5b", "3b"]
LABELS = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B"}


def load(stem: str, subject: str) -> dict:
    path = RESULTS / f"{stem}-qwen2.5-{subject}-instruct.json"
    with open(path) as f:
        return json.load(f)


def fig_s1_dose_response() -> None:
    """Report rate vs α, J-lens and J = I arms, one panel per subject.

    Data: results/s1-introspection-*.json, sweep.{jlens,identity}.{α}.report_hits
    over n = 101 concepts; the 9-point α grid {0, 0.5, 1, 2, 4, 8, 12, 16, 24}.
    α ticks are equally spaced (categorical) for legibility; the values are labeled.
    """
    alphas = ["0.0", "0.5", "1.0", "2.0", "4.0", "8.0", "12.0", "16.0", "24.0"]
    xticks = list(range(len(alphas)))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, subject in zip(axes, SUBJECTS):
        d = load("s1-introspection", subject)
        for arm, style, label in [("jlens", "o-", "J-lens"), ("identity", "s--", "J = I")]:
            cells = d["sweep"][arm]
            n = cells[alphas[0]]["n"]
            rates = [cells[a]["report_hits"] / cells[a]["n"] for a in alphas]
            hits = [cells[a]["report_hits"] for a in alphas]
            print(f"s1 dose-response {subject} {arm}: hits {hits} of n={n}")
            ax.plot(xticks, rates, style, label=label, markersize=4)
        ax.set_title(f"{LABELS[subject]} (n = 101)", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels(["0", ".5", "1", "2", "4", "8", "12", "16", "24"], fontsize=8)
        ax.set_xlabel("α (mean-residual-norm units)", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("report rate (rank-1 hits / 101)", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig-s1-dose-response.png", dpi=300)
    plt.close(fig)


def fig_s1_localization() -> None:
    """S1 sub-band localization bars at 1.5B, α = 24.

    Data: results/s1-introspection-qwen2.5-1.5b-instruct.json, localization block:
    report_hits of n = 101 when steering only L11–15 / L16–20 / L21–24, plus the
    full band L11–24, J-lens vs J = I arms.
    """
    d = load("s1-introspection", "1.5b")
    loc = d["localization"]
    bands = ["L11-15", "L16-20", "L21-24"]
    groups = [(b, loc["sub_bands"][b]) for b in bands] + [("L11-24 (full)", loc["full_band"])]
    x = list(range(len(groups)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for off, arm, label in [(-width / 2, "jlens", "J-lens"), (width / 2, "identity", "J = I")]:
        hits = [cell[arm]["report_hits"] for _, cell in groups]
        print(f"s1 localization 1.5b {arm}: hits {hits} of n=101")
        bars = ax.bar([xi + off for xi in x], hits, width, label=label)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in groups], fontsize=9)
    ax.set_ylabel("report hits (of 101)", fontsize=9)
    ax.set_title("1.5B sub-band steering, α = 24", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig-s1-localization.png", dpi=300)
    plt.close(fig)


def fig_s3_retention() -> None:
    """S3 two-hop retention vs ablation tier, per subject.

    Data: results/s3-selectivity-*.json, two_hop.primary: retention = hits/n over
    the primary cell (chains the subject answered correctly unablated, so the
    unablated baseline is n/n = 1.0 by construction; n = 28/41/43). The matched
    random-direction control at the medium tier is plotted as an open marker.
    """
    tiers = ["light", "medium", "heavy"]
    x = list(range(len(tiers)))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, subject in enumerate(SUBJECTS):
        d = load("s3-selectivity", subject)
        prim = d["two_hop"]["primary"]
        n = prim["jlens_light"]["n"]
        hits = [prim[f"jlens_{t}"]["hits"] for t in tiers]
        rates = [h / n for h in hits]
        rnd = prim["random_medium"]
        print(f"s3 retention {subject}: jlens {hits} of n={n}; random@medium {rnd['hits']}/{rnd['n']}")
        (line,) = ax.plot(x, rates, "o-", label=f"{LABELS[subject]} J-lens (n = {n})")
        ax.plot(1, rnd["hits"] / rnd["n"], "D", mfc="none", color=line.get_color(),
                markersize=8, label=f"{LABELS[subject]} random @ medium")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.text(0.02, 1.02, "unablated baseline (primary cell: all chains correct)",
            fontsize=7, color="gray", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["light\n(first third)", "medium\n(two-thirds)", "heavy\n(full band)"], fontsize=9)
    ax.set_ylabel("two-hop retention (hits / n)", fontsize=9)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig-s3-retention.png", dpi=300)
    plt.close(fig)


def fig_s2_alpha_cliff() -> None:
    """S2 pooled swap successes of 180 at α ∈ {1, 2, 4, 8}, per subject.

    Data: results/s2-generalization-*.json, pooled.unconditional.{jlens,identity}_a{α}
    .successes of n = 180 gradable trials.
    """
    alphas = [1, 2, 4, 8]
    xticks = list(range(len(alphas)))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, subject in zip(axes, SUBJECTS):
        d = load("s2-generalization", subject)
        pooled = d["pooled"]["unconditional"]
        for arm, style, label in [("jlens", "o-", "J-lens"), ("identity", "s--", "J = I")]:
            n = pooled[f"{arm}_a1"]["n"]
            hits = [pooled[f"{arm}_a{a}"]["successes"] for a in alphas]
            print(f"s2 alpha-cliff {subject} {arm}: hits {hits} of n={n}")
            ax.plot(xticks, hits, style, label=label, markersize=4)
        ax.set_title(f"{LABELS[subject]} (n = 180)", fontsize=10)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
        ax.set_xlabel("α", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("swap successes (of 180)", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig-s2-alpha-cliff.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig_s1_dose_response()
    fig_s1_localization()
    fig_s3_retention()
    fig_s2_alpha_cliff()
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
