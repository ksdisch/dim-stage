"""Render the paper's figures from the recorded per-trial results JSONs.

Pure re-rendering: this script reads results/*.json (the project's recorded
measurements) and draws PNGs. It loads no model, fits no lens, and computes
no new statistic — every plotted value and interval is read verbatim from a
results file. Run from the repo root:

    uv run --with matplotlib docs/paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

SUBJECTS = ["0.5b", "1.5b", "3b"]
LABELS = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B"}
COLORS = {"0.5b": "#8da0cb", "1.5b": "#fc8d62", "3b": "#66c2a5"}


def load(stem, subject):
    return json.load(open(ROOT / "results" / f"{stem}-qwen2.5-{subject}-instruct.json"))


def yerr(rate, wilson):
    # clamp at 0: zero-hit cells store a Wilson LB of ~1e-17 above the 0.0 rate
    return [[max(0.0, rate - wilson[0])], [max(0.0, wilson[1] - rate)]]


# ---------------------------------------------------------------- fig 1: M0
def fig1():
    dists = ["association", "poetry", "multihop", "multilingual", "order-ops", "typo"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.26
    for si, subj in enumerate(SUBJECTS):
        d = load("readability", subj)["distributions"]
        for di, dist in enumerate(dists):
            arm1 = d[dist]["arm1"]
            r, w = arm1["pass_at_10"], arm1["wilson_95"]
            x = di + (si - 1) * width
            ax.bar(x, r, width * 0.92, color=COLORS[subj],
                   label=LABELS[subj] if di == 0 else None)
            ax.errorbar(x, r, yerr=yerr(r, w), fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2)
    ax.axhline(0.5, color="crimson", linestyle="--", linewidth=1)
    ax.text(5.62, 0.515, "READS bar\n(Wilson LB ≥ .5)", color="crimson", fontsize=7.5,
            ha="right", va="bottom")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels(dists)
    ax.set_ylabel("J-lens pass@10")
    ax.set_ylim(0, 1)
    ax.set_title("M0 readability: 0/6 distributions reach the bar at any scale")
    ax.legend(title="subject", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-m0-readability.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------- fig 2: S1 dose-response
def fig2():
    d = load("s1-introspection", "1.5b")
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), width_ratios=[3, 2])
    for arm, color, label in [("jlens", "#d95f02", "J-lens (transport)"),
                              ("identity", "#7570b3", "J = I (no transport)")]:
        rates = [d["sweep"][arm][f"{a}"]["report_rate"] for a in alphas]
        los = [d["sweep"][arm][f"{a}"]["wilson_95"][0] for a in alphas]
        his = [d["sweep"][arm][f"{a}"]["wilson_95"][1] for a in alphas]
        xs = range(len(alphas))
        ax1.plot(xs, rates, "-o", color=color, label=label, markersize=4)
        ax1.fill_between(xs, los, his, color=color, alpha=0.15)
    ax1.set_xticks(range(len(alphas)))
    ax1.set_xticklabels([f"{a:g}" for a in alphas])
    ax1.set_xlabel("steering strength α (mean-residual-norm units)")
    ax1.set_ylabel("report rate (rank-1, n=101)")
    ax1.set_title("1.5B injected-thought dose–response")
    ax1.legend(frameon=False, fontsize=8)

    bands = ["L11-15", "L16-20", "L21-24"]
    loc = d["localization"]["sub_bands"]
    full = d["sweep"]
    width = 0.38
    for bi, band in enumerate(bands + ["full L11-24"]):
        if band == "full L11-24":
            jl, idn = full["jlens"]["24.0"], full["identity"]["24.0"]
        else:
            jl, idn = loc[band]["jlens"], loc[band]["identity"]
        for off, cell, color in [(-width / 2, jl, "#d95f02"), (width / 2, idn, "#7570b3")]:
            r, w = cell["report_rate"], cell["wilson_95"]
            ax2.bar(bi + off, r, width * 0.92, color=color)
            ax2.errorbar(bi + off, r, yerr=yerr(r, w), fmt="none", ecolor="black",
                         elinewidth=0.9, capsize=2)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(bands + ["full"], fontsize=8)
    ax2.set_ylabel("report rate at α = 24")
    ax2.set_title("Localization by sub-band third")
    fig.tight_layout()
    fig.savefig(OUT / "fig2-s1-dose-response.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------- fig 3: S2 α cliff
def fig3():
    alphas = ["1", "2", "4", "8"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for subj in SUBJECTS:
        p = load("s2-generalization", subj)["pooled"]["unconditional"]
        rates = [p[f"jlens_a{a}"]["rate"] for a in alphas]
        los = [p[f"jlens_a{a}"]["wilson_95"][0] for a in alphas]
        his = [p[f"jlens_a{a}"]["wilson_95"][1] for a in alphas]
        xs = range(len(alphas))
        ax.plot(xs, rates, "-o", color=COLORS[subj], label=f"{LABELS[subj]} J-lens (n=180)",
                markersize=4)
        ax.fill_between(xs, los, his, color=COLORS[subj], alpha=0.15)
    # paper anchors, recorded in docs/KICKOFF.md and docs/S2-BRIEF.md
    ax.plot([0, 1], [76 / 192, 101 / 192], "s--", color="gray",
            label="paper anchor (Claude-scale, n=192)", markersize=5)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α={a}" for a in alphas])
    ax.set_ylabel("swap-success rate")
    ax.set_title("S2 broadcast: the paper's α=2 rescue inverts at small scale")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-s2-alpha-cliff.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------- fig 4: S3 selectivity
def fig4():
    cells = ["jlens_light", "jlens_medium", "jlens_heavy", "random_medium"]
    names = ["light", "medium", "heavy", "random\n@medium"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    width = 0.26
    for si, subj in enumerate(SUBJECTS):
        d = load("s3-selectivity", subj)
        for ci, cell in enumerate(cells):
            c = d["two_hop"]["primary"][cell]
            r, w = c["rate"], c["wilson_95"]
            x = ci + (si - 1) * width
            ax1.bar(x, r, width * 0.92, color=COLORS[subj],
                    label=LABELS[subj] if ci == 0 else None)
            ax1.errorbar(x, r, yerr=yerr(r, w), fmt="none", ecolor="black",
                         elinewidth=0.9, capsize=2)
        for ci, cell in enumerate(cells):
            c = d["wikitext"][cell]
            r, w = c["rate"], c["wilson_95"]
            x = ci + (si - 1) * width
            ax2.bar(x, r, width * 0.92, color=COLORS[subj])
            ax2.errorbar(x, r, yerr=yerr(r, w), fmt="none", ecolor="black",
                         elinewidth=0.9, capsize=2)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("two-hop retention (primary cell)")
    ax1.set_title("Flexible task under ablation")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("WikiText top-1 match (~11k positions)")
    ax2.set_title("Automatic task under the same ablation")
    fig.tight_layout()
    fig.savefig(OUT / "fig4-s3-selectivity.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------- fig 5: S4b late switch
def fig5():
    conds = ["clean", "primed_late", "control_late"]
    names = ["clean", "primed late\n(concept ablated)", "control late\n(alt. concept ablated)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    width = 0.26
    for si, subj in enumerate(SUBJECTS):
        d = load("s4-avoidance", subj)["naming_success_gated"]
        for ci, cond in enumerate(conds):
            c = d[cond]
            r, w = c["rate"], c["wilson_95"]
            x = ci + (si - 1) * width
            ax.bar(x, r, width * 0.92, color=COLORS[subj],
                   label=f"{LABELS[subj]} (n={c['n']})" if ci == 0 else None)
            ax.errorbar(x, r, yerr=yerr(r, w), fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("naming success (gated cell)")
    ax.set_title("S4b: the late-band off-switch and its matched control")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-s4b-late-switch.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
