"""derived_contrasts.py — recompute the paper statistics that were recorded only in prose.

Run (from the repo root):

    uv run docs/paper/derived_contrasts.py

A handful of statistics cited by the paper live in the docs spine (DECISIONS.md,
S1-BRIEF.md, ROADMAP.md) but not in any per-run result file — they were derived
during a stage from counts that ARE in the result files. This script re-derives
each one via the repo's own stats.py from counts parsed out of results/*.json,
asserts it equals the prose-recorded value exactly (at the recorded precision),
and writes results/derived-contrasts.json so the paper can cite a result file
for them. If any assertion fails, nothing is written — a mismatch is a finding,
not something to paper over.

No models, no lenses, no re-measurement: inputs are committed counts only.
Run-metadata facts (fit wall-clocks, the AGREE gate, the $0.83 rental) are
deliberately NOT here — those are legitimately prose-recorded facts of runs.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # docs/paper/
REPO = HERE.parents[1]                          # repo root
sys.path.insert(0, str(REPO))

from stats import newcombe_diff  # noqa: E402  (path set up above)

RESULTS = REPO / "results"


def load(name: str) -> dict:
    with open(RESULTS / name) as f:
        return json.load(f)


def rounded(interval: tuple[float, float, float]) -> tuple[float, float, float]:
    """Round (d, lo, hi) to the 3-decimal precision the docs spine records."""
    return tuple(round(x, 3) for x in interval)


def main() -> None:
    out: dict = {
        "derived_by": "docs/paper/derived_contrasts.py — stats.py (newcombe_diff) and "
                      "statistics.median over counts/ranks parsed from results/*.json",
        "notes": [
            "Every value below was asserted equal to its prose-recorded form (at the "
            "recorded precision) before this file was written.",
            "M1 introspection medians use the 'default' prefill arm of "
            "results/introspection-*.json (concepts[].ranks.default) — the arm whose "
            "medians match the recorded 3747→1322 / 4430→15 / 3791→382 exactly.",
            "The '151,936' in the S2 median-rank sentence is the tokenizer vocabulary "
            "size (the worst possible rank) — prose-recorded context, not recomputed here.",
        ],
        "contrasts": {},
    }

    # ---- M3 scale growth: J-lens category-focus, 3B (9/110) vs 0.5B (2/110) ----
    foc05 = load("directed-modulation-qwen2.5-0.5b-instruct.json")["cells"]["category"]["focus"]["jlens"]
    foc3 = load("directed-modulation-qwen2.5-3b-instruct.json")["cells"]["category"]["focus"]["jlens"]
    assert (foc05["hits"], foc05["n"], foc3["hits"], foc3["n"]) == (2, 110, 9, 110)
    m3 = newcombe_diff(foc05["hits"], foc05["n"], foc3["hits"], foc3["n"])
    assert rounded(m3) == (0.064, 0.004, 0.131), rounded(m3)
    out["contrasts"]["m3_focus_scale_growth_3b_minus_0.5b"] = {
        "counts": {"base_0.5b": [foc05["hits"], foc05["n"]], "mech_3b": [foc3["hits"], foc3["n"]]},
        "interval": {"d": m3[0], "newcombe_95": [m3[1], m3[2]]},
        "derived_by": "stats.newcombe_diff",
        "matches_prose_source": "docs/DECISIONS.md (M3 outcomes): +.064 [+.004, +.131]",
    }

    # ---- S1 1.5B localization: J − I per sub-band, and full − mid (J-lens) ----
    loc = load("s1-introspection-qwen2.5-1.5b-instruct.json")["localization"]
    expected = {
        "L11-15": (0.040, -0.060, 0.139),
        "L16-20": (0.129, 0.014, 0.240),
        "L21-24": (0.129, 0.041, 0.219),
    }
    for band, exp in expected.items():
        j = loc["sub_bands"][band]["jlens"]
        i = loc["sub_bands"][band]["identity"]
        d = newcombe_diff(i["report_hits"], i["n"], j["report_hits"], j["n"])
        assert rounded(d) == exp, (band, rounded(d))
        out["contrasts"][f"s1_1.5b_localization_j_minus_i_{band}"] = {
            "counts": {"base_identity": [i["report_hits"], i["n"]], "mech_jlens": [j["report_hits"], j["n"]]},
            "interval": {"d": d[0], "newcombe_95": [d[1], d[2]]},
            "derived_by": "stats.newcombe_diff",
            "matches_prose_source": f"docs/S1-BRIEF.md (results): "
                                    f"{exp[0]:+.3f} [{exp[1]:+.3f}, {exp[2]:+.3f}]".replace("0.", "."),
        }
    full = loc["full_band"]["jlens"]
    mid = loc["sub_bands"]["L16-20"]["jlens"]
    d = newcombe_diff(mid["report_hits"], mid["n"], full["report_hits"], full["n"])
    assert rounded(d) == (0.020, -0.105, 0.144), rounded(d)
    out["contrasts"]["s1_1.5b_localization_full_minus_mid_jlens"] = {
        "counts": {"base_mid_L16-20": [mid["report_hits"], mid["n"]], "mech_full_L11-24": [full["report_hits"], full["n"]]},
        "interval": {"d": d[0], "newcombe_95": [d[1], d[2]]},
        "derived_by": "stats.newcombe_diff",
        "matches_prose_source": "docs/S1-BRIEF.md (results): +.020 [-.105, +.144]",
    }

    # ---- S2 1.5B: target-answer median rank at α = 2 over the α = 1 J-lens hits ----
    trials = load("s2-generalization-qwen2.5-1.5b-instruct.json")["trials"]
    ranks = [t["swaps"]["jlens_a2"]["rank"] for t in trials if t["swaps"]["jlens_a1"]["success"]]
    med = statistics.median(ranks)
    assert len(ranks) == 16 and med == 151844.5, (len(ranks), med)
    out["contrasts"]["s2_1.5b_a2_target_rank_median_over_a1_jlens_hits"] = {
        "counts": {"n_a1_jlens_hits": len(ranks), "ranks": sorted(ranks)},
        "interval": {"median": med},
        "derived_by": "statistics.median over trials[].swaps.jlens_a2.rank where trials[].swaps.jlens_a1.success",
        "matches_prose_source": "docs/S2-BRIEF.md '~151,845' (rounds up) and docs/ROADMAP.md / "
                                "docs/LEARNING.md '~151,844' (round down); raw median 151844.5 — "
                                "see the dated erratum in docs/S2-BRIEF.md",
    }

    # ---- M1 introspection: median steered rank, control (α=0) → α=8, default prefill ----
    expected_med = {"0.5b": (3747, 1322), "1.5b": (4430, 15), "3b": (3791, 382)}
    for subject, (e0, e8) in expected_med.items():
        concepts = load(f"introspection-qwen2.5-{subject}-instruct.json")["concepts"]
        m0 = statistics.median(c["ranks"]["default"]["0.0"] for c in concepts)
        m8 = statistics.median(c["ranks"]["default"]["8.0"] for c in concepts)
        assert (m0, m8) == (e0, e8), (subject, m0, m8)
        out["contrasts"][f"m1_{subject}_median_steered_rank_control_to_a8"] = {
            "counts": {"n_concepts": len(concepts)},
            "interval": {"median_rank_a0": m0, "median_rank_a8": m8},
            "derived_by": "statistics.median over concepts[].ranks.default at α = 0 and α = 8",
            "matches_prose_source": f"docs/DECISIONS.md / docs/M1-BRIEF.md: {e0} → {e8}",
        }

    out_path = RESULTS / "derived-contrasts.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"all assertions passed; wrote {out_path}")


if __name__ == "__main__":
    main()
