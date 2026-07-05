"""Where the models win and lose — Brier decomposition of the holdout.

Reads holdout_predictions_v3.csv (per-row holdout predictions written by
train_model_v3.py) and decomposes Brier scores by park, lineup slot,
batter handedness, opposing-starter HR-proneness quartile, and month.
Writes error_analysis.png and prints the markdown tables that back the
findings in EVALUATION.md. Pure post-processing: no model is trained or
scored here, so the numbers are exactly the holdout numbers.
"""

import numpy as np
import pandas as pd

PRED_PATH = "holdout_predictions_v3.csv"
FIG_PATH = "error_analysis.png"
MODELS = [("gbm_v3", "GBM v3"), ("structural_v2", "Structural v2"), ("base", "Base rate")]


def group_table(d, key):
    """Per-group n, actual rate, mean pred and Brier for each model, plus
    v3's Brier edge over structural v2 and over the base rate (negative =
    v3 better)."""
    def agg(g):
        out = {"n": len(g), "actual": g["is_hr_game"].mean()}
        for col, _ in MODELS:
            out[f"pred_{col}"] = g[col].mean()
            out[f"brier_{col}"] = ((g[col] - g["is_hr_game"]) ** 2).mean()
        out["v3_vs_v2"] = out["brier_gbm_v3"] - out["brier_structural_v2"]
        out["v3_vs_base"] = out["brier_gbm_v3"] - out["brier_base"]
        return pd.Series(out)
    return d.groupby(key, observed=True).apply(agg, include_groups=False)


def load():
    d = pd.read_csv(PRED_PATH, parse_dates=["game_date"])
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    d["slot_group"] = np.where(d["slot"].isna(), "sub (unknown)",
                       np.where(d["slot"] <= 3, "slots 1-3",
                       np.where(d["slot"] <= 6, "slots 4-6", "slots 7-9")))
    d["pf_quartile"] = pd.qcut(d["s_pitcher_factor"], 4,
                               labels=["Q1 (low-HR)", "Q2", "Q3", "Q4 (HR-prone)"])
    return d


def to_md(tbl, index_name):
    cols = ["n", "actual", "pred_gbm_v3", "pred_structural_v2",
            "brier_gbm_v3", "brier_structural_v2", "brier_base", "v3_vs_base"]
    hdr = [index_name, "n", "Actual", "v3 pred", "v2 pred",
           "v3 Brier", "v2 Brier", "Base Brier", "v3−base"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for idx, r in tbl[cols].iterrows():
        lines.append(
            f"| {idx} | {int(r['n']):,} | {r['actual']:.4f} | {r['pred_gbm_v3']:.4f} "
            f"| {r['pred_structural_v2']:.4f} | {r['brier_gbm_v3']:.5f} "
            f"| {r['brier_structural_v2']:.5f} | {r['brier_base']:.5f} "
            f"| {r['v3_vs_base']:+.5f} |")
    return "\n".join(lines)


def main():
    d = load()
    park = group_table(d, "home_team").sort_values("v3_vs_base")
    slot = group_table(d, "slot_group")
    hand = group_table(d, "stand")
    pfq = group_table(d, "pf_quartile")
    month = group_table(d, "month")

    for name, tbl, idx in [
        ("PARK — v3's five best and five worst (skill vs base rate)",
         pd.concat([park.head(5), park.tail(5)]), "Park"),
        ("LINEUP SLOT", slot, "Slot"),
        ("BATTER HAND", hand, "Stand"),
        ("OPPOSING-STARTER HR-PRONENESS QUARTILE", pfq, "Quartile"),
        ("MONTH (drift check)", month, "Month"),
    ]:
        print(f"\n### {name}\n")
        print(to_md(tbl, idx))

    # ── Figure ────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    c_v3, c_v2, c_act = "#1f6f43", "#b3403a", "#444444"

    ax = axes[0, 0]
    sel = pd.concat([park.head(5), park.tail(5)])
    colors = [c_v3 if v < 0 else c_v2 for v in sel["v3_vs_base"]]
    ax.barh(range(len(sel)), sel["v3_vs_base"] * 1000, color=colors)
    ax.set_yticks(range(len(sel)), sel.index)
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("GBM v3 Brier − base-rate Brier (×1000; negative = v3 better)")
    ax.set_title("Park: v3's five best and five worst")

    ax = axes[0, 1]
    x = np.arange(len(slot))
    w = 0.27
    ax.bar(x - w, slot["pred_gbm_v3"], w, label="v3 mean pred", color=c_v3)
    ax.bar(x, slot["pred_structural_v2"], w, label="v2 mean pred", color=c_v2)
    ax.bar(x + w, slot["actual"], w, label="actual rate", color=c_act)
    ax.set_xticks(x, slot.index, rotation=12)
    ax.set_ylabel("P(HR in game)")
    ax.set_title("Lineup slot: the formula overprices substitutes 2.3×")
    ax.legend()

    ax = axes[1, 0]
    x = np.arange(len(pfq))
    ax.bar(x - 0.2, pfq["v3_vs_base"] * 1000, 0.4, label="v3 − base", color=c_v3)
    ax.bar(x + 0.2, (pfq["brier_structural_v2"] - pfq["brier_base"]) * 1000, 0.4,
           label="v2 − base", color=c_v2)
    ax.set_xticks(x, pfq.index)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Brier − base (×1000)")
    ax.set_title("Starter HR-proneness: v3's edge grows with pitcher risk")
    ax.legend()

    ax = axes[1, 1]
    m = month[month["n"] >= 1000]  # drop the 2-day July stub
    ax.plot(m.index, m["v3_vs_base"] * 1000, "o-", color=c_v3, label="v3 − base")
    ax.plot(m.index, (m["brier_structural_v2"] - m["brier_base"]) * 1000, "s-",
            color=c_v2, label="v2 − base")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Brier − base (×1000)")
    ax.set_title("Month: no drift — v3's edge widens into summer")
    ax.legend()

    fig.suptitle("Holdout error analysis (2026-03-23 .. 2026-07-02, n = 27,258 player-games)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)
    print(f"\nSaved {FIG_PATH}")


if __name__ == "__main__":
    main()
