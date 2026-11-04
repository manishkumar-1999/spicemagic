"""
EvidenceRx -- Phase 6: tier-corrected re-analysis.

Reads final_database_v3.csv (from evidence_build_v3.py). If that does not exist
yet, falls back to a PRELIMINARY run using the rule tiers (pmid_studytype.csv)
so the pipeline + figure can be validated before the classifier is trained.

Tier direction: 1 synthesis (strongest) ... 6 review (weakest); 0 off-topic
is excluded from all analysis.

Produces:
  E5a. % positive within each tier (assumption-free)
  E5b. Raw vs evidence-weighted % positive (linear / exponential / human-only)
  Human-evidence-only headline (tiers 1-3) vs raw
  Beneficial-spice ranking: raw vs human-only (does it reorder?)
  evidence_pairs.csv                  per-pair table + 'flipped' flag
  evidence_figures/tier_composition.png   stacked bar, top-20 spices (best figure)

Usage:
    python evidence_aggregate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
V3 = BASE / "final_database_v3.csv"
V2 = BASE / "final_database_v2.csv"
RULES = BASE / "pmid_studytype.csv"
OUT_PAIRS = BASE / "evidence_pairs.csv"
FIG_DIR = BASE / "evidence_figures"
FIG = FIG_DIR / "tier_composition.png"

TIER_NAME = {1: "synthesis", 2: "rct", 3: "human-observational",
             4: "animal", 5: "in-vitro", 6: "review"}
# evidence weight: tier 1 strongest -> highest weight
WEIGHTS = {
    "linear":      {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1},
    "exponential": {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1},
    "human_only":  {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0},
}
HUMAN_TIERS = {1, 2, 3}


def pctf(n, d):
    return 100.0 * n / d if d else float("nan")


def load():
    """Return (df with Spice/Disease/label/tier, is_preliminary)."""
    if V3.exists():
        df = pd.read_csv(V3, usecols=lambda c: c in (
            "Spice", "Disease", "Association Nature", "evidence_tier"))
        df["tier"] = pd.to_numeric(df["evidence_tier"], errors="coerce")
        prelim = False
    else:
        df = pd.read_csv(V2, usecols=["PubMed ID", "Spice", "Disease",
                                      "Association Nature"])
        df["_pmid"] = pd.to_numeric(df["PubMed ID"], errors="coerce") \
            .astype("Int64").astype(str)
        rules = pd.read_csv(RULES, dtype={"pmid": str})
        tmap = {r.pmid: (int(r.tier) if str(r.tier) not in ("", "nan") else np.nan)
                for r in rules.itertuples()}
        df["tier"] = df["_pmid"].map(tmap)
        prelim = True
    df["label"] = df["Association Nature"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["tier"])
    df["tier"] = df["tier"].astype(int)
    df = df[df["tier"].between(1, 6)]          # drop off-topic (0) + unknown
    df["is_pos"] = (df["label"] == "positive").astype(int)
    df["is_neg"] = (df["label"] == "negative").astype(int)
    return df, prelim


def main():
    df, prelim = load()
    if prelim:
        print("=" * 64)
        print(" PRELIMINARY RUN -- using RULE tiers (final_database_v3.csv not")
        print(" found). Re-run after the classifier for final numbers.")
        print("=" * 64 + "\n")

    total = len(df)
    print(f"Associations with a tier (off-topic excluded): {total:,}\n")

    # ---- E5a: % positive within each tier ----
    print("=== E5a. % positive WITHIN each tier (no weighting) ===")
    print(f"  {'tier':22s} {'n':>9s} {'%pos(all)':>10s} {'%pos(decided)':>14s}")
    for t in range(1, 7):
        s = df[df.tier == t]
        if len(s):
            pos, neg = s.is_pos.sum(), s.is_neg.sum()
            print(f"  {t} {TIER_NAME[t]:19s} {len(s):9,d} "
                  f"{pctf(pos, len(s)):9.1f}% {pctf(pos, pos+neg):13.1f}%")

    # ---- E5b: raw vs weighted ----
    print("\n=== E5b. Overall % positive: raw vs evidence-weighted ===")
    raw = pctf(df.is_pos.sum(), total)
    print(f"  {'raw (w=1)':16s} {raw:6.1f}%")
    for name, wmap in WEIGHTS.items():
        w = df.tier.map(wmap).astype(float)
        wp = pctf((w * df.is_pos).sum(), w.sum())
        print(f"  {name:16s} {wp:6.1f}%   ({wp-raw:+.1f} pp)")

    # ---- Human-evidence-only headline (tiers 1-3) ----
    h = df[df.tier.isin(HUMAN_TIERS)]
    hraw = pctf(h.is_pos.sum(), len(h))
    print(f"\n  >>> Human evidence only (tiers 1-3): {hraw:.1f}% positive "
          f"(n={len(h):,})  vs raw {raw:.1f}%  [{hraw-raw:+.1f} pp]")

    # ---- Beneficial-spice ranking: raw vs human-only ----
    MIN = 50
    g = df.groupby("Spice")
    raw_rank = (g.is_pos.mean() * 100)[g.size() >= MIN].sort_values(ascending=False)
    hg = h.groupby("Spice")
    hum_rank = (hg.is_pos.mean() * 100)[hg.size() >= MIN].sort_values(ascending=False)
    print(f"\n=== Beneficial-spice ranking (>= {MIN} assoc): raw vs human-only ===")
    print(f"  {'rank':>4s}  {'RAW (all tiers)':28s}  {'HUMAN evidence only':28s}")
    for i in range(min(10, len(raw_rank), len(hum_rank))):
        rs, rv = raw_rank.index[i], raw_rank.iloc[i]
        hs, hv = hum_rank.index[i], hum_rank.iloc[i]
        print(f"  {i+1:>4d}  {rs[:20]:20s} {rv:5.1f}%   {hs[:20]:20s} {hv:5.1f}%")
    dropped = [s for s in raw_rank.index[:15] if s not in hum_rank.index[:15]]
    if dropped:
        print(f"  Drop out of top-15 under human evidence: {', '.join(dropped[:8])}")

    # ---- Per-pair flip analysis (vectorized) ----
    gp = df.groupby(["Spice", "Disease"])
    pair = pd.DataFrame({"n": gp.size(), "pos": gp.is_pos.sum(),
                         "neg": gp.is_neg.sum(), "best_tier": gp.tier.min()})
    hp = h.groupby(["Spice", "Disease"])
    pair["hpos"] = hp.is_pos.sum()
    pair["hneg"] = hp.is_neg.sum()
    pair[["hpos", "hneg"]] = pair[["hpos", "hneg"]].fillna(0)
    pair["raw_net_pos"] = (pair.pos > pair.neg).astype(int)
    pair["has_human"] = (pair.hpos + pair.hneg) > 0
    pair["flipped"] = ((pair.pos > pair.neg) & pair.has_human &
                       (pair.hpos <= pair.hneg)).astype(int)
    pair["best_tier_class"] = pair.best_tier.map(TIER_NAME)
    pair = pair.reset_index().sort_values(["flipped", "n"], ascending=[False, False])
    pair.to_csv(OUT_PAIRS, index=False)
    n_flip = int(pair.flipped.sum())
    n_posp = int(pair.raw_net_pos.sum())
    print(f"\n=== Per-pair flip analysis ===")
    print(f"  Pairs: {len(pair):,} | raw net-positive: {n_posp:,} | "
          f"flipped by human evidence: {n_flip:,} "
          f"({pctf(n_flip, n_posp):.1f}% of net-positive)")

    # ---- Figure: tier-composition stacked bar, top-20 spices ----
    make_figure(df)
    print(f"\nWrote {OUT_PAIRS.name} and {FIG.relative_to(BASE)}")


def make_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    FIG_DIR.mkdir(exist_ok=True)
    top = df.Spice.value_counts().head(20).index[::-1]
    sub = df[df.Spice.isin(top)]
    comp = (pd.crosstab(sub.Spice, sub.tier, normalize="index")
            .reindex(top).reindex(columns=range(1, 7), fill_value=0) * 100)
    # green (strong human) -> grey/red (weak preclinical)
    colors = {1: "#1a7a3a", 2: "#4caf50", 3: "#a5d66a",
              4: "#f4a13c", 5: "#e0673a", 6: "#b0b0b0"}
    fig, ax = plt.subplots(figsize=(11, 8))
    left = np.zeros(len(top))
    for t in range(1, 7):
        vals = comp[t].values
        ax.barh(range(len(top)), vals, left=left, color=colors[t],
                label=f"{t} {TIER_NAME[t]}", edgecolor="white", linewidth=0.4)
        left += vals
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top, fontsize=9)
    ax.set_xlabel("% of associations by evidence tier")
    ax.set_xlim(0, 100)
    ax.set_title("Evidence-tier composition of the top-20 most-studied spices\n"
                 "(green = human evidence, orange/red = preclinical, grey = review)",
                 fontsize=12)
    ax.legend(handles=[Patch(facecolor=colors[t], label=f"{t} {TIER_NAME[t]}")
                       for t in range(1, 7)],
              loc="lower right", fontsize=8, framealpha=0.9)
    ax.axvline(0, color="#333", lw=0.5)
    plt.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
