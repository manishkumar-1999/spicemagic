"""
EvidenceRx -- Phase 7: controversy scoring and evidence-conflict motifs.

Two different questions, deliberately kept apart:

  1. CONTROVERSY -- does the literature on a pair disagree with itself?
     Measured as the entropy of the positive/negative split. 1.0 = a perfect
     50/50 split, 0.0 = unanimous. Reported raw and tier-weighted, because a
     split between two RCTs matters more than a split between two cell assays.

  2. MOTIFS -- *how* does it disagree? The one that matters for this project
     is PRECLINICAL HYPE: animal/in-vitro work says "beneficial", human work
     does not. A raw association count cannot see this; a tiered one can.
     The reverse motif (human-positive, preclinical-negative) is reported too,
     as a check that the first is not just an artefact of the scoring.

Reads : final_database_v3.csv (needs evidence_tier; falls back to v2 + rule
        tiers with a loud warning, so the script is testable before the
        classifier has run)
Writes: controversy_pairs.csv
        evidence_figures/preclinical_hype.png

Usage:
    python evidence_controversy.py
    python evidence_controversy.py --min-n 10 --top 25
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
V3 = BASE / "final_database_v3.csv"
V2 = BASE / "final_database_v2.csv"
RULES = BASE / "pmid_studytype.csv"
OUT = BASE / "controversy_pairs.csv"
FIG_DIR = BASE / "evidence_figures"
FIG = FIG_DIR / "preclinical_hype.png"

TIER_NAME = {1: "synthesis", 2: "rct", 3: "human-observational",
             4: "animal", 5: "in-vitro", 6: "review"}
HUMAN = {1, 2, 3}
PRECLIN = {4, 5}

# Disease terms that are METHOD words, not clinical outcomes. Spot-checking the
# first run showed "Licorice / toxicity (positive)" was papers titled
# "Antiplasmodial activity of ethanolic extracts..." -- the term was mined from
# the word *cytotoxicity* in the methods section. Left in, these artifacts
# dominate every controversy ranking, because an assay word co-occurs with both
# positive and negative findings at random.
GENERIC_TERMS = {
    "toxicity", "cytotoxicity", "genotoxicity", "hepatotoxicity",
    "nephrotoxicity", "neurotoxicity", "cardiotoxicity", "phytotoxicity",
    "ototoxicity", "immunotoxicity", "apoptosis", "necrosis", "cell death",
    "oxidative stress", "inflammation", "inflammatory", "infection",
    "infections", "disease", "diseases", "disorder", "disorders", "syndrome",
    "injury", "lesion", "lesions", "damage", "pain", "stress", "symptoms",
    "complications", "mortality", "death", "deaths", "side effects",
    "adverse effects", "resistance", "tumor", "tumors", "tumour", "tumours",
    # adjective forms slip through the noun list and were still ranking top-12
    "toxic", "cytotoxic", "genotoxic", "hepatotoxic", "neurotoxic",
    "carcinogenic", "mutagenic", "antioxidant", "proliferation",
}
# same linear scale as evidence_aggregate.py -- tier 1 strongest
WEIGHT = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}


def entropy(pos, neg):
    """Binary entropy of the pos/neg split; 0 when unanimous or empty."""
    tot = pos + neg
    out = np.zeros(len(tot), dtype=float)
    ok = tot > 0
    p = np.divide(pos, tot, out=np.zeros_like(out), where=ok)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    out[ok] = np.nan_to_num(h[ok])          # p==0 or 1 -> 0, not nan
    return out


def load():
    if V3.exists():
        df = pd.read_csv(V3, usecols=lambda c: c in (
            "Spice", "Disease", "Association Nature", "evidence_tier"))
        df["tier"] = pd.to_numeric(df["evidence_tier"], errors="coerce")
        prelim = False
    else:
        print("=" * 64)
        print(" final_database_v3.csv missing -- PRELIMINARY run on rule tiers.")
        print("=" * 64)
        df = pd.read_csv(V2, usecols=["PubMed ID", "Spice", "Disease",
                                      "Association Nature"])
        pm = pd.to_numeric(df["PubMed ID"], errors="coerce").astype("Int64")
        rules = pd.read_csv(RULES, dtype={"pmid": str})
        tmap = {r.pmid: (int(r.tier) if str(r.tier) not in ("", "nan")
                         else np.nan) for r in rules.itertuples()}
        df["tier"] = pm.astype(str).map(tmap)
        prelim = True
    lab = df["Association Nature"].astype(str).str.strip().str.lower()
    df["is_pos"] = (lab == "positive").astype(int)
    df["is_neg"] = (lab == "negative").astype(int)
    df = df.dropna(subset=["tier"])
    df["tier"] = df["tier"].astype(int)
    df = df[df["tier"].between(1, 6)]        # off-topic (0) excluded

    # drop method-word "diseases" and self-pairs (e.g. Chicory / chicory)
    dl = df["Disease"].astype(str).str.strip().str.lower()
    sl = df["Spice"].astype(str).str.strip().str.lower()
    before = len(df)
    df = df[~dl.isin(GENERIC_TERMS) & (dl != sl)]
    print(f"Dropped {before-len(df):,} rows on generic/self disease terms "
          f"({100*(before-len(df))/max(before,1):.1f}%)")
    return df, prelim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=10,
                    help="minimum decided associations per pair")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    df, prelim = load()
    df["w"] = df["tier"].map(WEIGHT)
    df["decided"] = df.is_pos + df.is_neg
    df = df[df.decided == 1]                 # neutral rows carry no direction
    print(f"Decided associations with a tier: {len(df):,}\n")

    g = df.groupby(["Spice", "Disease"])
    p = pd.DataFrame({
        "n": g.size(),
        "pos": g.is_pos.sum(),
        "neg": g.is_neg.sum(),
        "w_pos": g.apply(lambda x: (x.w * x.is_pos).sum(), include_groups=False),
        "w_tot": g.w.sum(),
        "best_tier": g.tier.min(),
    })
    for name, mask in (("hum", HUMAN), ("pre", PRECLIN)):
        s = df[df.tier.isin(mask)].groupby(["Spice", "Disease"])
        p[f"{name}_n"] = s.size()
        p[f"{name}_pos"] = s.is_pos.sum()
    p[["hum_n", "hum_pos", "pre_n", "pre_pos"]] = \
        p[["hum_n", "hum_pos", "pre_n", "pre_pos"]].fillna(0).astype(int)

    p["pct_pos"] = 100 * p.pos / p.n
    p["w_pct_pos"] = 100 * p.w_pos / p.w_tot
    p["controversy"] = entropy(p.pos.values, p.neg.values)
    # tier-weighted: split the WEIGHT mass, not the paper count
    p["w_controversy"] = entropy(p.w_pos.values, (p.w_tot - p.w_pos).values)
    p["hum_pct_pos"] = np.where(p.hum_n > 0, 100 * p.hum_pos / p.hum_n, np.nan)
    p["pre_pct_pos"] = np.where(p.pre_n > 0, 100 * p.pre_pos / p.pre_n, np.nan)
    p["hype_gap"] = p.pre_pct_pos - p.hum_pct_pos

    p = p[p.n >= args.min_n].reset_index()
    print(f"Pairs with >= {args.min_n} decided associations: {len(p):,}")

    # ---- 1. most controversial ----
    print(f"\n=== Most controversial pairs (tier-weighted entropy) ===")
    print(f"  {'spice':16s} {'disease':24s} {'n':>5s} {'%pos':>6s} "
          f"{'H':>5s} {'H_w':>5s}")
    for r in p.sort_values("w_controversy", ascending=False).head(args.top).itertuples():
        print(f"  {r.Spice[:15]:16s} {r.Disease[:23]:24s} {r.n:5d} "
              f"{r.pct_pos:5.0f}% {r.controversy:5.2f} {r.w_controversy:5.2f}")

    # ---- 2. preclinical hype motif ----
    both = p[(p.hum_n >= 3) & (p.pre_n >= 3)].copy()
    hype = both[both.hype_gap > 0].sort_values("hype_gap", ascending=False)
    rev = both[both.hype_gap < 0].sort_values("hype_gap")
    print(f"\n=== PRECLINICAL HYPE: preclinical says yes, humans do not ===")
    print(f"  (pairs with >=3 human AND >=3 preclinical decided associations: "
          f"{len(both):,})")
    print(f"  {'spice':16s} {'disease':22s} {'pre%':>6s} {'hum%':>6s} "
          f"{'gap':>6s} {'n_pre':>6s} {'n_hum':>6s}")
    for r in hype.head(args.top).itertuples():
        print(f"  {r.Spice[:15]:16s} {r.Disease[:21]:22s} "
              f"{r.pre_pct_pos:5.0f}% {r.hum_pct_pos:5.0f}% "
              f"{r.hype_gap:+5.0f}  {r.pre_n:6d} {r.hum_n:6d}")

    print(f"\n=== Reverse motif (humans more positive than preclinical) ===")
    for r in rev.head(5).itertuples():
        print(f"  {r.Spice[:15]:16s} {r.Disease[:21]:22s} "
              f"{r.pre_pct_pos:5.0f}% {r.hum_pct_pos:5.0f}% {r.hype_gap:+5.0f}")

    n_hype, n_rev = len(hype), len(rev)
    print(f"\n  hype pairs {n_hype:,} vs reverse {n_rev:,} "
          f"({100*n_hype/max(n_hype+n_rev,1):.0f}% of directional pairs lean "
          f"preclinical-optimistic)")
    print(f"  mean gap: {both.hype_gap.mean():+.1f} pp")

    p.sort_values("w_controversy", ascending=False).to_csv(OUT, index=False)
    make_figure(hype.head(15))
    print(f"\nWrote {OUT.name} and {FIG.relative_to(BASE)}")
    if prelim:
        print("\nPRELIMINARY (rule tiers) -- re-run after evidence_build_v3.py")


def make_figure(top):
    if top.empty:
        print("(no hype pairs to plot)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    top = top.iloc[::-1]
    lbl = [f"{r.Spice[:18]} / {r.Disease[:24]}" for r in top.itertuples()]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(y + 0.19, top.pre_pct_pos, 0.38, color="#e0673a",
            label="preclinical (animal + in-vitro)")
    ax.barh(y - 0.19, top.hum_pct_pos, 0.38, color="#1a7a3a",
            label="human (synthesis + RCT + observational)")
    ax.set_yticks(y)
    ax.set_yticklabels(lbl, fontsize=8)
    ax.set_xlabel("% of decided associations reported as beneficial")
    ax.set_xlim(0, 100)
    ax.set_title("Preclinical hype: pairs where lab evidence is far more\n"
                 "optimistic than human evidence", fontsize=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
