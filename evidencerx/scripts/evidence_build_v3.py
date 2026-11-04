"""
EvidenceRx -- Phase 5: build final_database_v3.csv.

Attaches the model's evidence tier to every association (broadcast from the
paper), plus tier confidence, an off-topic flag, and -- for transparency -- the
original rule tier so rule/model agreement can be audited.

Run the Kaggle notebook and download e3_tier_predictions.csv first.

Inputs:
    final_database_v2.csv          (228,750 associations)
    e3_tier_predictions.csv        (per-paper model tier + confidence)
    pmid_studytype.csv             (rule tier, for the agreement column)
Output:
    final_database_v3.csv          (v2 columns + evidence_tier, tier_class,
                                    tier_confidence, is_offtopic, rule_tier)

Usage:
    python evidence_build_v3.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
V2 = BASE_DIR / "final_database_v2.csv"
PREDS = BASE_DIR / "e3_tier_predictions.csv"
RULES = BASE_DIR / "pmid_studytype.csv"
V3 = BASE_DIR / "final_database_v3.csv"

CLASSES = ["off-topic", "synthesis", "rct", "human-observational",
           "animal", "in-vitro", "review"]


def norm_pmid(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64").astype(str)


def main():
    if not PREDS.exists():
        sys.exit(f"Missing {PREDS.name}. Run the Kaggle tier-classifier "
                 f"notebook and download it first.")

    df = pd.read_csv(V2)
    df["_pmid"] = norm_pmid(df["PubMed ID"])

    preds = pd.read_csv(PREDS, dtype={"pmid": str})
    preds["pmid"] = norm_pmid(preds["pmid"])
    pmap = preds.set_index("pmid")
    tier_of = pmap["pred_tier"].to_dict()
    conf_of = pmap["confidence"].to_dict()

    rules = pd.read_csv(RULES, dtype={"pmid": str})
    rules["pmid"] = norm_pmid(rules["pmid"])
    rule_tier = rules.set_index("pmid")["tier"].to_dict()

    df["evidence_tier"] = df["_pmid"].map(tier_of)
    df["tier_confidence"] = df["_pmid"].map(conf_of).round(4)
    df["rule_tier"] = df["_pmid"].map(rule_tier)
    df["tier_class"] = df["evidence_tier"].map(
        lambda t: CLASSES[int(t)] if pd.notna(t) else "")
    df["is_offtopic"] = (df["evidence_tier"] == 0).astype("Int64")

    covered = df["evidence_tier"].notna().sum()
    df.drop(columns=["_pmid"]).to_csv(V3, index=False)

    print(f"Associations: {len(df):,}")
    print(f"With an evidence tier: {covered:,} ({covered/len(df)*100:.1f}%)")
    print(f"Off-topic (tier 0): {(df['evidence_tier']==0).sum():,} "
          f"({(df['evidence_tier']==0).mean()*100:.1f}%)")
    print("\n=== Evidence-tier distribution over associations ===")
    for t in range(0, 7):
        n = (df["evidence_tier"] == t).sum()
        if n:
            print(f"  tier {t} {CLASSES[t]:20s} {n:8,d}  ({n/len(df)*100:5.1f}%)")
    # rule vs model agreement (on-topic, both present)
    m = df.dropna(subset=["evidence_tier", "rule_tier"])
    m = m[m["evidence_tier"] != 0]
    if len(m):
        agree = (m["evidence_tier"].astype(int) == m["rule_tier"].astype(int)).mean()
        print(f"\nRule vs model agreement (on-topic assoc): {agree*100:.1f}%")
    print(f"\nWrote {V3.name}")


if __name__ == "__main__":
    main()
