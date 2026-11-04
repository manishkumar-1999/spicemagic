"""
EvidenceRx -- Active-learning round 2: select the ~400 MOST INFORMATIVE papers
to annotate next (instead of 400 random ones).

Three complementary selection signals:
  A. Weak-class margin cases  -- papers predicted human-observational or
     off-topic (the two classes with fewest gold examples and lowest model
     confidence), chosen by SMALLEST top-2 probability margin.
  B. Global margin cases      -- smallest top-2 margin anywhere (boundaries the
     model has not learned), with a per-class cap for diversity.
  C. Rule-model disagreement  -- the high-precision metadata rule says one tier,
     the model says another. Informative by construction.

All 1,000 existing gold papers are EXCLUDED (no duplicates), and the sheet is
BLIND (model predictions hidden) so round-2 labels stay unbiased.

Inputs : e3_tier_predictions.csv, gold_train/val/test.csv,
         pmid_studytype.csv, articles_cache.json
Outputs: evidence_gold_round2.xlsx        (Guidelines + Annotation sheets)
         evidence_gold_round2_key.csv     (hidden: model pred, margin, bucket)

Usage:
    python evidence_round2_sample.py
    python evidence_round2_sample.py --n 400
"""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

BASE = Path(__file__).parent
PREDS = BASE / "e3_tier_predictions.csv"
RULES = BASE / "pmid_studytype.csv"
ARTICLES = BASE / "articles_cache.json"
GOLD_SPLITS = [BASE / f"gold_{s}.csv" for s in ("train", "val", "test")]
OUT_XLSX = BASE / "evidence_gold_round2.xlsx"
OUT_KEY = BASE / "evidence_gold_round2_key.csv"

SEED = 42
WEAK_CLASSES = {0, 3}          # off-topic, human-observational
PROB_COLS = ["p_0_off-topic", "p_1_synthesis", "p_2_rct",
             "p_3_human-observational", "p_4_animal", "p_5_in-vitro",
             "p_6_review"]
TIERS = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}

TIER_DEFS = [
    (0, "off-topic",
     "The paper is NOT really about a spice/herb health effect at all - "
     "association-mining noise.",
     "plant genomics, food packaging/shelf-life, crop pathology, agronomy, "
     "pure analytical chemistry of a food"),
    (1, "synthesis", "Meta-analysis or systematic review OF HUMAN TRIALS.",
     "\"a meta-analysis of 12 randomized trials...\"; "
     "\"scoping review of randomised controlled trials\""),
    (2, "rct", "Randomized controlled trial in humans.",
     "\"randomized, double-blind, placebo-controlled trial of 120 patients\""),
    (3, "human-observational",
     "Non-randomized HUMAN study: cohort, case-control, cross-sectional, "
     "survey, observational, case report, or non-randomized clinical trial.",
     "\"a survey of 340 patients\"; \"case report of anaphylaxis after...\"; "
     "\"hospital-based case-control study\""),
    (4, "animal", "In-vivo ANIMAL study (any species: mice, rats, sheep, "
     "ostrich, insects, fish...).",
     "\"administered orally to Wistar rats for 8 weeks\"; "
     "\"ostrich hatchlings were fed...\""),
    (5, "in-vitro", "Cell line / biochemical / microbiology assay - no living "
     "organism dosed. Human or animal CELLS in a dish count here.",
     "\"apoptosis in HCT-116 cells at 25 uM\"; \"IC50\"; "
     "\"antibacterial activity against S. mutans\""),
    (6, "review", "Narrative review, opinion/editorial, or IN-SILICO only.",
     "\"this review summarizes...\"; \"molecular docking of piperine\""),
]


def load_gold_pmids():
    seen = set()
    for p in GOLD_SPLITS:
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                seen.add(str(r["pmid"]))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="total papers to select")
    args = ap.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    if not PREDS.exists():
        raise SystemExit(f"Missing {PREDS.name} -- run the Kaggle notebook first.")

    df = pd.read_csv(PREDS, dtype={"pmid": str})
    missing = [c for c in PROB_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Prediction file missing columns: {missing}")

    # --- margin = p(top1) - p(top2); small margin == model is torn ---
    P = df[PROB_COLS].to_numpy()
    part = np.sort(P, axis=1)
    df["margin"] = part[:, -1] - part[:, -2]
    df["pred_tier"] = df["pred_tier"].astype(int)

    # --- exclude papers already annotated ---
    gold = load_gold_pmids()
    before = len(df)
    df = df[~df["pmid"].isin(gold)].copy()
    print(f"Candidates: {len(df):,}  (excluded {before-len(df):,} existing gold)")

    # --- rule tiers, for the disagreement bucket ---
    rules = {}
    for r in csv.DictReader(open(RULES, encoding="utf-8")):
        if r["tier"]:
            rules[r["pmid"]] = (int(r["tier"]), r["rule"])
    df["rule_tier"] = df["pmid"].map(lambda p: rules.get(p, (None, ""))[0])
    df["rule"] = df["pmid"].map(lambda p: rules.get(p, (None, ""))[1])

    n_weak = int(args.n * 0.50)      # 200 of 400
    n_margin = int(args.n * 0.30)    # 120
    n_disagree = args.n - n_weak - n_margin   # 80

    picked, buckets = {}, {}

    def take(sub, k, bucket):
        got = 0
        for pmid in sub["pmid"]:
            if got >= k:
                break
            if pmid in picked:
                continue
            picked[pmid] = bucket
            buckets[bucket] = buckets.get(bucket, 0) + 1
            got += 1
        return got

    # A. weak-class boundary cases
    a = df[df.pred_tier.isin(WEAK_CLASSES)].sort_values("margin")
    take(a, n_weak, "A_weak_class_margin")

    # B. global boundary cases, capped per predicted class for diversity
    cap = max(1, int(n_margin * 0.35))
    per = Counter()
    b_rows = []
    for row in df.sort_values("margin").itertuples():
        if row.pmid in picked:
            continue
        if per[row.pred_tier] >= cap:
            continue
        b_rows.append(row.pmid)
        per[row.pred_tier] += 1
        if len(b_rows) >= n_margin:
            break
    for pmid in b_rows:
        picked[pmid] = "B_global_margin"
        buckets["B_global_margin"] = buckets.get("B_global_margin", 0) + 1

    # C. rule vs model disagreement (prefer high-precision metadata rules)
    c = df[df.rule_tier.notna() & (df.rule_tier != df.pred_tier)].copy()
    c["is_meta_rule"] = c["rule"].str.startswith("M").astype(int)
    c = c.sort_values(["is_meta_rule", "margin"], ascending=[False, True])
    take(c, n_disagree, "C_rule_model_disagree")

    # backfill if any bucket came up short
    if len(picked) < args.n:
        rest = df[~df.pmid.isin(picked)].sort_values("margin")
        take(rest, args.n - len(picked), "D_backfill_margin")

    sel = df[df.pmid.isin(picked)].copy()
    sel["bucket"] = sel["pmid"].map(picked)

    # --- write hidden key ---
    key = sel[["pmid", "pred_tier", "pred_class", "confidence", "margin",
               "rule_tier", "rule", "bucket"]].copy()
    key.columns = ["pmid", "model_tier", "model_class", "model_conf",
                   "margin", "rule_tier", "rule", "bucket"]
    key.to_csv(OUT_KEY, index=False)

    # --- write blind annotation workbook ---
    arts = json.load(open(ARTICLES, encoding="utf-8"))
    order = list(sel["pmid"])
    random.shuffle(order)

    wb = Workbook()
    ws = wb.active
    ws.title = "Guidelines"
    ws["A1"] = "EvidenceRx - Round 2 (active learning) Annotation Guidelines"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A3"] = ("These are the HARD cases - papers the model was least sure "
                "about, plus cases where the rules and the model disagree. "
                "Expect them to be trickier than round 1. Read title + "
                "abstract, then put ONE tier number (0-6) in gold_tier. Judge "
                "the study the paper itself performed (a review is tier 6 even "
                "if it discusses trials).")
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:F6")
    ws.append([])
    ws.append(["tier", "class", "definition", "example"])
    for c_ in ws[ws.max_row]:
        c_.font = Font(bold=True)
    for t, name, d, ex in TIER_DEFS:
        ws.append([t, name, d, ex])
    ws.column_dimensions["C"].width = 58
    ws.column_dimensions["D"].width = 52
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 20

    an = wb.create_sheet("Annotation")
    an.append(["row_id", "pmid", "title", "abstract", "gold_tier", "notes"])
    for c_ in an[1]:
        c_.font = Font(bold=True, color="FFFFFF")
        c_.fill = PatternFill("solid", fgColor="2E75B6")
        c_.alignment = Alignment(horizontal="center")
    for i, pmid in enumerate(order, 1):
        art = arts.get(pmid, {}) or {}
        ab = (art.get("abstract") or "").strip()
        if len(ab) > 2000:
            ab = ab[:2000] + " ..."
        an.append([i, pmid, (art.get("title") or "").strip(), ab, "", ""])
    an.column_dimensions["C"].width = 55
    an.column_dimensions["D"].width = 90
    an.column_dimensions["E"].width = 10
    an.column_dimensions["F"].width = 25
    an.freeze_panes = "A2"
    for row in an.iter_rows(min_row=2):
        row[2].alignment = Alignment(wrap_text=True, vertical="top")
        row[3].alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(OUT_XLSX)

    # --- report ---
    print(f"\nSelected {len(sel)} papers for round-2 annotation.")
    print("\n=== Selection buckets ===")
    for b, n in sorted(buckets.items()):
        print(f"  {b:24s} {n:4d}")
    print("\n=== Model's predicted class for the selected papers ===")
    for t, n in sorted(Counter(sel.pred_tier).items()):
        print(f"  {TIERS[t]:22s} {n:4d}")
    print(f"\nMean top-2 margin of selection: {sel.margin.mean():.3f} "
          f"(vs {df.margin.mean():.3f} for all candidates)")
    print(f"Mean model confidence of selection: {sel.confidence.mean():.3f}")
    print(f"\nWrote {OUT_XLSX.name}  (label the 'gold_tier' column, 0-6)")
    print(f"Wrote {OUT_KEY.name}   (hidden model predictions, for later analysis)")


if __name__ == "__main__":
    main()
