"""
EvidenceRx -- apply the round-2 annotations (AI-drafted, second-annotator pass)
to evidence_gold_round2.xlsx and report what the model got wrong on its own
hardest cases.

Reads : r2_labels_part1..4.csv   (row_id, tier)
        evidence_gold_round2.xlsx      (blind sheet, row_id/pmid/title/abstract)
        evidence_gold_round2_key.csv   (hidden model prediction + bucket)
Writes: evidence_gold_round2_annotated.xlsx   (gold_tier filled in)
        evidence_round2_labels.csv            (pmid, gold_tier, annotator)

Usage:
    python evidence_round2_apply_labels.py
"""

import csv
from collections import Counter
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

BASE = Path(__file__).parent
PARTS = [BASE / f"r2_labels_part{i}.csv" for i in (1, 2, 3, 4)]
XLSX_IN = BASE / "evidence_gold_round2.xlsx"
XLSX_OUT = BASE / "evidence_gold_round2_annotated.xlsx"
KEY = BASE / "evidence_gold_round2_key.csv"
OUT_CSV = BASE / "evidence_round2_labels.csv"

TIERS = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}
ANNOTATOR = "claude-2nd-annotator"   # AI-drafted, human-reviewed


def main():
    labels = {}
    for p in PARTS:
        if not p.exists():
            raise SystemExit(f"Missing {p.name}")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            labels[int(r["row_id"])] = int(r["tier"])
    print(f"Loaded {len(labels)} labels from {len(PARTS)} batch files")

    wb = load_workbook(XLSX_IN)
    ws = wb["Annotation"]
    head = [c.value for c in ws[1]]
    i_row, i_pmid, i_gold = (head.index("row_id"), head.index("pmid"),
                             head.index("gold_tier"))

    rows, missing = [], 0
    for r in ws.iter_rows(min_row=2):
        rid = r[i_row].value
        if rid is None:
            continue
        t = labels.get(int(rid))
        if t is None:
            missing += 1
            continue
        r[i_gold].value = t
        rows.append((str(r[i_pmid].value), t))
    if missing:
        print(f"WARNING: {missing} sheet rows had no label")

    # CSV first: it is what the rest of the pipeline consumes, so an .xlsx
    # locked open in Excel must never block the run.
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "gold_tier", "annotator"])
        for pmid, t in rows:
            w.writerow([pmid, t, ANNOTATOR])
    print(f"Wrote {OUT_CSV.name}  ({len(rows)} papers)")

    try:
        wb.save(XLSX_OUT)
        print(f"Wrote {XLSX_OUT.name}\n")
    except PermissionError:
        print(f"SKIPPED {XLSX_OUT.name} -- it is open in Excel. "
              f"Close it and re-run to refresh the workbook; "
              f"{OUT_CSV.name} is already up to date.\n")

    # ---------- distribution ----------
    dist = Counter(t for _, t in rows)
    print("=== Round-2 gold distribution ===")
    for t in range(7):
        n = dist.get(t, 0)
        print(f"  {t} {TIERS[t]:22s} {n:4d}  {100*n/len(rows):5.1f}%")

    if not KEY.exists():
        return
    key = pd.read_csv(KEY, dtype={"pmid": str})
    gold = pd.DataFrame(rows, columns=["pmid", "gold_tier"])
    m = key.merge(gold, on="pmid", how="inner")
    m["correct"] = (m.model_tier == m.gold_tier)

    print(f"\n=== Model accuracy on its own hardest cases (n={len(m)}) ===")
    acc = 100 * m.correct.mean()
    print(f"  overall: {acc:.1f}%   (mean margin {m.margin.mean():.3f}, "
          f"mean confidence {m.model_conf.mean():.3f})")

    print("\n  by selection bucket:")
    for b, g in m.groupby("bucket"):
        print(f"    {b:24s} n={len(g):4d}  acc={100*g.correct.mean():5.1f}%")

    print("\n  by model-predicted class (what the model thought it was):")
    for t, g in m.groupby("model_tier"):
        print(f"    pred {TIERS[t]:22s} n={len(g):4d}  "
              f"precision={100*g.correct.mean():5.1f}%")

    print("\n  by TRUE class (what the model missed):")
    for t, g in m.groupby("gold_tier"):
        print(f"    true {TIERS[t]:22s} n={len(g):4d}  "
              f"recall={100*g.correct.mean():5.1f}%")

    print("\n  top confusions (model -> true):")
    conf = Counter((r.model_tier, r.gold_tier)
                   for r in m.itertuples() if not r.correct)
    for (p_, g_), n in conf.most_common(10):
        print(f"    {TIERS[p_]:22s} -> {TIERS[g_]:22s} {n:4d}")

    # value of the round: how many NEW examples per class we gain
    print("\n=== Gold set after merging round 1 + round 2 ===")
    r1 = Counter()
    for split in ("train", "val", "test"):
        p = BASE / f"gold_{split}.csv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                r1[int(r["gold_tier"])] += 1
    print(f"  {'class':22s} {'round1':>7s} {'round2':>7s} {'total':>7s} {'gain':>7s}")
    for t in range(7):
        a, b = r1.get(t, 0), dist.get(t, 0)
        gain = f"+{100*b/a:.0f}%" if a else "new"
        print(f"  {TIERS[t]:22s} {a:7d} {b:7d} {a+b:7d} {gain:>7s}")
    print(f"  {'TOTAL':22s} {sum(r1.values()):7d} {len(rows):7d} "
          f"{sum(r1.values())+len(rows):7d}")


if __name__ == "__main__":
    main()
