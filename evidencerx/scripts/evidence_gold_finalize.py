"""
EvidenceRx -- Phase 3 (finalize): score the CURRENT rules against the gold
labels and produce the 7-class train/val/test split.

Classes: 0 off-topic (association-mining noise) + tiers 1-6.
The off-topic class (0) is learned by the model from the gold set; the rules
themselves only emit tiers 1-6 (they cannot judge relevance).

Reads:
    evidence_gold_sample_annotated.xlsx   (Annotation sheet, gold_tier 0-6)
    pmid_studytype.csv                     (current weak rule labels)
Writes:
    gold_train.csv / gold_val.csv / gold_test.csv   (pmid, gold_tier)

Usage:
    python evidence_gold_finalize.py
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook

BASE_DIR = Path(__file__).parent
GOLD_XLSX = BASE_DIR / "evidence_gold_sample_annotated.xlsx"
CORRECTIONS = BASE_DIR / "evidence_gold_corrections.csv"  # audited overlay
WEAK_CSV = BASE_DIR / "pmid_studytype.csv"
TRAIN = BASE_DIR / "gold_train.csv"
VAL = BASE_DIR / "gold_val.csv"
TEST = BASE_DIR / "gold_test.csv"

SEED = 42
TIERS = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}
SHORT = {0: "offtop", 1: "synth", 2: "rct", 3: "human",
         4: "animal", 5: "invitro", 6: "review"}


def load_gold():
    wb = load_workbook(GOLD_XLSX, read_only=True, data_only=True)
    ws = wb["Annotation"]
    gold, blanks = {}, 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[1] is None:
            continue
        v = r[4]
        if v is None or str(v).strip() == "":
            blanks += 1
            continue
        try:
            gold[str(r[1])] = int(v)
        except (TypeError, ValueError):
            blanks += 1
    # apply audited corrections overlay (original file stays untouched)
    n_corr = 0
    if CORRECTIONS.exists():
        for c in csv.DictReader(open(CORRECTIONS, encoding="utf-8")):
            if c["pmid"] in gold:
                gold[c["pmid"]] = int(c["new_tier"])
                n_corr += 1
        if n_corr:
            print(f"Applied {n_corr} audited correction(s) from "
                  f"{CORRECTIONS.name}\n")
    return gold, blanks


def main():
    if not GOLD_XLSX.exists():
        raise SystemExit(f"Missing {GOLD_XLSX.name}.")
    gold, blanks = load_gold()
    if not gold:
        raise SystemExit("No gold_tier values found -- label the sheet first.")
    if blanks:
        print(f"WARNING: {blanks} rows unlabeled; scoring the {len(gold)} done.\n")

    weak = {}
    for r in csv.DictReader(open(WEAK_CSV, encoding="utf-8")):
        t = int(r["tier"]) if r["tier"] else None
        weak[r["pmid"]] = (t, "text" if r["rule"].startswith("T") else
                           ("metadata" if r["rule"].startswith("M") else "abstain"))

    # ---- split fired vs abstain, on-topic vs off-topic ----
    fired = [(p, g, weak[p][0], weak[p][1]) for p, g in gold.items()
             if weak.get(p) and weak[p][0] is not None]
    abst = [(p, g) for p, g in gold.items()
            if weak.get(p) and weak[p][0] is None]

    n_off = sum(1 for _, g in gold.items() if g == 0)
    print(f"Gold papers: {len(gold)}")
    print(f"  off-topic (tier 0): {n_off} ({n_off/len(gold)*100:.0f}%) "
          f"-- association-mining noise")
    print(f"  rules fired: {len(fired)} | rules abstained: {len(abst)}\n")

    ont = [(g, w, k) for _, g, w, k in fired if g != 0]

    def pr(pairs):
        return (100 * sum(1 for g, w in pairs if g == w) / len(pairs),
                len(pairs)) if pairs else (float("nan"), 0)

    p_all, n_all = pr([(g, w) for _, g, w, _ in fired])       # off-topic = error
    p_on, n_on = pr([(g, w) for g, w, _ in ont])              # on-topic only
    print("=== Rule precision vs gold ===")
    print(f"  On-topic study-type precision : {p_on:.1f}%  (n={n_on})   <- report this")
    for kind in ("metadata", "text"):
        pk, nk = pr([(g, w) for g, w, k in ont if k == kind])
        print(f"    {kind:9s} rules        : {pk:.1f}%  (n={nk})")
    print(f"  (Counting off-topic as errors : {p_all:.1f}% -- reflects noise, "
          f"not tier accuracy)")

    print("\n=== Confusion: rule tier (row) x gold tier (col), on-topic ===")
    M = defaultdict(Counter)
    for g, w, _ in ont:
        M[w][g] += 1
    print("  rule\\gold " + " ".join(f"{SHORT[t]:>7s}" for t in range(1, 7)) + "   prec")
    for w in range(1, 7):
        row = M[w]
        tot = sum(row.values())
        if tot:
            cells = " ".join(f"{row.get(g, 0):7d}" for g in range(1, 7))
            print(f"  {SHORT[w]:8s} {cells}   {row.get(w, 0)/tot*100:4.0f}%")

    if abst:
        print("\n=== What the rule-ABSTAINED hard cases are (gold) ===")
        ag = Counter(g for _, g in abst)
        for t in range(0, 7):
            c = ag.get(t, 0)
            if c:
                print(f"    {TIERS[t]:20s} {c:3d}")

    chk = "PASS (>=90%)" if p_on >= 90 else f"BELOW 90% (at {p_on:.0f}%) -- residual is the model's job"
    print(f"\n  PDF Phase-2 checkpoint: {chk}")

    # ---- stratified 70/15/15 split over ALL 7 classes (incl off-topic) ----
    random.seed(SEED)
    by = defaultdict(list)
    for p, g in gold.items():
        if g in TIERS:
            by[g].append(p)
    tr, va, te = [], [], []
    for g, pmids in by.items():
        random.shuffle(pmids)
        n = len(pmids)
        nte, nva = int(n * 0.15), int(n * 0.15)
        te += [(p, g) for p in pmids[:nte]]
        va += [(p, g) for p in pmids[nte:nte + nva]]
        tr += [(p, g) for p in pmids[nte + nva:]]
    for path, data in [(TRAIN, tr), (VAL, va), (TEST, te)]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pmid", "gold_tier"])
            w.writerows(data)
    print(f"\n7-class split (0-6) -> train {len(tr)} | val {len(va)} | test {len(te)}")
    print("  class balance (train):",
          dict(sorted(Counter(g for _, g in tr).items())))
    print(f"Wrote {TRAIN.name}, {VAL.name}, {TEST.name}")


if __name__ == "__main__":
    main()
