"""
EvidenceRx -- merge round-1 and round-2 gold into one training set.

Design decision (deliberate): the ROUND-1 TEST SPLIT IS FROZEN.
Round-2 papers were chosen by active learning, i.e. they are adversarially hard
by construction, so putting them in the test set would make the headline metric
un-comparable with the round-1 model and unrepresentative of the 49,499-paper
corpus. All 400 new papers therefore go to train/val (stratified 85/15), and
accuracy is still reported on the same 147-paper held-out test set as before.

Round-2 papers carry an annotator tag: they are AI-drafted second-annotator
labels (Cohen's kappa 0.903 vs the human annotator on an 84-paper blind
calibration set), not an independent human gold standard. Provenance is written
to gold_provenance.csv so the report can state this plainly.

Inputs : gold_train.csv, gold_val.csv, gold_test.csv   (round 1)
         evidence_round2_labels.csv                    (round 2)
Outputs: gold_train.csv, gold_val.csv   (rewritten; originals backed up once)
         gold_provenance.csv

Usage:
    python evidence_gold_merge_rounds.py
    python evidence_gold_merge_rounds.py --dry-run
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
TRAIN, VAL, TEST = (BASE / "gold_train.csv", BASE / "gold_val.csv",
                    BASE / "gold_test.csv")
R2 = BASE / "evidence_round2_labels.csv"
PROV = BASE / "gold_provenance.csv"
BACKUP_DIR = BASE / "gold_round1_backup"

SEED = 42
VAL_FRAC = 0.15
TIERS = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}


def read(path):
    return [(r["pmid"], int(r["gold_tier"]))
            for r in csv.DictReader(open(path, encoding="utf-8"))]


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "gold_tier"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    random.seed(SEED)

    for p in (TRAIN, VAL, TEST, R2):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}")

    tr1, va1, te1 = read(TRAIN), read(VAL), read(TEST)
    seen = {p for p, _ in tr1 + va1 + te1}

    r2, dup = [], 0
    for r in csv.DictReader(open(R2, encoding="utf-8")):
        if r["pmid"] in seen:
            dup += 1
            continue
        r2.append((r["pmid"], int(r["gold_tier"])))
    if dup:
        print(f"Skipped {dup} round-2 papers already in round-1 gold")

    # --- stratified 85/15 split of the round-2 papers into train/val ---
    by = defaultdict(list)
    for pmid, t in r2:
        by[t].append(pmid)
    add_tr, add_va = [], []
    for t, pmids in by.items():
        random.shuffle(pmids)
        n_va = int(round(len(pmids) * VAL_FRAC))
        add_va += [(p, t) for p in pmids[:n_va]]
        add_tr += [(p, t) for p in pmids[n_va:]]

    tr2, va2 = tr1 + add_tr, va1 + add_va

    print(f"\nRound 1: train {len(tr1)} | val {len(va1)} | test {len(te1)} "
          f"(test FROZEN)")
    print(f"Round 2: +{len(add_tr)} train | +{len(add_va)} val "
          f"({len(r2)} new papers)")
    print(f"Merged : train {len(tr2)} | val {len(va2)} | test {len(te1)}")

    print(f"\n  {'class':22s} {'train':>7s} {'val':>6s} {'test':>6s} {'total':>7s}")
    ctr, cva, cte = (Counter(t for _, t in x) for x in (tr2, va2, te1))
    for t in range(7):
        tot = ctr.get(t, 0) + cva.get(t, 0) + cte.get(t, 0)
        print(f"  {TIERS[t]:22s} {ctr.get(t,0):7d} {cva.get(t,0):6d} "
              f"{cte.get(t,0):6d} {tot:7d}")
    print(f"  {'TOTAL':22s} {len(tr2):7d} {len(va2):6d} {len(te1):6d} "
          f"{len(tr2)+len(va2)+len(te1):7d}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir()
        for p in (TRAIN, VAL, TEST):
            shutil.copy2(p, BACKUP_DIR / p.name)
        print(f"\nBacked up round-1 splits -> {BACKUP_DIR.name}/")
    else:
        print(f"\n(backup {BACKUP_DIR.name}/ already exists -- left untouched)")

    write(TRAIN, tr2)
    write(VAL, va2)

    r2_set = {p for p, _ in r2}
    with open(PROV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "gold_tier", "round", "annotator", "split"])
        for split, rows in (("train", tr2), ("val", va2), ("test", te1)):
            for pmid, t in rows:
                if pmid in r2_set:
                    w.writerow([pmid, t, 2, "claude-2nd-annotator", split])
                else:
                    w.writerow([pmid, t, 1, "human", split])

    print(f"Wrote {TRAIN.name}, {VAL.name}, {PROV.name}")
    print("\nNext:  python evidence_prep_e3_data.py   "
          "(rebuilds e3_* with the 1,400-paper gold set)")


if __name__ == "__main__":
    main()
