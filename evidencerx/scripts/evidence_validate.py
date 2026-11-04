"""
EvidenceRx -- pipeline health check. Run before trusting any downstream result.

Checks, in dependency order:
  1. Which prediction file is in play, and does it match the reported model
  2. Gold split integrity (sizes, no overlap, provenance)
  3. Leakage: no gold TEST paper may appear in the weak training set
  4. Reported test accuracy, recomputed from the shipped predictions
  5. final_database_v3 integrity vs v2 (no rows gained or lost, all tiered)
  6. Internal consistency of is_offtopic / evidence_tier / tier_class
  7. Staleness: every artefact newer than the file it derives from

Exit code 0 = all pass, 1 = at least one FAIL.

Usage:
    python evidence_validate.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
FAILS, WARNS = [], []


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg):
    WARNS.append(msg)
    print(f"  [WARN] {msg}")


def mtime(p):
    return os.path.getmtime(p) if Path(p).exists() else None


def stamp(p):
    t = mtime(p)
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "MISSING"


def main():
    print("=" * 68)
    print(" EvidenceRx pipeline validation")
    print("=" * 68)

    # ---- 1. prediction file in play ----
    print("\n[1] Prediction file")
    pf = BASE / "e3_tier_predictions.csv"
    if not pf.exists():
        fail("e3_tier_predictions.csv missing")
        return finish()
    preds = pd.read_csv(pf, dtype={"pmid": str})
    conf = preds.confidence.mean()
    print(f"      {pf.name}  modified {stamp(pf)}  rows {len(preds):,}  "
          f"mean conf {conf:.3f}")
    known = {0.897: "round-1 (80.3%)", 0.921: "round-2 v1 (83.0%)",
             0.925: "round-2 v2 (86.4%)"}
    tag = next((v for k, v in known.items() if abs(conf - k) < 0.0015), None)
    if tag == "round-2 v2 (86.4%)":
        ok(f"in use: {tag} -- the best model")
    elif tag:
        warn(f"in use: {tag} -- NOT the best model (v2 = 86.4%)")
    else:
        warn("unrecognised model fingerprint; verify provenance manually")
    if len(preds) != 49499:
        fail(f"expected 49,499 prediction rows, found {len(preds):,}")
    else:
        ok("49,499 papers predicted")

    # ---- 2. gold splits ----
    print("\n[2] Gold splits")
    g = {}
    for s in ("train", "val", "test"):
        p = BASE / f"gold_{s}.csv"
        g[s] = {r["pmid"]: int(r["gold_tier"])
                for r in csv.DictReader(open(p, encoding="utf-8"))}
    sizes = {k: len(v) for k, v in g.items()}
    print(f"      train {sizes['train']} | val {sizes['val']} | "
          f"test {sizes['test']} | total {sum(sizes.values())}")
    if sum(sizes.values()) != 1400:
        fail(f"expected 1,400 gold papers, found {sum(sizes.values())}")
    else:
        ok("1,400 gold papers")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        dup = set(g[a]) & set(g[b])
        if dup:
            fail(f"{len(dup)} pmid(s) in BOTH {a} and {b}")
        else:
            ok(f"no overlap {a}/{b}")

    prov = BASE / "gold_provenance.csv"
    if prov.exists():
        pv = pd.read_csv(prov, dtype={"pmid": str})
        t = pv[pv.split == "test"]
        if (t.annotator == "human").all():
            ok(f"test split is 100% human-labelled ({len(t)} papers)")
        else:
            warn(f"test split contains {(t.annotator!='human').sum()} "
                 f"AI-labelled papers -- weakens the headline metric")
        ai = (pv.annotator != "human").sum()
        print(f"      provenance: {len(pv)-ai} human, {ai} AI "
              f"({100*ai/len(pv):.0f}% AI overall)")

    # ---- 3. leakage ----
    print("\n[3] Leakage guard")
    wk = BASE / "e3_weak.csv"
    if wk.exists():
        weak_ids = set(pd.read_csv(wk, usecols=["pmid"],
                                   dtype={"pmid": str}).pmid)
        for s in ("test", "val", "train"):
            bad = set(g[s]) & weak_ids
            if bad:
                fail(f"{len(bad)} gold {s} paper(s) also in the weak set")
            else:
                ok(f"no gold {s} paper in the weak training set")
    else:
        warn("e3_weak.csv absent (fine -- v2/v3 do not use the weak stage)")

    # ---- 4. recompute the reported accuracy ----
    print("\n[4] Test accuracy, recomputed from the shipped predictions")
    gt = pd.DataFrame(list(g["test"].items()), columns=["pmid", "gold_tier"])
    m = gt.merge(preds[["pmid", "pred_tier"]], on="pmid", how="left")
    if m.pred_tier.isna().any():
        fail(f"{int(m.pred_tier.isna().sum())} test paper(s) absent from "
             f"predictions")
    else:
        acc = (m.gold_tier == m.pred_tier).mean()
        n = len(m)
        ci = 1.96 * np.sqrt(acc * (1 - acc) / n)
        print(f"      accuracy {acc:.3f} on n={n}   95% CI +/-{100*ci:.1f} pp "
              f"-> [{100*(acc-ci):.1f}, {100*(acc+ci):.1f}]")
        if acc >= 0.86:
            ok(f"matches the reported 86.4%")
        else:
            warn(f"recomputed {acc:.3f}, below the reported 0.864")

    # ---- 5. database integrity ----
    print("\n[5] final_database_v3")
    v3p, v2p = BASE / "final_database_v3.csv", BASE / "final_database_v2.csv"
    if not v3p.exists():
        fail("final_database_v3.csv missing -- run evidence_build_v3.py")
        return finish()
    v3 = pd.read_csv(v3p, usecols=["Spice", "Disease", "Association Nature",
                                   "evidence_tier", "tier_class",
                                   "is_offtopic"])
    n2 = sum(1 for _ in open(v2p, encoding="utf-8", errors="ignore")) - 1
    print(f"      v3 rows {len(v3):,}   (v2 raw lines {n2:,})")
    if len(v3) != 228750:
        warn(f"expected 228,750 associations, found {len(v3):,}")
    else:
        ok("228,750 associations")
    nt = v3.evidence_tier.isna().sum()
    if nt:
        fail(f"{nt:,} associations without an evidence_tier")
    else:
        ok("every association carries a tier")
    bad = v3[~v3.evidence_tier.between(0, 6)]
    if len(bad):
        fail(f"{len(bad):,} tiers outside 0-6")
    else:
        ok("all tiers in range 0-6")

    # ---- 6. internal consistency ----
    print("\n[6] Internal consistency")
    mism = ((v3.evidence_tier == 0) != (v3.is_offtopic.astype(bool))).sum()
    if mism:
        fail(f"{mism:,} rows where is_offtopic disagrees with tier==0")
    else:
        ok("is_offtopic matches evidence_tier == 0")
    names = {0: "off-topic", 1: "synthesis", 2: "rct",
             3: "human-observational", 4: "animal", 5: "in-vitro",
             6: "review"}
    wrong = (v3.tier_class != v3.evidence_tier.map(names)).sum()
    if wrong:
        fail(f"{wrong:,} rows where tier_class disagrees with evidence_tier")
    else:
        ok("tier_class matches evidence_tier")
    lab = v3["Association Nature"].astype(str).str.strip().str.lower()
    print(f"      association nature: {lab.value_counts().head(3).to_dict()}")

    # ---- 7. staleness ----
    print("\n[7] Staleness (each artefact must be newer than its input)")
    chain = [("e3_tier_predictions.csv", "final_database_v3.csv"),
             ("final_database_v3.csv", "evidence_pairs.csv"),
             ("final_database_v3.csv", "controversy_pairs.csv"),
             ("final_database_v3.csv", "evidence_figures/tier_composition.png"),
             ("final_database_v3.csv", "evidence_figures/preclinical_hype.png")]
    for src, out in chain:
        ts, to = mtime(BASE / src), mtime(BASE / out)
        if to is None:
            fail(f"{out} missing")
        elif ts and to < ts:
            fail(f"{out} ({stamp(BASE/out)}) is OLDER than {src} "
                 f"({stamp(BASE/src)}) -- re-run it")
        else:
            ok(f"{out} is current")

    return finish()


def finish():
    print("\n" + "=" * 68)
    if FAILS:
        print(f" {len(FAILS)} FAILURE(S) -- do not proceed to Phase 9:")
        for f in FAILS:
            print(f"   - {f}")
    else:
        print(" ALL CHECKS PASSED")
    if WARNS:
        print(f"\n {len(WARNS)} warning(s):")
        for w in WARNS:
            print(f"   - {w}")
    print("=" * 68)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
