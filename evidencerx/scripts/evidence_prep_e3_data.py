"""
EvidenceRx -- Phase 4 (data prep): build the two-stage, 7-class training and
inference sets for the evidence-tier classifier.

Classes (model index = tier):
    0 off-topic   1 synthesis   2 rct   3 human-observational
    4 animal      5 in-vitro     6 review

Two-stage design (per the spec):
    Stage 1  -> train on WEAK rule labels (large, noisy, tiers 1-6 only)
    Stage 2  -> fine-tune on GOLD labels (small, clean, tiers 0-6 incl off-topic)
    Eval     -> gold test split (the honest metric)

Leakage guard: the 1,000 gold papers are EXCLUDED from the weak set, so the
gold test papers never appear in stage-1 training with noisy labels.

Inputs : pmid_studytype.csv, articles_cache.json,
         gold_train.csv / gold_val.csv / gold_test.csv
Outputs: e3_weak.csv, e3_gold_train.csv, e3_gold_val.csv, e3_gold_test.csv,
         e3_predict.csv

Usage:
    python evidence_prep_e3_data.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
WEAK_CSV = BASE_DIR / "pmid_studytype.csv"
ARTICLES = BASE_DIR / "articles_cache.json"
METADATA = BASE_DIR / "pubmed_metadata_cache.json"

# v3: fold PublicationType + MeSH into the model input.
# Set False to reproduce the v2 inputs (title+abstract only, 86.4% model).
WITH_METADATA = True
MESH_CAP = 20          # keep room for the abstract inside 512 tokens
GOLD = {"train": BASE_DIR / "gold_train.csv",
        "val": BASE_DIR / "gold_val.csv",
        "test": BASE_DIR / "gold_test.csv"}

OUT_WEAK = BASE_DIR / "e3_weak.csv"
OUT_GOLD = {"train": BASE_DIR / "e3_gold_train.csv",
            "val": BASE_DIR / "e3_gold_val.csv",
            "test": BASE_DIR / "e3_gold_test.csv"}
OUT_PREDICT = BASE_DIR / "e3_predict.csv"

CLASS_NAMES = ["off-topic", "synthesis", "rct", "human-observational",
               "animal", "in-vitro", "review"]  # index == tier


def build_text(art, meta=None):
    """Model input. Metadata goes BEFORE the abstract on purpose: BERT
    truncates from the right, so putting PublicationType/MeSH at the end
    would silently drop it for long abstracts -- exactly the papers where
    it matters most.

    PublicationType covers 100% of the corpus and is near-decisive for
    tiers 1, 2 and 6 (Randomized Controlled Trial / Systematic Review /
    Review). MeSH covers ~71% and disambiguates animal vs in-vitro.
    """
    title = (art.get("title") or "").strip()
    abstract = (art.get("abstract") or "").strip()
    if not title and not abstract:
        return None
    parts = [f"[TITLE] {title}"]
    if WITH_METADATA and meta:
        pts = meta.get("pub_types") or []
        mh = meta.get("mesh") or []
        # 'none' is a real signal: un-indexed papers are mostly very recent
        parts.append("[PUBTYPE] " + ("; ".join(pts) if pts else "none"))
        parts.append("[MESH] " + ("; ".join(mh[:MESH_CAP]) if mh else "none"))
    parts.append(f"[ABSTRACT] {abstract}")
    return " ".join(parts).strip()


def read_gold(path):
    return {r["pmid"]: int(r["gold_tier"])
            for r in csv.DictReader(open(path, encoding="utf-8"))}


def write(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    arts = json.load(open(ARTICLES, encoding="utf-8"))
    meta = json.load(open(METADATA, encoding="utf-8")) if (
        WITH_METADATA and METADATA.exists()) else {}
    weak_rows = list(csv.DictReader(open(WEAK_CSV, encoding="utf-8")))

    gold = {split: read_gold(p) for split, p in GOLD.items()}
    gold_pmids = set().union(*[set(g) for g in gold.values()])

    # ---- WEAK training set (tiers 1-6), excluding all gold papers ----
    weak_out, dropped = [], 0
    for r in weak_rows:
        pmid, tier = r["pmid"], r["tier"]
        if not tier:                      # abstained -> not a training label
            continue
        if pmid in gold_pmids:            # leakage guard
            continue
        text = build_text(arts.get(pmid, {}) or {}, meta.get(pmid))
        if text is None:
            dropped += 1
            continue
        weak_out.append((pmid, text, int(tier)))
    write(OUT_WEAK, weak_out, ["pmid", "text", "label"])

    # ---- GOLD splits (tiers 0-6) ----
    gold_counts = {}
    for split, gmap in gold.items():
        rows = []
        for pmid, gt in gmap.items():
            text = build_text(arts.get(pmid, {}) or {}, meta.get(pmid))
            if text is None:
                continue
            rows.append((pmid, text, gt))
        write(OUT_GOLD[split], rows, ["pmid", "text", "label"])
        gold_counts[split] = len(rows)

    # ---- PREDICT set: only the papers in our associations (from the weak
    #      label file = the 49,499 with metadata), not all cached papers ----
    pred = []
    for r in weak_rows:
        pmid = r["pmid"]
        text = build_text(arts.get(pmid, {}) or {}, meta.get(pmid))
        if text is not None:
            pred.append((pmid, text))
    write(OUT_PREDICT, pred, ["pmid", "text"])

    # ---- report ----
    print(f"metadata in input: {WITH_METADATA}"
          + (f" (cache: {len(meta):,} papers)" if meta else ""))
    print(f"WEAK training rows (tiers 1-6, gold excluded): {len(weak_out):,}")
    print(f"  dropped (no text): {dropped}")
    print("  weak label balance:",
          dict(sorted(Counter(r[2] for r in weak_out).items())))
    print(f"\nGOLD rows (tiers 0-6): "
          f"train {gold_counts['train']} | val {gold_counts['val']} | "
          f"test {gold_counts['test']}")
    print(f"\nPREDICT set (all papers): {len(pred):,}")
    print(f"\nWrote {OUT_WEAK.name}, e3_gold_[train|val|test].csv, "
          f"{OUT_PREDICT.name}")
    print("Classes (index=tier):", {i: n for i, n in enumerate(CLASS_NAMES)})


if __name__ == "__main__":
    main()
