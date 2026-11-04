"""
EvidenceRx — Phase 3: build the gold-set annotation sheet (~1,000 papers).

Stratified across the 6 tiers using the weak labels, OVERSAMPLING the rare
tiers (synthesis / rct / review) and forcing in a hard-case slice:
  - abstains (rules fired nothing)
  - text-rule labels (regex, the precision-risk ones)

The weak label is HIDDEN from the annotator (separate key file) to avoid
anchoring, so we can measure true rule precision against the gold labels.

Outputs:
    evidence_gold_sample.xlsx   two sheets: Guidelines + Annotation (blank
                                gold_tier column for you to fill)
    evidence_gold_key.csv       pmid -> weak tier/studytype/rule/stratum
                                (for measuring rule precision AFTER labelling)

After labelling, run evidence_gold_finalize.py to split 70/15/15 and score
the rules.

Usage:
    python evidence_gold_sample.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

BASE_DIR = Path(__file__).parent
LABELS_CSV = BASE_DIR / "pmid_studytype.csv"
ARTICLES = BASE_DIR / "articles_cache.json"
OUT_XLSX = BASE_DIR / "evidence_gold_sample.xlsx"
OUT_KEY = BASE_DIR / "evidence_gold_key.csv"

SEED = 42
TARGET_PER_TIER = 140     # 6 tiers x 140 = 840
TARGET_ABSTAIN = 160      # hard cases -> ~1,000 total
TEXT_RULE_FRACTION = 0.5  # within a tier, aim ~half from text (T*) rules

TIERS = {1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}

TIER_DEFS = [
    (1, "synthesis", "Meta-analysis or systematic review OF HUMAN TRIALS.",
     "e.g. 'A meta-analysis of 12 randomized trials of curcumin in ...'"),
    (2, "rct", "Randomized controlled trial in humans.",
     "e.g. 'A randomized, double-blind, placebo-controlled trial of 120 patients'"),
    (3, "human-observational",
     "Non-randomized human study: cohort, case-control, cross-sectional, "
     "observational, case report, or non-randomized clinical trial.",
     "e.g. 'A survey of 340 patients...', 'a case report of anaphylaxis after...'"),
    (4, "animal", "In-vivo animal study (mice, rats, murine, zebrafish...).",
     "e.g. 'Curcumin was administered orally to Wistar rats for 8 weeks'"),
    (5, "in-vitro", "Cell line / biochemical / in-vitro assay (no organism).",
     "e.g. 'induced apoptosis in HCT-116 cells at 25 uM'"),
    (6, "review", "Narrative review, opinion/editorial, or in-silico only "
     "(molecular docking, computational).",
     "e.g. 'This review summarizes...', 'molecular docking of piperine'"),
]


def rule_kind(rule):
    if rule.startswith("M"):
        return "metadata"
    if rule.startswith("T"):
        return "text"
    return "abstain"


def main():
    random.seed(SEED)
    arts = json.load(open(ARTICLES, encoding="utf-8"))
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))

    # bucket by studytype, tracking rule kind
    by_tier = defaultdict(lambda: {"metadata": [], "text": []})
    abstains = []
    for r in rows:
        st = r["studytype"]
        kind = rule_kind(r["rule"])
        if st and kind in ("metadata", "text"):
            by_tier[st][kind].append(r)
        elif not st:
            abstains.append(r)

    picked = []

    def take(pool, n):
        random.shuffle(pool)
        return pool[:n]

    for t in range(1, 7):
        st = TIERS[t]
        meta_pool = by_tier[st]["metadata"]
        text_pool = by_tier[st]["text"]
        n_text = min(len(text_pool), int(TARGET_PER_TIER * TEXT_RULE_FRACTION))
        n_meta = TARGET_PER_TIER - n_text
        n_meta = min(n_meta, len(meta_pool))
        # if one pool is short, backfill from the other
        chosen = take(text_pool, n_text) + take(meta_pool, n_meta)
        deficit = TARGET_PER_TIER - len(chosen)
        if deficit > 0:
            extra_pool = (text_pool[n_text:] + meta_pool[n_meta:])
            chosen += take(extra_pool, deficit)
        picked += chosen

    picked += take(abstains, TARGET_ABSTAIN)
    random.shuffle(picked)

    # ---- write key (hidden weak labels) ----
    with open(OUT_KEY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "weak_tier", "weak_studytype", "rule",
                    "rule_kind", "is_case_report"])
        for r in picked:
            w.writerow([r["pmid"], r["tier"], r["studytype"], r["rule"],
                        rule_kind(r["rule"]), r["is_case_report"]])

    # ---- write annotation workbook (blind) ----
    wb = Workbook()

    ws = wb.active
    ws.title = "Guidelines"
    ws["A1"] = "EvidenceRx — Gold Annotation Guidelines"
    ws["A1"].font = Font(bold=True, size=14)
    ws["A3"] = ("Read the title + abstract, then put the single best TIER "
                "NUMBER (1-6) in the gold_tier column on the Annotation sheet. "
                "Judge the STUDY the paper itself performed (for a review, the "
                "paper is tier 6 even if it discusses trials).")
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A3:F5")
    hdr = ["tier", "class", "definition", "example"]
    ws.append([])
    ws.append(hdr)
    for c in ws[ws.max_row]:
        c.font = Font(bold=True)
    for t, name, d, ex in TIER_DEFS:
        ws.append([t, name, d, ex])
    ws.append([])
    ws.append(["", "If genuinely unclear or off-topic (not a spice-health "
               "study at all), write 0 and add a note."])
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 50
    for col in ("A", "B"):
        ws.column_dimensions[col].width = 14

    an = wb.create_sheet("Annotation")
    headers = ["row_id", "pmid", "title", "abstract", "gold_tier", "notes"]
    an.append(headers)
    for c in an[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="2E75B6")
        c.alignment = Alignment(horizontal="center")
    for i, r in enumerate(picked, 1):
        art = arts.get(r["pmid"], {}) or {}
        title = (art.get("title") or "").strip()
        abstract = (art.get("abstract") or "").strip()
        if len(abstract) > 2000:
            abstract = abstract[:2000] + " ..."
        an.append([i, r["pmid"], title, abstract, "", ""])
    an.column_dimensions["C"].width = 55
    an.column_dimensions["D"].width = 90
    an.column_dimensions["E"].width = 10
    an.column_dimensions["F"].width = 25
    an.freeze_panes = "A2"
    for row in an.iter_rows(min_row=2):
        row[2].alignment = Alignment(wrap_text=True, vertical="top")
        row[3].alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(OUT_XLSX)

    # ---- report ----
    from collections import Counter
    strata = Counter((r["studytype"] or "ABSTAIN", rule_kind(r["rule"]))
                     for r in picked)
    print(f"Sampled {len(picked):,} papers for gold annotation.\n")
    print("=== Stratum composition (weak label x rule kind) ===")
    for t in range(1, 7):
        st = TIERS[t]
        m = strata.get((st, "metadata"), 0)
        tx = strata.get((st, "text"), 0)
        print(f"  tier {t} {st:20s} metadata={m:3d}  text={tx:3d}  total={m+tx:3d}")
    print(f"  {'ABSTAIN (hard cases)':27s} {strata.get(('ABSTAIN','abstain'),0):3d}")
    print(f"\nWrote {OUT_XLSX.name} (label the 'gold_tier' column, 1-6)")
    print(f"Wrote {OUT_KEY.name} (hidden weak labels for precision scoring)")


if __name__ == "__main__":
    main()
