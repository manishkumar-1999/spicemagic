"""
EvidenceRx -- audit the round-2 annotations against PubMed metadata.

The labels were assigned from title+abstract ONLY. PublicationType and MeSH
descriptors are an INDEPENDENT signal (assigned by NLM indexers), so
disagreements are a genuine check, not a circular one.

Only high-confidence contradictions are flagged -- the checks that produced
false alarms in the round-1 audit are deliberately NOT repeated:
  * "Humans MeSH but label=in-vitro" is NOT flagged (human cell lines are
    indexed with Humans).
  * Un-indexed papers (no MeSH at all) are never flagged -- recent papers are
    simply not indexed yet, and treating that as evidence was the round-1 trap.

Reads : r2_labels_part1..4.csv, pubmed_metadata_cache.json,
        evidence_gold_round2.xlsx (for pmid<->row_id and titles)
Writes: evidence_round2_audit.csv

Usage:
    python evidence_round2_audit.py
"""

import csv
from collections import Counter
from pathlib import Path
import json

from openpyxl import load_workbook

BASE = Path(__file__).parent
PARTS = [BASE / f"r2_labels_part{i}.csv" for i in (1, 2, 3, 4)]
META = BASE / "pubmed_metadata_cache.json"
XLSX = BASE / "evidence_gold_round2.xlsx"
OUT = BASE / "evidence_round2_audit.csv"

TIERS = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}

PT_SYNTH = {"Meta-Analysis", "Systematic Review"}
PT_RCT = {"Randomized Controlled Trial"}
PT_TRIAL = {"Clinical Trial", "Controlled Clinical Trial",
            "Clinical Trial, Phase I", "Clinical Trial, Phase II",
            "Clinical Trial, Phase III"}
PT_REVIEW = {"Review", "Editorial", "Comment", "Letter", "News",
             "Practice Guideline", "Guideline"}
PT_CASE = {"Case Reports"}
PT_RETRACT = {"Retraction of Publication", "Retracted Publication",
              "Published Erratum"}
# MeSH that mean "this was a dish, not a living animal"
CELL_MESH = {"Cells, Cultured", "Cell Line", "Cell Line, Tumor",
             "In Vitro Techniques", "Cell Survival", "Macrophages",
             "Microbial Sensitivity Tests", "Plant Extracts"}
# MeSH that mean "this was a computer, not a bench"
INSILICO_MESH = {"Molecular Docking Simulation", "Computer Simulation",
                 "Network Pharmacology", "Medicine, Chinese Traditional",
                 "Computational Biology", "Molecular Dynamics Simulation",
                 "Protein Interaction Maps", "Drugs, Chinese Herbal"}


def main():
    labels = {}
    for p in PARTS:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            labels[int(r["row_id"])] = int(r["tier"])

    wb = load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["Annotation"]
    head = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    i_row, i_pmid, i_title = (head.index("row_id"), head.index("pmid"),
                              head.index("title"))
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[i_row] is None:
            continue
        rows.append((int(r[i_row]), str(r[i_pmid]), r[i_title] or ""))

    meta = json.load(open(META, encoding="utf-8"))

    flags, n_indexed, n_unindexed = [], 0, 0
    for rid, pmid, title in rows:
        t = labels.get(rid)
        m = meta.get(pmid) or {}
        pts = set(m.get("pub_types") or [])
        mh = set(m.get("mesh") or [])
        if not mh:
            n_unindexed += 1
            continue                      # never flag un-indexed papers
        n_indexed += 1
        has_hum = "Humans" in mh
        has_ani = "Animals" in mh

        def flag(reason, sev):
            flags.append({"row_id": rid, "pmid": pmid, "my_tier": t,
                          "my_class": TIERS[t], "severity": sev,
                          "reason": reason,
                          "pub_types": "; ".join(sorted(pts)),
                          "title": title[:110]})

        # --- HIGH: indexer says RCT / synthesis, I said something weaker ---
        if pts & PT_RCT and t != 2:
            flag("PublicationType=Randomized Controlled Trial", "HIGH")
        elif pts & PT_SYNTH and t not in (1, 6):
            flag("PublicationType=Meta-Analysis/Systematic Review", "HIGH")
        # --- HIGH: I claimed strong human evidence, no Humans MeSH ---
        elif t in (1, 2, 3) and not has_hum:
            flag("labelled human evidence but no 'Humans' MeSH", "HIGH")
        # --- HIGH: I claimed animal, no Animals MeSH ---
        elif t == 4 and not has_ani:
            flag("labelled animal but no 'Animals' MeSH", "HIGH")
        # --- MED: I said off-topic, but indexed as a human clinical study ---
        elif t == 0 and (pts & (PT_RCT | PT_TRIAL)) and has_hum:
            flag("labelled off-topic but indexed as a human clinical trial",
                 "MED")
        # --- MED: I said in-vitro, indexer marks a live organism.
        # Suppressed when the record also carries a cell-culture descriptor:
        # animal-derived cells, animal-source enzymes and animal parasites all
        # get 'Animals' MeSH, so the bare tag is not evidence of an in-vivo
        # study. (Mirror of the round-1 'Humans MeSH -> human tier' trap.)
        elif t == 5 and has_ani and not has_hum and not (mh & CELL_MESH):
            flag("labelled in-vitro but 'Animals' MeSH with no 'Humans' "
                 "and no cell-culture descriptor", "MED")
        # --- MED: I said review, indexer disagrees on the document type.
        # Suppressed for in-silico work: NLM has no PublicationType for
        # docking / network pharmacology, so it is always 'Journal Article'.
        elif (t == 6 and not (pts & PT_REVIEW) and not (pts & PT_RETRACT)
                and not (mh & INSILICO_MESH)):
            flag("labelled review/in-silico but no Review/Editorial/Letter "
                 "PublicationType", "MED")
        # --- LOW: case report indexed, I did not call it human-observational
        elif pts & PT_CASE and t != 3:
            flag("PublicationType=Case Reports", "LOW")

    flags.sort(key=lambda f: ({"HIGH": 0, "MED": 1, "LOW": 2}[f["severity"]],
                              f["row_id"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "pmid", "my_tier",
                                          "my_class", "severity", "reason",
                                          "pub_types", "title"])
        w.writeheader()
        w.writerows(flags)

    print(f"Papers labelled            : {len(rows)}")
    print(f"  MeSH-indexed (auditable) : {n_indexed}")
    print(f"  un-indexed (not checked) : {n_unindexed}")
    print(f"\nFlagged for review: {len(flags)} "
          f"({100*len(flags)/max(n_indexed,1):.1f}% of auditable)")
    for sev, n in sorted(Counter(f["severity"] for f in flags).items()):
        print(f"  {sev:5s} {n:4d}")
    print("\n=== by reason ===")
    for reason, n in Counter(f["reason"] for f in flags).most_common():
        print(f"  {n:4d}  {reason}")
    print("\n=== my class among flagged ===")
    for c, n in Counter(f["my_class"] for f in flags).most_common():
        print(f"  {n:4d}  {c}")
    print(f"\nWrote {OUT.name}")


if __name__ == "__main__":
    main()
