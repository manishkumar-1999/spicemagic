"""
EvidenceRx -- Phase 9: build the confounder-audit sheet.

Phase 6 and 7 report numbers produced by three stacked automatic layers:
association mining -> sentiment -> evidence tier. Nobody has checked whether
the inputs to those numbers are real. This builds the sheet so a human can,
and so the headline figures can carry a MEASURED error rate instead of an
assumed one.

Already known to be a live risk: spot-checking Phase 7 showed
"Licorice / toxicity (85% positive)" was papers titled "Antiplasmodial
activity of ethanolic extracts..." -- the disease term was mined from the
word *cytotoxicity* in the methods. ~17.5% of association rows carry a method
word as the "disease", so DISEASE-TERM VALIDITY is audited explicitly, not
just tier correctness.

Sampling is stratified on purpose:
  A HYPE      -- largest preclinical-minus-human gaps   (is the motif real?)
  B CONTESTED -- highest tier-weighted entropy          (is the split real?)
  C RANDOM    -- a plain random control                 (base error rate)
Without C you would only ever inspect cases you already distrust, and could
not tell a targeted error rate from the ordinary one.

Reads : controversy_pairs.csv, final_database_v3.csv
Writes: evidence_confounder_audit.xlsx   (Guidelines / Pairs / Papers)

Usage:
    python evidence_confounder_audit.py
    python evidence_confounder_audit.py --per-bucket 15 --max-papers 12
"""

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

BASE = Path(__file__).parent
PAIRS = BASE / "controversy_pairs.csv"
DB = BASE / "final_database_v3.csv"
OUT = BASE / "evidence_confounder_audit.xlsx"

SEED = 42
TIER_NAME = {0: "off-topic", 1: "synthesis", 2: "rct",
             3: "human-observational", 4: "animal", 5: "in-vitro",
             6: "review"}

GUIDE = [
    ("What you are checking", "", ""),
    ("Each row on the Pairs sheet is one spice-disease pair that Phase 6/7 "
     "reported on. The Papers sheet lists the actual papers behind it. Read "
     "the titles, then answer three questions per pair.", "", ""),
    ("", "", ""),
    ("Q1  disease_ok", "Is the 'disease' a real clinical outcome?",
     "yes = 'liver injury', 'asthma'.  no = 'cytotoxicity', 'proliferation', "
     "'oxidative stress' -- assay words mined from the methods section."),
    ("Q2  tier_ok", "Do the tiers look right for these papers?",
     "Spot-check 2-3. 'Wistar rats' should be animal; 'HepG2 cells' should be "
     "in-vitro; 'randomised 120 patients' should be rct."),
    ("Q3  direction_ok", "Does the positive/negative split match the titles?",
     "If 9 of 10 are marked positive, do the titles actually report a "
     "beneficial effect? Flag if the sentiment looks inverted or arbitrary."),
    ("", "", ""),
    ("verdict", "Overall: is this pair's number trustworthy?",
     "ok = usable as reported.  minor = usable with a caveat.  "
     "bad = should be excluded from the headline figures."),
    ("", "", ""),
    ("Note", "Bucket C is a RANDOM control sample.", "Do not skip it -- it is "
     "what makes the error rate interpretable. Without it you cannot tell a "
     "targeted error rate from the ordinary one."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=15)
    ap.add_argument("--max-papers", type=int, default=10,
                    help="papers listed per pair")
    args = ap.parse_args()

    if not PAIRS.exists():
        raise SystemExit("controversy_pairs.csv missing -- run "
                         "evidence_controversy.py first")
    p = pd.read_csv(PAIRS)
    n = args.per_bucket

    # --- A: preclinical-hype motif (needs evidence on both sides) ---
    both = p[(p.hum_n >= 3) & (p.pre_n >= 3)]
    a = both.sort_values("hype_gap", ascending=False).head(n).copy()
    a["bucket"] = "A_hype"

    # --- B: most contested, excluding anything already in A ---
    rest = p[~p.index.isin(a.index)]
    b = rest.sort_values("w_controversy", ascending=False).head(n).copy()
    b["bucket"] = "B_contested"

    # --- C: random control ---
    rest2 = p[~p.index.isin(a.index) & ~p.index.isin(b.index)]
    c = rest2.sample(n=min(n, len(rest2)), random_state=SEED).copy()
    c["bucket"] = "C_random"

    sel = pd.concat([a, b, c], ignore_index=True)
    print(f"Selected {len(sel)} pairs: "
          + ", ".join(f"{k} {v}" for k, v in
                      sel.bucket.value_counts().sort_index().items()))

    # --- pull the papers behind each pair ---
    want = set(zip(sel.Spice, sel.Disease))
    db = pd.read_csv(DB, usecols=["Spice", "Disease", "PubMed ID", "Title",
                                  "Association Nature", "evidence_tier",
                                  "tier_confidence"])
    db = db[[t in want for t in zip(db.Spice, db.Disease)]]
    # itertuples mangles names with spaces into _3/_5 positionals; rename so
    # the loop below reads by name and cannot silently grab the wrong column
    db = db.rename(columns={"PubMed ID": "pmid",
                            "Association Nature": "assoc"})
    print(f"Pulled {len(db):,} underlying associations")

    wb = Workbook()

    # ---------- Guidelines ----------
    ws = wb.active
    ws.title = "Guidelines"
    ws["A1"] = "EvidenceRx -- Phase 9 confounder audit"
    ws["A1"].font = Font(bold=True, size=14)
    r = 3
    for col1, col2, col3 in GUIDE:
        ws.cell(r, 1, col1).font = Font(bold=bool(col1 and not col1.startswith(" ")))
        ws.cell(r, 2, col2)
        ws.cell(r, 3, col3).alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 46
    ws.column_dimensions["C"].width = 70

    # ---------- Pairs ----------
    ps = wb.create_sheet("Pairs")
    head = ["pair_id", "bucket", "spice", "disease", "n", "%pos",
            "human_n", "human_%pos", "preclin_n", "preclin_%pos", "hype_gap",
            "controversy", "disease_ok", "tier_ok", "direction_ok", "verdict",
            "notes"]
    ps.append(head)
    for cell in ps[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2E75B6")
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, r_ in enumerate(sel.itertuples(), 1):
        ps.append([i, r_.bucket, r_.Spice, r_.Disease, int(r_.n),
                   round(r_.pct_pos, 1), int(r_.hum_n),
                   None if pd.isna(r_.hum_pct_pos) else round(r_.hum_pct_pos, 1),
                   int(r_.pre_n),
                   None if pd.isna(r_.pre_pct_pos) else round(r_.pre_pct_pos, 1),
                   None if pd.isna(r_.hype_gap) else round(r_.hype_gap, 1),
                   round(r_.w_controversy, 2), "", "", "", "", ""])
    yn = DataValidation(type="list", formula1='"yes,no,unsure"', allow_blank=True)
    vd = DataValidation(type="list", formula1='"ok,minor,bad"', allow_blank=True)
    ps.add_data_validation(yn)
    ps.add_data_validation(vd)
    last = len(sel) + 1
    for col in ("M", "N", "O"):
        yn.add(f"{col}2:{col}{last}")
    vd.add(f"P2:P{last}")
    for col, w in zip("ABCDEFGHIJKLMNOPQ",
                      [7, 13, 16, 26, 6, 7, 8, 10, 9, 11, 9, 11, 11, 9, 12, 9, 34]):
        ps.column_dimensions[col].width = w
    ps.freeze_panes = "C2"

    # ---------- Papers ----------
    pp = wb.create_sheet("Papers")
    pp.append(["pair_id", "counted", "spice", "disease", "pmid", "tier",
               "tier_class", "conf", "association", "title"])
    for cell in pp[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="548235")
        cell.alignment = Alignment(horizontal="center")
    for i, r_ in enumerate(sel.itertuples(), 1):
        sub = db[(db.Spice == r_.Spice) & (db.Disease == r_.Disease)].copy()
        # tier 0 is EXCLUDED from the Phase 6/7 numbers, so it must not head
        # the list -- sorting by tier alone put off-topic papers first and
        # invited judgements on rows that never contributed to the figure.
        sub["counted"] = sub.evidence_tier.between(1, 6)
        sub = sub.sort_values(["counted", "evidence_tier"],
                              ascending=[False, True]).head(args.max_papers)
        for s in sub.itertuples():
            t = s.evidence_tier
            t = int(t) if pd.notna(t) else None
            pp.append([i, "yes" if s.counted else "no (off-topic)",
                       r_.Spice, r_.Disease, s.pmid, t,
                       TIER_NAME.get(t, ""),
                       None if pd.isna(s.tier_confidence)
                       else round(s.tier_confidence, 2),
                       s.assoc, str(s.Title)[:300]])
    for col, w in zip("ABCDEFGHIJ", [7, 15, 14, 22, 11, 6, 20, 6, 12, 95]):
        pp.column_dimensions[col].width = w
    pp.freeze_panes = "E2"
    for row in pp.iter_rows(min_row=2):
        row[9].alignment = Alignment(wrap_text=True, vertical="top")

    wb.save(OUT)
    print(f"\nWrote {OUT.name}")
    print(f"  Pairs sheet  : {len(sel)} pairs to judge")
    print(f"  Papers sheet : {pp.max_row-1} paper rows "
          f"(up to {args.max_papers} per pair, strongest tier first)")
    print("\nFill disease_ok / tier_ok / direction_ok / verdict on the Pairs "
          "sheet,\nthen run: python evidence_confounder_score.py")


if __name__ == "__main__":
    main()
