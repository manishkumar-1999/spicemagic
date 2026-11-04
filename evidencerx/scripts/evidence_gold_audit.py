"""
EvidenceRx -- deep quality audit of the gold annotation.

Flags likely annotation errors using only HIGH-PRECISION signals, so false
alarms stay low:
  A. Curated PublicationType contradictions (RCT / Meta-Analysis / Case Report
     tags are reliable study-type indicators).
  B. Rare high-tier OVER-calls: gold=synthesis/rct with no supporting
     synthesis/randomization signal anywhere -- these are the most damaging
     errors because they inflate the human-evidence tiers that drive the
     headline result.
  C. animal / in-vitro calls with no organism/lab signal at all.

We deliberately DO NOT flag 'Humans MeSH but gold=in-vitro' -- that tag appears
on in-vitro work using human cells or studying human pathogens, and the
annotator is usually right there.

Writes evidence_gold_review.xlsx with the flagged rows to re-check.

Usage:
    python evidence_gold_audit.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, Alignment, PatternFill

BASE = Path(__file__).parent
GOLD = BASE / "evidence_gold_sample_annotated.xlsx"
META = BASE / "pubmed_metadata_cache.json"
OUT = BASE / "evidence_gold_review.xlsx"

ANIMAL_MESH = {"animals", "mice", "rats", "rats, sprague-dawley", "rats, wistar",
               "disease models, animal", "rabbits", "swine", "zebrafish",
               "guinea pigs", "dogs", "cattle", "sheep"}
RX_RANDOM = re.compile(r"\brandomi[sz]", re.I)
RX_TRIAL = re.compile(r"\btrial\b|\bplacebo\b|\bdouble-?blind\b", re.I)
RX_SYNTH = re.compile(r"\bmeta-analysis\b|\bsystematic review\b|"
                      r"\bscoping review\b|\bumbrella review\b|\bpooled\b|"
                      r"\bsearched\b.{0,40}\bdatabases?\b", re.I)
RX_ANIMAL = re.compile(
    r"\bmice\b|\bmouse\b|\brats?\b|\bmurine\b|\bin vivo\b|\bWistar\b|"
    r"\bSprague-?Dawley\b|\bzebrafish\b|\brabbits?\b|\bsheep\b|\brumen\b|"
    r"\brumin(?:al|ant)\b|\bpigs?\b|\bporcine\b|\bswine\b|\bbovine\b|"
    r"\bcattle\b|\bcal(?:f|ves)\b|\bcanine\b|\bdogs?\b|\bfeline\b|\bcats?\b|"
    r"\bpoultry\b|\bchickens?\b|\bbroilers?\b|\bgoats?\b|\bcaprine\b|"
    r"\bbuffalo\b|\btrout\b|\btilapia\b|\bshrimp\b|\bdrosophila\b|"
    r"\bcaenorhabditis\b|\bnematode\b|\bhamsters?\b|\bgerbils?\b|"
    r"\bmonkeys?\b|\bprimates?\b|\bovine\b|\bequine\b|\bhorses?\b|"
    r"\blarvae\b|\banimals?\b", re.I)
RX_HUMAN = re.compile(r"\bpatients?\b|\bparticipants?\b|\bvolunteers?\b|"
                      r"\benrolled\b|\bsubjects?\b", re.I)
TIER = {0: "off-topic", 1: "synthesis", 2: "rct", 3: "human-observational",
        4: "animal", 5: "in-vitro", 6: "review"}


def main():
    wb = load_workbook(GOLD, read_only=True, data_only=True)
    ws = wb["Annotation"]
    rows = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[1] is None:
            continue
        rows[str(r[1])] = {"gold": int(r[4]), "title": r[2] or "",
                           "abstract": r[3] or ""}
    meta = json.load(open(META, encoding="utf-8"))

    print(f"Gold papers: {len(rows)}")
    print("Distribution:", {TIER[k]: v for k, v in
                            sorted(Counter(d['gold'] for d in rows.values()).items())})
    print()

    flags = []
    for p, d in rows.items():
        g = d["gold"]
        # search title + abstract together (study-type is often stated in the
        # title, e.g. "...: A Randomized Controlled Trial")
        ab = (d["title"] or "") + ". " + (d["abstract"] or "")
        pt = {t.lower() for t in meta.get(p, {}).get("pub_types", [])}
        mh = {t.lower() for t in meta.get(p, {}).get("mesh", [])}
        ani_mesh = bool(mh & ANIMAL_MESH)

        reason = None
        sev = ""
        # A. curated pub-type contradictions
        if "randomized controlled trial" in pt and g not in (2, 0, 3):
            reason, sev = f"pub-type=RCT but gold={TIER[g]}", "high"
        elif (pt & {"meta-analysis", "systematic review"}) and g in (2, 3, 4, 5):
            reason, sev = f"pub-type=Meta/SR but gold={TIER[g]}", "high"
        elif "case reports" in pt and g in (1, 2, 4, 5):
            reason, sev = f"pub-type=CaseReport but gold={TIER[g]}", "med"
        # B. rare high-tier OVER-calls (most damaging)
        elif g == 1 and not (pt & {"meta-analysis", "systematic review"}) \
                and not RX_SYNTH.search(ab):
            reason, sev = "gold=synthesis but no meta/SR signal in text", "high"
        elif g == 2 and "randomized controlled trial" not in pt \
                and not RX_RANDOM.search(ab):
            reason, sev = "gold=RCT but no 'randomi*' signal anywhere", "high"
        # C. organism call with no organism signal
        elif g == 4 and not ani_mesh and not RX_ANIMAL.search(ab):
            reason, sev = "gold=animal but no animal MeSH/text", "med"
        elif g == 3 and not RX_HUMAN.search(ab) and "humans" not in mh \
                and not (pt & {"observational study", "case reports",
                               "clinical trial"}):
            reason, sev = "gold=human-obs but no human signal", "low"

        if reason:
            flags.append((sev, p, g, d["title"], reason))

    order = {"high": 0, "med": 1, "low": 2}
    flags.sort(key=lambda x: order[x[0]])
    by_sev = Counter(f[0] for f in flags)
    print(f"Flagged for review: {len(flags)}  "
          f"(high={by_sev['high']}, med={by_sev['med']}, low={by_sev['low']})")
    print(f"  = {len(flags)/len(rows)*100:.1f}% of the gold set\n")
    for sev, p, g, title, reason in flags[:15]:
        print(f"  [{sev:4s}] PMID {p} gold={TIER[g]:18s} | {reason}")

    # write review workbook
    out = Workbook()
    s = out.active
    s.title = "Review"
    hdr = ["severity", "pmid", "gold_tier", "gold_class", "flag_reason",
           "title", "corrected_tier"]
    s.append(hdr)
    for c in s[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="C0392B")
    for sev, p, g, title, reason in flags:
        s.append([sev, p, g, TIER[g], reason, title[:120], ""])
    s.column_dimensions["E"].width = 42
    s.column_dimensions["F"].width = 70
    s.column_dimensions["G"].width = 14
    s.freeze_panes = "A2"
    out.save(OUT)
    print(f"\nWrote {OUT.name} -- re-check flagged rows; put the right tier in "
          f"'corrected_tier' where you agree it's wrong.")


if __name__ == "__main__":
    main()
