"""
EvidenceRx — Phase 2: weak evidence-tier labels (PDF 6-tier taxonomy).

Tiers (strongest -> weakest), matching the problem spec:
    1 synthesis           meta-analysis / systematic review of human trials
    2 rct                 randomized controlled trial
    3 human-observational non-randomized human (cohort, case-control,
                          observational, case reports, non-RCT trials)
    4 animal              animal study (mice, rats, in vivo)
    5 in-vitro            cell line / biochemical assay
    6 review              narrative review, opinion, or in-silico only

Two labellers combined, metadata first (high precision), then text rules
(recover un-indexed papers that have an abstract but no MeSH), per the spec:
    - Metadata rules : PublicationType + MeSH  -> tiers 1-5
    - Text rules     : regex over the abstract -> tiers 2-6
Tuned for PRECISION, not coverage: unresolved cases ABSTAIN for the model.

A per-paper `is_case_report` flag is kept for the harm-surveillance
sub-analysis (case reports are folded into tier 3 but flagged).

Input : pubmed_metadata_cache.json, articles_cache.json
Output: pmid_studytype.csv  (pmid, tier, studytype, rule, is_case_report)

Usage:
    python evidence_label_studytype.py
    python evidence_label_studytype.py --selftest
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
META_CACHE = BASE_DIR / "pubmed_metadata_cache.json"
ARTICLES = BASE_DIR / "articles_cache.json"
OUT_CSV = BASE_DIR / "pmid_studytype.csv"

# tier number -> class name (1 strongest .. 6 weakest)
TIERS = {1: "synthesis", 2: "rct", 3: "human-observational",
         4: "animal", 5: "in-vitro", 6: "review"}
CLASS_ORDER = [TIERS[t] for t in range(1, 7)]

# ---------------- metadata vocabularies (lowercased) --------------------
SYNTHESIS_PT = {"meta-analysis", "systematic review"}
RCT_PT = {"randomized controlled trial"}
NONRAND_HUMAN_PT = {
    "clinical trial", "controlled clinical trial", "clinical trial, phase i",
    "clinical trial, phase ii", "clinical trial, phase iii",
    "clinical trial, phase iv", "pragmatic clinical trial",
    "observational study", "multicenter study",
}
CASE_REPORT_PT = {"case reports"}
REVIEW_PT = {"review"}   # narrative review (systematic review handled above)

EPI_MESH = {
    "cohort studies", "case-control studies", "cross-sectional studies",
    "prospective studies", "retrospective studies", "follow-up studies",
    "longitudinal studies", "incidence", "prevalence", "risk factors",
    "odds ratio", "proportional hazards models",
}
ANIMAL_MESH = {
    "animals", "mice", "rats", "rats, sprague-dawley", "rats, wistar",
    "mice, inbred balb c", "mice, inbred c57bl", "mice, nude",
    "disease models, animal", "rabbits", "dogs", "swine", "zebrafish",
    "drosophila melanogaster", "caenorhabditis elegans", "chickens",
    "guinea pigs", "rats, inbred strains", "cattle", "sheep",
}
INVITRO_MESH = {
    "in vitro techniques", "cells, cultured", "cell line", "cell line, tumor",
    "tumor cells, cultured", "hela cells", "cell survival",
    "cell proliferation", "apoptosis", "structure-activity relationship",
    "mcf-7 cells", "caco-2 cells", "hep g2 cells",
}

# ---------------- text-rule regexes (high precision) --------------------
_CELL_LINES = (r"HeLa|MCF-?7|HCT-?116|Caco-?2|HepG2|A549|PC-?3|MDA-MB-\d+|"
               r"SH-SY5Y|Jurkat|THP-1|RAW\s?264\.7|HT-?29|SW480|HEK-?293|U87")
RX_INVITRO = re.compile(
    r"\bin vitro\b|\bIC\s?50\b|IC₅₀|\b\d+(?:\.\d+)?\s?[µμu]M\b|"
    r"\bcell lines?\b|\bcytotoxic|\bcultured\s+\w*\s?cells?\b|(?:" + _CELL_LINES + r")",
    re.IGNORECASE)
# Strong, unambiguous in-vitro markers (dose in uM, IC50, named cell lines).
# These override animal/human text rules -- e.g. cultured animal cells at 10 uM
# is an in-vitro study, not an animal study.
RX_INVITRO_STRONG = re.compile(
    r"\bIC\s?50\b|IC₅₀|\b\d+(?:\.\d+)?\s?[µμu]M\b|\bcell lines?\b|"
    r"(?:" + _CELL_LINES + r")",
    re.IGNORECASE)
RX_ANIMAL = re.compile(
    r"\bmice\b|\brats?\b|\bmurine\b|\bWistar\b|\bSprague-?Dawley\b|"
    r"\bin vivo\b|\bC57BL|\bBALB/?c\b|\b(?:mouse|rat) model\b|"
    r"\badministered orally to\b|\bzebrafish\b",
    re.IGNORECASE)
RX_HUMAN = re.compile(
    r"\bpatients?\b|\bparticipants?\b|\bhealthy volunteers?\b|\benrolled\b|"
    r"\bhuman subjects?\b|\bin patients with\b",
    re.IGNORECASE)
RX_RCT = re.compile(
    r"\brandomi[sz]ed\b[^.]{0,40}\b(?:controlled|trial|double-?blind|placebo)\b|"
    r"\bdouble-?blind[^.]{0,30}\b(?:randomi[sz]ed|placebo)\b|"
    r"\bplacebo-controlled\b[^.]{0,30}\btrial\b",
    re.IGNORECASE)
RX_INSILICO = re.compile(
    r"\bin silico\b|\bmolecular docking\b|\bmolecular dynamics simulation|"
    r"\bcomputational (?:model|screening|analysis)\b|\bdocking (?:study|analysis|simulation)\b",
    re.IGNORECASE)


def _lc(items):
    return {s.lower() for s in items}


def classify(pub_types, mesh, abstract):
    """Return (tier|None, studytype|'', rule, is_case_report)."""
    pt = _lc(pub_types)
    mh = _lc(mesh)
    text = abstract or ""
    has_humans = "humans" in mh
    has_animals = bool(mh & ANIMAL_MESH)
    is_cr = bool(pt & CASE_REPORT_PT)

    # ---- metadata: strongest human signals ----
    if pt & SYNTHESIS_PT:
        # tier 1 is synthesis OF HUMAN TRIALS. A systematic review of in-vitro
        # or animal work (no trial/patient signal) is tier 6, not tier 1.
        t_human = re.search(r"\b(trial|randomi|patient|clinical|cohort|"
                            r"participant|volunteer)", text, re.I)
        t_preclin = re.search(r"\b(in vitro|in silico|animal model|mice|"
                              r"murine|preclinical|molecular docking)", text, re.I)
        if t_preclin and not t_human:
            return 6, TIERS[6], "M1b:synthesis-preclinical->review", False
        return 1, TIERS[1], "M1:synthesis-pt", False
    if pt & RCT_PT:
        return 2, TIERS[2], "M2:rct-pt", False
    if (pt & NONRAND_HUMAN_PT) or is_cr:
        return 3, TIERS[3], "M3:nonrand-human-pt", is_cr
    if has_humans and (mh & EPI_MESH):
        return 3, TIERS[3], "M3:humans+epi-mesh", False

    # ---- narrative review (before text rules, so its topic MeSH/text
    #      about the studies it discusses doesn't misfire) ----
    if pt & REVIEW_PT:
        return 6, TIERS[6], "M6:review-pt", False

    # ---- metadata: organism from MeSH ----
    if has_animals and not has_humans:
        return 4, TIERS[4], "M4:animal-mesh", False
    if mh & INVITRO_MESH:
        return 5, TIERS[5], "M5:invitro-mesh", False

    # ---- text rules (recover un-indexed / metadata-poor papers) ----
    if RX_RCT.search(text):
        return 2, TIERS[2], "T2:rct-text", False
    if RX_INSILICO.search(text) and not RX_ANIMAL.search(text) \
            and not RX_HUMAN.search(text):
        return 6, TIERS[6], "T6:insilico-text", False
    # strong in-vitro markers dominate animal/human text (cultured cells, IC50,
    # uM doses) -- fixes 'cultured rat cells at 10 uM' -> animal mislabel
    if RX_INVITRO_STRONG.search(text):
        return 5, TIERS[5], "T5:invitro-strong", False
    if RX_HUMAN.search(text) and not RX_ANIMAL.search(text):
        return 3, TIERS[3], "T3:human-text", False
    if RX_ANIMAL.search(text):
        return 4, TIERS[4], "T4:animal-text", False
    if RX_INVITRO.search(text):
        return 5, TIERS[5], "T5:invitro-text", False

    # ---- indexed floor: MeSH present but no organism -> in-vitro/lab ----
    if mh and not has_humans and not has_animals:
        return 5, TIERS[5], "M5:indexed-floor", False

    # ---- unresolved -> abstain for the model ----
    if not mh:
        return None, "", "abstain:no-mesh-no-text", False
    return None, "", "abstain:unresolved", False


# ------------------------------------------------------------------------
def selftest():
    T = lambda *a: classify(*a)[:2]
    cases = [
        # (pub_types, mesh, abstract, expected_tier)
        (["Meta-Analysis"], ["Humans"], "", 1),
        (["Systematic Review"], [], "", 1),
        (["Randomized Controlled Trial"], ["Humans"], "", 2),
        # animal-only RCT tag -> still tier 2 by pub-type (rare; pub-type trusted)
        (["Randomized Controlled Trial"], ["Animals"], "", 2),
        (["Clinical Trial"], ["Humans"], "", 3),          # non-RCT -> tier 3
        (["Observational Study"], ["Humans"], "", 3),
        (["Case Reports"], ["Humans"], "", 3),            # folded into tier 3
        (["Journal Article"], ["Humans", "Cohort Studies"], "", 3),
        (["Journal Article"], ["Animals", "Rats"], "", 4),
        (["Journal Article"], ["Cells, Cultured"], "", 5),
        (["Journal Article"], ["Bacteria", "Colony Count, Microbial"], "", 5),
        (["Review"], ["Animals"], "", 6),                 # narrative review -> tier 6
        # ---- text rules on UN-INDEXED papers (no MeSH) ----
        (["Journal Article"], [],
         "A randomized, double-blind, placebo-controlled trial of 120 patients", 2),
        (["Journal Article"], [],
         "Curcumin induced apoptosis in HCT-116 colon cancer cells at 25 uM", 5),
        (["Journal Article"], [],
         "Curcumin was administered orally to Wistar rats for 8 weeks", 4),
        (["Journal Article"], [],
         "A survey of 340 patients with rheumatic disease and their diets", 3),
        (["Journal Article"], [],
         "Molecular docking and in silico ADMET analysis of piperine", 6),
        # un-indexed, no signal -> abstain
        (["Journal Article"], [], "The chromosome-level genome of ginseng", None),
    ]
    ok = True
    for pt, mh, ab, exp in cases:
        tier, name = T(pt, mh, ab)
        status = "PASS" if tier == exp else "FAIL"
        if tier != exp:
            ok = False
        print(f"  [{status}] tier={str(tier):4s} {str(name):20s} "
              f"expected={exp}  :: {(ab or (pt[0]+' '+str(mh)))[:50]}")
    print("\nSELFTEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def run():
    for p in (META_CACHE, ARTICLES):
        if not p.exists():
            sys.exit(f"Missing {p.name}.")
    cache = json.load(open(META_CACHE, encoding="utf-8"))
    arts = json.load(open(ARTICLES, encoding="utf-8"))

    tier_counts, rule_counts = Counter(), Counter()
    n_cr = 0
    rows = []
    for pmid, rec in cache.items():
        abstract = (arts.get(pmid, {}) or {}).get("abstract", "") or ""
        tier, name, rule, is_cr = classify(
            rec.get("pub_types", []), rec.get("mesh", []), abstract)
        rows.append([pmid, tier if tier else "", name, rule, int(is_cr)])
        tier_counts[name if name else "ABSTAIN"] += 1
        rule_counts[rule] += 1
        n_cr += is_cr

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "tier", "studytype", "rule", "is_case_report"])
        w.writerows(rows)

    total = len(rows)
    labeled = total - tier_counts["ABSTAIN"]
    print(f"Papers processed : {total:,}")
    print(f"Weakly labeled   : {labeled:,} ({labeled/total*100:.1f}%)")
    print(f"Abstained        : {tier_counts['ABSTAIN']:,} "
          f"({tier_counts['ABSTAIN']/total*100:.1f}%)")
    print(f"Case reports flagged (within tier 3): {n_cr:,}")
    print("\n=== Evidence-tier distribution (1 strongest .. 6 weakest) ===")
    for t in range(1, 7):
        c = tier_counts.get(TIERS[t], 0)
        print(f"  tier {t} {TIERS[t]:20s} {c:7,d}  ({c/total*100:5.1f}%)")
    print(f"  {'ABSTAIN':27s} {tier_counts['ABSTAIN']:7,d}  "
          f"({tier_counts['ABSTAIN']/total*100:5.1f}%)")
    print("\n=== Rule firing counts (metadata M* vs text T*) ===")
    for rule, c in rule_counts.most_common():
        print(f"  {c:7,d}  {rule}")
    print(f"\nWrote {OUT_CSV.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run()
