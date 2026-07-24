"""
Enhanced Relation Classifier for Spice-Disease Associations

Upgrades the basic keyword heuristic with:
1. NegEx-based negation detection ("did NOT reduce", "no effect on")
2. Hedging/uncertainty detection ("may", "might", "could potentially")
3. Scope-aware keyword matching (only keywords near spice+disease mentions)
4. Study-type detection (in-vivo, in-vitro, clinical trial, review)

Input:  spice_disease_associations.csv  (original)
Output: spice_disease_associations_enhanced.csv (improved classifications)

Does NOT modify the original file.
"""

import csv
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "spice_disease_associations.csv"
OUTPUT_CSV = BASE_DIR / "spice_disease_associations_enhanced.csv"


# ═══════════════════════════════════════════════════════════════════
# NEGATION DETECTION (NegEx-inspired)
# ═══════════════════════════════════════════════════════════════════

# Pre-negation cues: negation word BEFORE the target
PRE_NEGATION_CUES = [
    r'\bno\b', r'\bnot\b', r'\bnor\b', r'\bnever\b',
    r'\bwithout\b', r'\babsence\s+of\b', r'\black\s+of\b',
    r'\bfailed\s+to\b', r'\bfailure\s+to\b',
    r'\bdid\s+not\b', r'\bdoes\s+not\b', r'\bdo\s+not\b',
    r'\bwas\s+not\b', r'\bwere\s+not\b', r'\bis\s+not\b',
    r'\bcannot\b', r'\bcould\s+not\b', r'\bwould\s+not\b',
    r'\bno\s+significant\b', r'\bno\s+effect\b',
    r'\bno\s+evidence\b', r'\binsufficient\s+evidence\b',
    r'\bnon[\-]?significant\b', r'\bnonsignificant\b',
    r'\bineffective\b', r'\bneither\b',
    r"\bdidn't\b", r"\bdoesn't\b", r"\bwasn't\b", r"\bweren't\b",
    r"\bcan't\b", r"\bcouldn't\b", r"\bwon't\b",
    r'\brather\s+than\b', r'\binstead\s+of\b',
    r'\bdenied\b', r'\bexcluded?\b',
    r'\bunlikely\b', r'\bimprobable\b',
]

# Post-negation cues: negation word AFTER the target
POST_NEGATION_CUES = [
    r'\bwas\s+not\s+(?:observed|found|detected|seen|noted)\b',
    r'\bwere\s+not\s+(?:observed|found|detected|seen|noted)\b',
    r'\bnot\s+(?:observed|found|detected|seen|noted|confirmed)\b',
]

PRE_NEG_PATTERN = re.compile('|'.join(PRE_NEGATION_CUES), re.IGNORECASE)
POST_NEG_PATTERN = re.compile('|'.join(POST_NEGATION_CUES), re.IGNORECASE)

# Negation scope: how many words after a negation cue to check
NEGATION_SCOPE = 6  # words


# ═══════════════════════════════════════════════════════════════════
# HEDGING / UNCERTAINTY DETECTION
# ═══════════════════════════════════════════════════════════════════

HEDGE_CUES = re.compile(
    r'\b(?:'
    r'may\b|might\b|could\b|possibly|potentially|'
    r'suggest(?:s|ed|ing)?\b|'
    r'appear(?:s|ed)?\s+to\b|'
    r'seem(?:s|ed)?\s+to\b|'
    r'hypothesiz|speculate[sd]?|'
    r'warrant(?:s|ed)?\s+further|'
    r'further\s+(?:study|research|investigation)\s+(?:is|are)\s+needed|'
    r'remain(?:s)?\s+(?:unclear|uncertain|unknown)|'
    r'preliminary|'
    r'inconclusive'
    r')',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# POSITIVE / NEGATIVE KEYWORD PATTERNS (improved)
# ═══════════════════════════════════════════════════════════════════

POSITIVE_KEYWORDS = re.compile(
    r'\b(?:'
    r'treat(?:s|ed|ing|ment)?\b|'
    r'prevent(?:s|ed|ing|ion|ive)?\b|prophylaxis\b|prophylactic\b|'
    r'inhibit(?:s|ed|ing|ion|ory)?\b|'
    r'reduc(?:e[sd]?|ing|tion)\b|'
    r'alleviat\w*|ameliorat\w*|attenuat\w*|mitigat\w*|suppress\w*|'
    r'protect\w*|'
    r'anti[\-]?(?:inflammatory|oxidant|cancer|tumor|tumour|microbial|'
    r'bacterial|viral|fungal|diabetic|hypertensive|nociceptive|proliferative|'
    r'apoptotic|mutagenic|genotoxic)\b|'
    r'therapeutic\b|beneficial\b|efficac\w*|effective\b|'
    r'heal(?:s|ed|ing)?\b|'
    r'improv(?:e[sd]?|ing|ement)\b|'
    r'decreas(?:e[sd]?|ing)\b|'
    r'lower(?:s|ed|ing)?\b|'
    r'reliev\w*|revers\w*|restor\w*|recover\w*|'
    r'upregulat\w*|downregulat\w*|modulat\w*|'
    r'apoptosis\b|cytotoxic\w*\s+(?:against|toward|on)\b|'
    r'scaveng\w*|neuroprotect\w*|hepatoprotect\w*|cardioprotect\w*|'
    r'chemoprevent\w*|antiproliferat\w*'
    r')',
    re.IGNORECASE,
)

NEGATIVE_KEYWORDS = re.compile(
    r'\b(?:'
    r'caus(?:e[sd]?|ing)\b|'
    r'induc(?:e[sd]?|ing|tion)\b|'
    r'worsen(?:s|ed|ing)?\b|'
    r'exacerbat\w*|aggravat\w*|'
    r'toxic(?:ity)?\b|poisoning\b|'
    r'adverse\b|side\s+effect\w*\b|'
    r'allerg(?:ic|y|en)\w*\b|hypersensitiv\w*|contraindic\w*|'
    r'increas(?:e[sd]?)?\s+risk\b|'
    r'hepatotoxic\w*|nephrotoxic\w*|neurotoxic\w*|'
    r'carcinogen\w*|mutagen\w*|teratogen\w*|'
    r'overdos\w*|lethal\b|fatal\b|'
    r'damag(?:e[sd]?|ing)\b|'
    r'promot(?:e[sd]?|ing)\s+(?:cancer|tumor|tumour|growth|progression)\b'
    r')',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# STUDY TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════

STUDY_PATTERNS = {
    "clinical_trial": re.compile(
        r'\b(?:clinical\s+trial|randomized|randomised|placebo[\-\s]controlled|'
        r'double[\-\s]blind|single[\-\s]blind|crossover|phase\s+[I1-4]|'
        r'RCT|controlled\s+trial|patients?\s+(?:were|received))\b',
        re.IGNORECASE,
    ),
    "in_vivo": re.compile(
        r'\b(?:in\s+vivo|animal\s+(?:model|study|experiment)|'
        r'mice|mouse|rats?\b|rabbits?\b|'
        r'(?:male|female)\s+(?:rats?|mice|rabbits?)|'
        r'Swiss\s+albino|Wistar|Sprague[\-\s]Dawley|BALB|C57BL)\b',
        re.IGNORECASE,
    ),
    "in_vitro": re.compile(
        r'\b(?:in\s+vitro|cell\s+(?:line|culture|viability)|'
        r'HeLa|MCF[\-]?7|A549|HepG2|Caco[\-]?2|'
        r'MTT\s+assay|cytotoxicity\s+assay|'
        r'IC50|EC50|MIC\b)\b',
        re.IGNORECASE,
    ),
    "review": re.compile(
        r'\b(?:systematic\s+review|meta[\-\s]analysis|'
        r'literature\s+review|narrative\s+review|'
        r'we\s+review(?:ed)?|this\s+review)\b',
        re.IGNORECASE,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# CORE CLASSIFICATION LOGIC
# ═══════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def is_negated(text: str, keyword_match) -> bool:
    """Check if a keyword match is negated using NegEx-style detection.

    Looks for negation cues within NEGATION_SCOPE words before/after the keyword.
    """
    match_start = keyword_match.start()
    match_end = keyword_match.end()

    # Get text before the keyword (up to NEGATION_SCOPE words)
    text_before = text[:match_start]
    words_before = text_before.split()
    scope_before = ' '.join(words_before[-NEGATION_SCOPE:]) if words_before else ''

    # Check for pre-negation cues
    if PRE_NEG_PATTERN.search(scope_before):
        return True

    # Get text around keyword for post-negation patterns
    text_around = text[max(0, match_start - 50):min(len(text), match_end + 50)]
    if POST_NEG_PATTERN.search(text_around):
        return True

    return False


def classify_enhanced(abstract: str, title: str, spice: str, disease: str) -> dict:
    """Enhanced classification with negation detection, hedging, and study type.

    Returns dict with:
      - association_nature_enhanced: "positive", "negative", "neutral", "negated"
      - is_negated: True/False
      - is_hedged: True/False
      - study_type: "clinical_trial", "in_vivo", "in_vitro", "review", ""
      - confidence_modifier: float multiplier for confidence adjustment
    """
    full_text = f"{title} {abstract}" if abstract else title or ""

    if not full_text.strip():
        return {
            "association_nature_enhanced": "neutral",
            "is_negated": False,
            "is_hedged": False,
            "study_type": "",
            "confidence_modifier": 1.0,
        }

    # Find relevant sentences (containing both spice and disease)
    relevant_sents = []
    spice_lower = spice.lower()
    disease_lower = disease.lower()

    for sent in split_sentences(full_text):
        sent_lower = sent.lower()
        if spice_lower in sent_lower and disease_lower in sent_lower:
            relevant_sents.append(sent)

    # Use relevant sentences if available, otherwise full abstract
    analysis_text = ' '.join(relevant_sents) if relevant_sents else full_text

    # Count positive keywords and check for negation
    pos_matches = list(POSITIVE_KEYWORDS.finditer(analysis_text))
    neg_matches = list(NEGATIVE_KEYWORDS.finditer(analysis_text))

    # Check each positive keyword for negation
    pos_affirmed = 0
    pos_negated = 0
    for m in pos_matches:
        if is_negated(analysis_text, m):
            pos_negated += 1
        else:
            pos_affirmed += 1

    # Check each negative keyword for negation
    neg_affirmed = 0
    neg_negated = 0
    for m in neg_matches:
        if is_negated(analysis_text, m):
            neg_negated += 1
        else:
            neg_affirmed += 1

    # Detect hedging
    hedge_count = len(HEDGE_CUES.findall(analysis_text))
    is_hedged = hedge_count >= 2  # Multiple hedge words = uncertain

    # Detect study type
    study_type = ""
    for stype, pattern in STUDY_PATTERNS.items():
        if pattern.search(full_text):
            study_type = stype
            break

    # Classification logic
    total_pos = pos_affirmed
    total_neg = neg_affirmed

    # If positive keywords were mostly negated ("did not reduce", "no improvement")
    mostly_negated = (pos_negated > pos_affirmed) and pos_negated > 0

    if mostly_negated:
        nature = "negated_positive"
    elif total_pos > 0 and total_neg == 0:
        nature = "positive"
    elif total_neg > 0 and total_pos == 0:
        nature = "negative"
    elif total_pos > total_neg:
        nature = "positive"
    elif total_neg > total_pos:
        nature = "negative"
    else:
        nature = "neutral"

    # Confidence modifier
    conf_modifier = 1.0

    # Hedging reduces confidence
    if is_hedged:
        conf_modifier *= 0.85

    # Negated results reduce confidence
    if mostly_negated:
        conf_modifier *= 0.7

    # Study type affects confidence
    if study_type == "clinical_trial":
        conf_modifier *= 1.15  # Clinical trials are stronger evidence
    elif study_type == "review":
        conf_modifier *= 1.10  # Reviews synthesize multiple studies
    elif study_type == "in_vitro":
        conf_modifier *= 0.90  # In-vitro is weaker than in-vivo

    return {
        "association_nature_enhanced": nature,
        "is_negated": mostly_negated,
        "is_hedged": is_hedged,
        "study_type": study_type,
        "confidence_modifier": round(conf_modifier, 3),
    }


def compute_enhanced_confidence(original_conf: float, modifier: float) -> float:
    """Adjust original confidence with the modifier."""
    return round(min(max(original_conf * modifier, 0.0), 1.0), 2)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Enhanced Relation Classifier")
    print("=" * 65)

    # Load original associations
    print(f"\n[Step 1] Loading original associations...")
    rows = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows")

    # We need abstracts — load from spice results
    print(f"\n[Step 2] Loading abstracts from spice results...")
    abstracts = {}
    spice_csv = BASE_DIR / "pubmed_spice_results_clean.csv"
    with open(spice_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row["pubmed_id"]
            if pmid not in abstracts:
                abstracts[pmid] = row.get("abstract", "")
    print(f"  Loaded abstracts for {len(abstracts):,} PMIDs")

    # Classify
    print(f"\n[Step 3] Running enhanced classification...")
    changes = {"upgraded": 0, "downgraded": 0, "negated": 0, "unchanged": 0}
    enhanced_rows = []
    total = len(rows)

    for i, row in enumerate(rows):
        if (i + 1) % 25000 == 0:
            print(f"    Processing {i+1:,}/{total:,}...", end="\r", flush=True)

        pmid = row["pubmed_id"]
        abstract = abstracts.get(pmid, "")
        title = row.get("title", "")
        spice = row["spice"]
        disease = row["disease"]

        result = classify_enhanced(abstract, title, spice, disease)

        # Map "negated_positive" to a display-friendly label
        enhanced_nature = result["association_nature_enhanced"]
        if enhanced_nature == "negated_positive":
            enhanced_nature = "neutral"  # Negated positive = no real positive effect

        original_nature = row["association_nature"]
        original_conf = float(row["confidence"])
        new_conf = compute_enhanced_confidence(original_conf, result["confidence_modifier"])

        # Track changes
        if enhanced_nature != original_nature:
            if original_nature == "positive" and enhanced_nature == "neutral":
                changes["downgraded"] += 1
            elif original_nature == "neutral" and enhanced_nature in ("positive", "negative"):
                changes["upgraded"] += 1
            else:
                changes["unchanged"] += 1  # lateral change
        else:
            changes["unchanged"] += 1

        if result["is_negated"]:
            changes["negated"] += 1

        # Build enhanced row
        new_row = dict(row)
        new_row["association_nature_original"] = original_nature
        new_row["association_nature"] = enhanced_nature
        new_row["confidence_original"] = row["confidence"]
        new_row["confidence"] = new_conf
        new_row["is_negated"] = str(result["is_negated"])
        new_row["is_hedged"] = str(result["is_hedged"])
        new_row["study_type"] = result["study_type"]
        enhanced_rows.append(new_row)

    print(f"\n  Classification complete")

    # Write output
    print(f"\n[Step 4] Writing enhanced output...")
    fieldnames = [
        "pubmed_id", "spice", "disease", "mesh_id", "disease_category",
        "association_nature", "association_nature_original",
        "confidence", "confidence_original",
        "co_occurrence_level", "spice_location", "disease_location",
        "is_negated", "is_hedged", "study_type",
        "evidence_sentence", "publication_date", "title",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enhanced_rows)

    print(f"  Written {len(enhanced_rows):,} rows to {OUTPUT_CSV.name}")

    # Stats
    print(f"\n{'='*65}")
    print(f"  ENHANCED CLASSIFICATION RESULTS")
    print(f"{'='*65}")

    # Nature comparison
    orig_counts = Counter(r["association_nature_original"] for r in enhanced_rows)
    new_counts = Counter(r["association_nature"] for r in enhanced_rows)

    print(f"\n  Association nature comparison:")
    print(f"  {'Category':<15s} {'Original':>10s} {'Enhanced':>10s} {'Change':>10s}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for cat in ["positive", "negative", "neutral"]:
        orig = orig_counts.get(cat, 0)
        new = new_counts.get(cat, 0)
        diff = new - orig
        sign = "+" if diff > 0 else ""
        print(f"  {cat:<15s} {orig:>10,} {new:>10,} {sign}{diff:>9,}")

    # Change tracking
    print(f"\n  Changes made:")
    print(f"    Downgraded (positive -> neutral): {changes['downgraded']:>8,}")
    print(f"    Upgraded (neutral -> pos/neg):     {changes['upgraded']:>8,}")
    print(f"    Negated associations found:        {changes['negated']:>8,}")
    print(f"    Unchanged:                         {changes['unchanged']:>8,}")

    # Hedging stats
    hedged = sum(1 for r in enhanced_rows if r["is_hedged"] == "True")
    print(f"\n  Hedged (uncertain) associations:     {hedged:>8,}  "
          f"({hedged/len(enhanced_rows)*100:.1f}%)")

    # Study type distribution
    study_counts = Counter(r["study_type"] for r in enhanced_rows if r["study_type"])
    print(f"\n  Study type distribution:")
    for stype, count in study_counts.most_common():
        print(f"    {stype:<20s} {count:>8,}  ({count/len(enhanced_rows)*100:.1f}%)")

    no_study = sum(1 for r in enhanced_rows if not r["study_type"])
    print(f"    {'(unclassified)':<20s} {no_study:>8,}  ({no_study/len(enhanced_rows)*100:.1f}%)")

    # Confidence comparison
    orig_confs = [float(r["confidence_original"]) for r in enhanced_rows]
    new_confs = [float(r["confidence"]) for r in enhanced_rows]
    avg_orig = sum(orig_confs) / len(orig_confs)
    avg_new = sum(new_confs) / len(new_confs)

    print(f"\n  Confidence scores:")
    print(f"    Original avg: {avg_orig:.3f}")
    print(f"    Enhanced avg: {avg_new:.3f}")

    new_high = sum(1 for c in new_confs if c >= 0.7)
    new_med = sum(1 for c in new_confs if 0.4 <= c < 0.7)
    new_low = sum(1 for c in new_confs if c < 0.4)
    print(f"    High (>=0.7):   {new_high:>10,}")
    print(f"    Medium (0.4-0.7): {new_med:>8,}")
    print(f"    Low (<0.4):     {new_low:>10,}")

    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
