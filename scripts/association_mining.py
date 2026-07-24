"""
Spice-Disease Association Mining

Combines spice NER results with disease NER annotations to produce
spice-disease co-occurrence pairs with association metrics.

Input:
  - pubmed_spice_results_clean.csv  (spice NER results)
  - disease_annotations.csv         (PubTator3 disease annotations)

Output:
  - spice_disease_associations.csv

Columns: pubmed_id, spice, disease, mesh_id, disease_category,
         association_nature, confidence, co_occurrence_level,
         spice_location, disease_location, evidence_sentence,
         publication_date, title
"""

import csv
import re
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent
SPICE_CSV = BASE_DIR / "pubmed_spice_results_clean.csv"
DISEASE_CSV = BASE_DIR / "disease_annotations.csv"
OUTPUT_CSV = BASE_DIR / "spice_disease_associations.csv"
MESH_CACHE = BASE_DIR / "mesh_category_cache.json"

# ── MeSH disease category tree-number prefixes ──────────────────────
MESH_CATEGORIES = {
    "C01": "Infections",
    "C04": "Neoplasms",
    "C05": "Musculoskeletal Diseases",
    "C06": "Digestive System Diseases",
    "C07": "Stomatognathic Diseases",
    "C08": "Respiratory Tract Diseases",
    "C09": "Otorhinolaryngologic Diseases",
    "C10": "Nervous System Diseases",
    "C11": "Eye Diseases",
    "C12": "Urogenital Diseases",
    "C13": "Female Urogenital Diseases",
    "C14": "Cardiovascular Diseases",
    "C15": "Hemic and Lymphatic Diseases",
    "C16": "Congenital Abnormalities",
    "C17": "Skin and Connective Tissue Diseases",
    "C18": "Nutritional and Metabolic Diseases",
    "C19": "Endocrine System Diseases",
    "C20": "Immune System Diseases",
    "C23": "Pathological Conditions, Signs and Symptoms",
    "C24": "Occupational Diseases",
    "C25": "Chemically-Induced Disorders",
    "C26": "Wounds and Injuries",
    "F03": "Mental Disorders",
}

# ── Keywords for association nature classification ───────────────────
POSITIVE_KEYWORDS = re.compile(
    r'\b(?:'
    r'treat(?:s|ed|ing|ment)?|'
    r'prevent(?:s|ed|ing|ion|ive)?|prophylaxis|'
    r'inhibit(?:s|ed|ing|ion|ory)?|'
    r'reduc(?:e[sd]?|ing|tion)|'
    r'alleviat\w*|ameliorat\w*|attenuat\w*|mitigat\w*|suppress\w*|'
    r'protect\w*|'
    r'anti[\-]?(?:inflammatory|oxidant|cancer|tumor|tumour|microbial|'
    r'bacterial|viral|fungal|diabetic|hypertensive|nociceptive)|'
    r'therapeutic|beneficial|efficac\w*|effective|'
    r'heal(?:s|ed|ing)?|'
    r'improv(?:e[sd]?|ing|ement)|'
    r'decreas(?:e[sd]?|ing)|'
    r'lower(?:s|ed|ing)?|'
    r'reliev\w*|revers\w*|restor\w*|recover\w*'
    r')\b',
    re.IGNORECASE,
)

NEGATIVE_KEYWORDS = re.compile(
    r'\b(?:'
    r'caus(?:e[sd]?|ing)|'
    r'induc(?:e[sd]?|ing|tion)|'
    r'worsen(?:s|ed|ing)?|'
    r'exacerbat\w*|aggravat\w*|'
    r'toxic(?:ity)?|poisoning|'
    r'adverse|side\s+effect|'
    r'allerg(?:ic|y)|hypersensitiv\w*|contraindic\w*|'
    r'increas(?:e[sd]?)?\s+risk'
    r')\b',
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_spice_results() -> dict:
    """Load spice NER results grouped by PMID.

    Returns {pmid: [list of row dicts]}.
    """
    by_pmid = defaultdict(list)
    with open(SPICE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_pmid[row["pubmed_id"]].append(row)
    return dict(by_pmid)


def load_disease_annotations() -> dict:
    """Load disease annotations grouped by PMID.

    Returns {pmid: [list of row dicts]}.
    Deduplicates (disease, mesh_id) per PMID, keeping first occurrence.
    """
    by_pmid = defaultdict(list)
    seen = defaultdict(set)
    with open(DISEASE_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row["pubmed_id"]
            key = (row["disease"], row["mesh_id"])
            if key not in seen[pmid]:
                seen[pmid].add(key)
                by_pmid[pmid].append(row)
    return dict(by_pmid)


# ═══════════════════════════════════════════════════════════════════
# MeSH CATEGORY LOOKUP
# ═══════════════════════════════════════════════════════════════════

def fetch_mesh_categories(mesh_ids: set) -> dict:
    """Fetch MeSH tree numbers to categorize diseases.

    Two-step approach:
      1. esearch to convert descriptor IDs (D######) to numeric UIDs
      2. esummary to get tree numbers from ds_idxlinks

    Returns {descriptor_id: category_name} mapping.
    """
    # Load cache
    cache = {}
    if MESH_CACHE.exists():
        with open(MESH_CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    # Clean and filter IDs
    to_fetch = set()
    for mid in mesh_ids:
        if not mid:
            continue
        clean = mid.replace("MESH:", "").replace("mesh:", "").strip()
        if clean and clean.startswith("D") and clean not in cache:
            to_fetch.add(clean)

    if not to_fetch:
        print(f"  All {len(cache)} MeSH categories already cached")
        return cache

    to_fetch = sorted(to_fetch)
    print(f"  Fetching MeSH categories for {len(to_fetch)} unique descriptor IDs...")

    # Step A: Convert descriptor IDs to numeric UIDs via esearch (batch of 50)
    SEARCH_BATCH = 50
    desc_to_uid = {}
    search_batches = (len(to_fetch) + SEARCH_BATCH - 1) // SEARCH_BATCH

    print(f"    Phase 1: Converting descriptor IDs to UIDs ({search_batches} batches)...")

    for i in range(0, len(to_fetch), SEARCH_BATCH):
        batch = to_fetch[i:i + SEARCH_BATCH]
        batch_num = i // SEARCH_BATCH + 1
        print(f"      esearch batch {batch_num:3d}/{search_batches}", end="\r", flush=True)

        try:
            term = " OR ".join(f"{d}[MHUI]" for d in batch)
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {"db": "mesh", "term": term, "retmode": "json", "retmax": len(batch) + 10}
            query = urllib.parse.urlencode(params)
            req = urllib.request.Request(f"{url}?{query}")
            req.add_header("User-Agent", "SpiceDisease/1.0 (research; python)")

            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())

            uids = data.get("esearchresult", {}).get("idlist", [])
            # We'll resolve desc_id <-> uid in the esummary step
            for uid in uids:
                desc_to_uid[uid] = None  # Placeholder, resolved in step B

        except Exception as e:
            print(f"\n      esearch batch {batch_num} error: {e}")

        time.sleep(0.35)

    all_uids = sorted(desc_to_uid.keys())
    print(f"\n    Found {len(all_uids)} numeric UIDs")

    # Step B: Fetch summaries to get tree numbers and descriptor IDs
    SUMMARY_BATCH = 200
    summary_batches = (len(all_uids) + SUMMARY_BATCH - 1) // SUMMARY_BATCH
    fetched = 0

    print(f"    Phase 2: Fetching tree numbers ({summary_batches} batches)...")

    for i in range(0, len(all_uids), SUMMARY_BATCH):
        batch = all_uids[i:i + SUMMARY_BATCH]
        batch_num = i // SUMMARY_BATCH + 1
        print(f"      esummary batch {batch_num:3d}/{summary_batches} "
              f"({fetched} categorized)", end="\r", flush=True)

        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            data = urllib.parse.urlencode({
                "db": "mesh",
                "id": ",".join(batch),
                "retmode": "json",
            }).encode("utf-8")

            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("User-Agent", "SpiceDisease/1.0 (research; python)")

            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read())

            records = result.get("result", {})
            for uid in batch:
                rec = records.get(uid, {})
                mesh_ui = rec.get("ds_meshui", "")
                if not mesh_ui:
                    continue

                # Extract tree numbers from ds_idxlinks
                idx_links = rec.get("ds_idxlinks", [])
                category = None
                for link in idx_links:
                    tree = link.get("treenum", "")
                    prefix = tree[:3] if tree else ""
                    if prefix in MESH_CATEGORIES:
                        category = MESH_CATEGORIES[prefix]
                        break

                cache[mesh_ui] = category or "Other"
                fetched += 1

        except Exception as e:
            print(f"\n      esummary batch {batch_num} error: {e}")

        time.sleep(0.35)

        # Periodic save
        if batch_num % 10 == 0:
            with open(MESH_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f)

    # Final save
    with open(MESH_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"\n  Categorized {fetched} new MeSH IDs ({len(cache)} total cached)")
    return cache


def get_disease_category(mesh_id: str, mesh_categories: dict) -> str:
    """Look up disease category for a MeSH ID."""
    if not mesh_id:
        return ""
    clean = mesh_id.replace("MESH:", "").replace("mesh:", "").strip()
    return mesh_categories.get(clean, "")


# ═══════════════════════════════════════════════════════════════════
# ASSOCIATION CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    if not text:
        return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def find_evidence_sentence(abstract: str, spice: str, disease: str) -> str:
    """Find the sentence mentioning both spice and disease."""
    if not abstract:
        return ""

    spice_lower = spice.lower()
    disease_lower = disease.lower()

    for sent in split_sentences(abstract):
        sent_lower = sent.lower()
        if spice_lower in sent_lower and disease_lower in sent_lower:
            return sent[:500]

    return ""


def classify_association(abstract: str, spice: str, disease: str) -> str:
    """Classify association as positive, negative, or neutral.

    Looks at sentences containing both the spice and disease mentions.
    Falls back to full abstract if no co-occurring sentence found.
    """
    if not abstract:
        return "neutral"

    # Prefer sentences with both mentions
    relevant = ""
    for sent in split_sentences(abstract):
        sent_lower = sent.lower()
        if spice.lower() in sent_lower and disease.lower() in sent_lower:
            relevant += " " + sent

    # Fallback to full abstract
    if not relevant:
        relevant = abstract

    pos = len(POSITIVE_KEYWORDS.findall(relevant))
    neg = len(NEGATIVE_KEYWORDS.findall(relevant))

    if pos > 0 and neg == 0:
        return "positive"
    if neg > 0 and pos == 0:
        return "negative"
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def compute_co_occurrence_level(spice_loc: str, disease_loc: str) -> str:
    """Where do spice and disease co-occur in the paper?

    spice_loc:   "title", "abstract", or "title;abstract"
    disease_loc: "title" or "abstract"
    """
    spice_in_title = "title" in spice_loc
    spice_in_abstract = "abstract" in spice_loc
    dis_in_title = disease_loc == "title"
    dis_in_abstract = disease_loc == "abstract"

    both_title = spice_in_title and dis_in_title
    both_abstract = spice_in_abstract and dis_in_abstract

    if both_title and both_abstract:
        return "title;abstract"
    if both_title:
        return "title"
    if both_abstract:
        return "abstract"
    return "cross-section"


def compute_confidence(co_occ_level: str, assoc_nature: str, pair_freq: int) -> float:
    """Compute confidence score (0.0–1.0) from multiple signals.

    Signals:
      - Location strength   (0–0.30)
      - Association clarity  (0–0.30)
      - Pair frequency       (0–0.40)
    """
    score = 0.0

    # Location signal
    loc_scores = {"title;abstract": 0.30, "title": 0.25, "abstract": 0.20, "cross-section": 0.10}
    score += loc_scores.get(co_occ_level, 0.10)

    # Association nature signal
    nature_scores = {"positive": 0.30, "negative": 0.25, "neutral": 0.10}
    score += nature_scores.get(assoc_nature, 0.10)

    # Frequency signal (how many papers report same spice-disease pair)
    if pair_freq >= 10:
        score += 0.40
    elif pair_freq >= 5:
        score += 0.30
    elif pair_freq >= 3:
        score += 0.20
    elif pair_freq >= 2:
        score += 0.15
    else:
        score += 0.05

    return round(min(score, 1.0), 2)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build_associations(spice_data: dict, disease_data: dict,
                       mesh_categories: dict) -> list[dict]:
    """Build spice-disease association rows by joining on PMID.

    For each PMID that has BOTH spice mentions and disease annotations,
    create one row per (spice, disease) pair.
    """
    # Find common PMIDs
    common_pmids = set(spice_data.keys()) & set(disease_data.keys())
    print(f"  PMIDs with both spice and disease data: {len(common_pmids):,}")

    # First pass: count (spice, disease) pair frequencies across all PMIDs
    pair_counter = Counter()
    for pmid in common_pmids:
        spices_in_paper = {row["spice"] for row in spice_data[pmid]}
        diseases_in_paper = {row["disease"] for row in disease_data[pmid]}
        for sp in spices_in_paper:
            for dis in diseases_in_paper:
                pair_counter[(sp, dis)] += 1

    print(f"  Unique (spice, disease) pairs: {len(pair_counter):,}")

    # Second pass: build full association rows
    rows = []
    processed = 0
    total = len(common_pmids)

    for pmid in sorted(common_pmids):
        processed += 1
        if processed % 5000 == 0:
            print(f"    Processing PMID {processed:,}/{total:,} "
                  f"({len(rows):,} rows so far)", end="\r", flush=True)

        spice_rows = spice_data[pmid]
        disease_rows = disease_data[pmid]

        # Get abstract and metadata from first spice row (same for all rows of same PMID)
        abstract = spice_rows[0].get("abstract", "")
        pub_date = spice_rows[0].get("publication_date", "")
        title = spice_rows[0].get("title", "")

        # Deduplicate spice entries (keep unique spice names with their locations)
        spice_map = {}
        for sr in spice_rows:
            sp = sr["spice"]
            if sp not in spice_map:
                spice_map[sp] = sr["location"]

        # Deduplicate disease entries
        disease_map = {}
        for dr in disease_rows:
            key = (dr["disease"], dr["mesh_id"])
            if key not in disease_map:
                disease_map[key] = dr["disease_location"]

        # Cross-join spices × diseases
        for spice, spice_loc in spice_map.items():
            for (disease, mesh_id), disease_loc in disease_map.items():
                co_occ = compute_co_occurrence_level(spice_loc, disease_loc)
                assoc = classify_association(abstract, spice, disease)
                pair_freq = pair_counter[(spice, disease)]
                conf = compute_confidence(co_occ, assoc, pair_freq)
                evidence = find_evidence_sentence(abstract, spice, disease)
                category = get_disease_category(mesh_id, mesh_categories)

                rows.append({
                    "pubmed_id": pmid,
                    "spice": spice,
                    "disease": disease,
                    "mesh_id": mesh_id,
                    "disease_category": category,
                    "association_nature": assoc,
                    "confidence": conf,
                    "co_occurrence_level": co_occ,
                    "spice_location": spice_loc,
                    "disease_location": disease_loc,
                    "evidence_sentence": evidence,
                    "publication_date": pub_date,
                    "title": title,
                })

    print(f"\n  Built {len(rows):,} association rows")
    return rows


def write_output(rows: list[dict]) -> None:
    """Write final association CSV."""
    fieldnames = [
        "pubmed_id", "spice", "disease", "mesh_id", "disease_category",
        "association_nature", "confidence", "co_occurrence_level",
        "spice_location", "disease_location", "evidence_sentence",
        "publication_date", "title",
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(rows: list[dict]) -> None:
    """Print comprehensive summary statistics."""
    if not rows:
        print("  No association rows generated!")
        return

    unique_pmids = {r["pubmed_id"] for r in rows}
    unique_spices = {r["spice"] for r in rows}
    unique_diseases = {r["disease"] for r in rows}
    unique_mesh = {r["mesh_id"] for r in rows if r["mesh_id"]}
    unique_pairs = {(r["spice"], r["disease"]) for r in rows}
    unique_categories = {r["disease_category"] for r in rows if r["disease_category"]}

    # Association nature breakdown
    nature_counts = Counter(r["association_nature"] for r in rows)

    # Co-occurrence level breakdown
    co_occ_counts = Counter(r["co_occurrence_level"] for r in rows)

    # Confidence distribution
    conf_high = sum(1 for r in rows if r["confidence"] >= 0.7)
    conf_med = sum(1 for r in rows if 0.4 <= r["confidence"] < 0.7)
    conf_low = sum(1 for r in rows if r["confidence"] < 0.4)

    # Evidence sentence coverage
    with_evidence = sum(1 for r in rows if r["evidence_sentence"])

    print(f"\n{'='*65}")
    print(f"  SPICE-DISEASE ASSOCIATION MINING — RESULTS")
    print(f"{'='*65}")
    print(f"  Total association rows:     {len(rows):>10,}")
    print(f"  Unique PMIDs:               {len(unique_pmids):>10,}")
    print(f"  Unique spices:              {len(unique_spices):>10,}")
    print(f"  Unique diseases:            {len(unique_diseases):>10,}")
    print(f"  Unique MeSH IDs:            {len(unique_mesh):>10,}")
    print(f"  Unique (spice, disease):    {len(unique_pairs):>10,}")
    print(f"  Disease categories found:   {len(unique_categories):>10,}")
    print()
    print(f"  Association nature:")
    for nature in ["positive", "negative", "neutral"]:
        c = nature_counts.get(nature, 0)
        pct = c / len(rows) * 100
        print(f"    {nature:<12s} {c:>10,}  ({pct:5.1f}%)")
    print()
    print(f"  Co-occurrence level:")
    for level in ["title;abstract", "title", "abstract", "cross-section"]:
        c = co_occ_counts.get(level, 0)
        pct = c / len(rows) * 100
        print(f"    {level:<18s} {c:>10,}  ({pct:5.1f}%)")
    print()
    print(f"  Confidence distribution:")
    print(f"    High (>=0.7):   {conf_high:>10,}  ({conf_high/len(rows)*100:5.1f}%)")
    print(f"    Medium (0.4-0.7): {conf_med:>8,}  ({conf_med/len(rows)*100:5.1f}%)")
    print(f"    Low (<0.4):     {conf_low:>10,}  ({conf_low/len(rows)*100:5.1f}%)")
    print()
    print(f"  Evidence sentences found:   {with_evidence:>10,}  "
          f"({with_evidence/len(rows)*100:5.1f}%)")
    print()

    # Top 20 spice-disease pairs by frequency
    pair_counts = Counter((r["spice"], r["disease"]) for r in rows)
    print(f"  Top 20 spice-disease pairs by PMID frequency:")
    print(f"  {'Spice':<25s} {'Disease':<30s} {'PMIDs':>6s}")
    print(f"  {'-'*25} {'-'*30} {'-'*6}")
    for (spice, disease), count in pair_counts.most_common(20):
        print(f"  {spice:<25s} {disease:<30s} {count:>6,}")

    # Top 15 spices by association count
    print(f"\n  Top 15 spices by total associations:")
    spice_counts = Counter(r["spice"] for r in rows)
    for spice, count in spice_counts.most_common(15):
        print(f"    {spice:<25s} {count:>8,}")

    # Top 15 diseases by association count
    print(f"\n  Top 15 diseases by total associations:")
    disease_counts = Counter(r["disease"] for r in rows)
    for disease, count in disease_counts.most_common(15):
        print(f"    {disease:<35s} {count:>8,}")

    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"{'='*65}")


def main():
    print("=" * 65)
    print("  Spice-Disease Association Mining")
    print("=" * 65)

    # Step 1: Load spice results
    print(f"\n[Step 1] Loading spice NER results...")
    spice_data = load_spice_results()
    print(f"  Loaded {len(spice_data):,} PMIDs with spice mentions")

    # Step 2: Load disease annotations
    print(f"\n[Step 2] Loading disease annotations...")
    disease_data = load_disease_annotations()
    print(f"  Loaded {len(disease_data):,} PMIDs with disease annotations")

    # Step 3: Fetch MeSH categories
    print(f"\n[Step 3] Fetching MeSH disease categories...")
    all_mesh_ids = set()
    for diseases in disease_data.values():
        for d in diseases:
            if d["mesh_id"]:
                all_mesh_ids.add(d["mesh_id"])
    print(f"  Found {len(all_mesh_ids):,} unique MeSH IDs to categorize")
    mesh_categories = fetch_mesh_categories(all_mesh_ids)

    # Step 4: Build associations
    print(f"\n[Step 4] Building spice-disease associations...")
    rows = build_associations(spice_data, disease_data, mesh_categories)

    # Step 5: Write output
    print(f"\n[Step 5] Writing output CSV...")
    write_output(rows)
    print(f"  Written {len(rows):,} rows to {OUTPUT_CSV.name}")

    # Step 6: Print stats
    print_stats(rows)


if __name__ == "__main__":
    main()
