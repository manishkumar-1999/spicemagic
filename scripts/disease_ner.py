"""
Disease NER using PubTator3 API (NCBI)

Fetches pre-computed disease annotations (with MeSH IDs) for all PMIDs
from our clean spice results. PubTator3 uses TaggerOne — same tool as SpiceRx.

Output: disease_annotations.csv
Columns: pubmed_id, disease, mesh_id, disease_location, offset, length
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "pubmed_spice_results_clean.csv"
OUTPUT_CSV = BASE_DIR / "disease_annotations.csv"
CACHE_FILE = BASE_DIR / "disease_cache.json"

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

BATCH_SIZE = 100        # PMIDs per request (safe limit)
RATE_LIMIT_DELAY = 2.0  # ~30 requests/min
MAX_RETRIES = 3
SAVE_INTERVAL = 25      # Save cache every N batches


def http_get(url: str, params: dict, retries: int = MAX_RETRIES) -> bytes:
    """Make an HTTP GET request with retry logic."""
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "SpiceDisease/1.0 (research; python)")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def load_unique_pmids() -> list[str]:
    """Load unique PMIDs from the clean spice results CSV."""
    pmids = set()
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmids.add(row["pubmed_id"])
    return sorted(pmids)


def parse_disease_annotations(data: bytes) -> dict:
    """Parse PubTator3 BioC JSON response and extract disease annotations.

    Returns {pmid: [list of disease annotation dicts]}.
    """
    try:
        result = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        return {}

    annotations_by_pmid = {}

    # PubTator3 returns a dict with "PubTator3" key containing a list of docs
    if isinstance(result, list):
        documents = result
    elif isinstance(result, dict):
        documents = result.get("PubTator3", [])
    else:
        return {}

    for doc in documents:
        # Extract PMID — prefer direct "pmid" field, fallback to _id
        pmid = str(doc.get("pmid", "")) if doc.get("pmid") else ""
        if not pmid and "_id" in doc:
            pmid = str(doc["_id"]).split("|")[0]
        if not pmid:
            continue

        diseases = []
        for passage in doc.get("passages", []):
            passage_type = passage.get("infons", {}).get("type", "")  # "title" or "abstract"

            for ann in passage.get("annotations", []):
                ann_type = ann.get("infons", {}).get("type", "")
                if ann_type != "Disease":
                    continue

                identifier = ann.get("infons", {}).get("identifier", "")
                disease_text = ann.get("text", "")
                locations = ann.get("locations", [{}])
                offset = locations[0].get("offset", 0) if locations else 0
                length = locations[0].get("length", 0) if locations else 0

                diseases.append({
                    "disease": disease_text,
                    "mesh_id": identifier,
                    "disease_location": passage_type,
                    "offset": offset,
                    "length": length,
                })

        annotations_by_pmid[pmid] = diseases

    return annotations_by_pmid


def fetch_disease_annotations(pmids: list[str]) -> dict:
    """Fetch disease annotations for all PMIDs from PubTator3 API.

    Returns {pmid: [list of disease dicts]} with resume support via cache.
    """
    # Load cache
    all_annotations = {}
    if CACHE_FILE.exists():
        print(f"  Loading cached annotations from {CACHE_FILE.name}...")
        with open(CACHE_FILE, encoding="utf-8") as f:
            all_annotations = json.load(f)
        print(f"  Loaded {len(all_annotations)} cached PMIDs")

    # Filter out already-fetched PMIDs
    remaining = [p for p in pmids if p not in all_annotations]
    total = len(pmids)
    print(f"  {len(remaining)} PMIDs remaining to fetch ({total - len(remaining)} already cached)")

    if not remaining:
        return all_annotations

    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
    fetched_count = 0
    empty_count = 0

    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"  Batch {batch_num:4d}/{total_batches} "
              f"({len(all_annotations):6d}/{total} total, "
              f"{fetched_count} diseases found)", end="\r", flush=True)

        try:
            params = {"pmids": ",".join(batch)}
            data = http_get(PUBTATOR_URL, params)
            batch_annotations = parse_disease_annotations(data)

            for pmid in batch:
                if pmid in batch_annotations:
                    all_annotations[pmid] = batch_annotations[pmid]
                    fetched_count += len(batch_annotations[pmid])
                else:
                    # PMID not found in PubTator (no annotations or not indexed)
                    all_annotations[pmid] = []
                    empty_count += 1

        except Exception as e:
            print(f"\n  ERROR in batch {batch_num}: {e}")
            # Mark failed PMIDs so we don't retry them endlessly
            # (they can be retried by deleting from cache)

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Periodic save
        if batch_num % SAVE_INTERVAL == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(all_annotations, f)

    # Final save
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f)

    print(f"\n  Fetched annotations for {len(all_annotations)} PMIDs "
          f"({fetched_count} disease mentions, {empty_count} PMIDs with no diseases)")
    return all_annotations


def write_disease_csv(annotations: dict) -> list[dict]:
    """Write disease annotations to CSV. Returns list of rows written."""
    rows = []
    for pmid, diseases in annotations.items():
        # Deduplicate same disease in same location for same PMID
        seen = set()
        for d in diseases:
            key = (pmid, d["disease"], d["mesh_id"], d["disease_location"])
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "pubmed_id": pmid,
                "disease": d["disease"],
                "mesh_id": d["mesh_id"],
                "disease_location": d["disease_location"],
                "offset": d["offset"],
                "length": d["length"],
            })

    fieldnames = ["pubmed_id", "disease", "mesh_id", "disease_location", "offset", "length"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_stats(rows: list[dict], total_pmids: int) -> None:
    """Print summary statistics."""
    unique_pmids = {r["pubmed_id"] for r in rows}
    unique_diseases = {r["disease"] for r in rows}
    unique_mesh = {r["mesh_id"] for r in rows if r["mesh_id"]}

    # Location breakdown
    title_only = sum(1 for r in rows if r["disease_location"] == "title")
    abstract_only = sum(1 for r in rows if r["disease_location"] == "abstract")

    print(f"\n{'='*60}")
    print(f"DISEASE NER SUMMARY")
    print(f"{'='*60}")
    print(f"  Input PMIDs:              {total_pmids:,}")
    print(f"  PMIDs with diseases:      {len(unique_pmids):,}")
    print(f"  Total disease mentions:   {len(rows):,}")
    print(f"  Unique disease names:     {len(unique_diseases):,}")
    print(f"  Unique MeSH IDs:          {len(unique_mesh):,}")
    print(f"")
    print(f"  Location breakdown:")
    print(f"    Title:                  {title_only:,}")
    print(f"    Abstract:               {abstract_only:,}")
    print(f"")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*60}")

    # Top 30 diseases
    disease_counts = Counter(r["disease"] for r in rows)
    print(f"\n  Top 30 diseases by mention count:")
    for disease, count in disease_counts.most_common(30):
        mesh = ""
        for r in rows:
            if r["disease"] == disease and r["mesh_id"]:
                mesh = r["mesh_id"]
                break
        print(f"    {disease:<40s} {count:6,}  ({mesh})")


def main():
    print("=" * 60)
    print("Disease NER via PubTator3 API")
    print("=" * 60)

    # Step 1: Load unique PMIDs
    print(f"\n[Step 1] Loading unique PMIDs from {INPUT_CSV.name}...")
    pmids = load_unique_pmids()
    print(f"  Found {len(pmids):,} unique PMIDs")

    # Step 2: Fetch disease annotations
    print(f"\n[Step 2] Fetching disease annotations from PubTator3...")
    print(f"  Batch size: {BATCH_SIZE}, Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print(f"  Estimated time: ~{len(pmids) / BATCH_SIZE * RATE_LIMIT_DELAY / 60:.0f} minutes")
    annotations = fetch_disease_annotations(pmids)

    # Step 3: Write CSV
    print(f"\n[Step 3] Writing disease annotations to CSV...")
    rows = write_disease_csv(annotations)

    # Step 4: Print stats
    print_stats(rows, len(pmids))


if __name__ == "__main__":
    main()
