"""
PubMed Spice Mining Script
Searches PubMed for research papers mentioning culinary spices/herbs,
fetches abstracts, runs NER, and outputs results to CSV.

Usage:
    python pubmed_search.py
"""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Import NER functions from main.py
from main import load_spice_dictionary, find_spices_in_text

# --- Configuration ---
BASE_DIR = Path(__file__).parent
DICT_PATH = BASE_DIR / "spice_ner_dictionary.csv"
PMIDS_CACHE = BASE_DIR / "pmids_cache.json"
ARTICLES_CACHE = BASE_DIR / "articles_cache.json"
OUTPUT_CSV = BASE_DIR / "pubmed_spice_results.csv"

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

MAX_PMIDS_PER_SPICE = 2000  # Cap per spice to keep total manageable
EFETCH_BATCH_SIZE = 500     # PMIDs per efetch request
RATE_LIMIT_DELAY = 0.35     # seconds between API calls (3 req/sec limit)
MAX_RETRIES = 3


def http_get(url: str, params: dict, retries: int = MAX_RETRIES) -> bytes:
    """Make an HTTP GET request with retry logic."""
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "SpiceDisease/1.0 (research; python)")
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def http_post(url: str, params: dict, retries: int = MAX_RETRIES) -> bytes:
    """Make an HTTP POST request with retry logic (for large payloads)."""
    post_data = urllib.parse.urlencode(params).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=post_data, method="POST")
            req.add_header("User-Agent", "SpiceDisease/1.0 (research; python)")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt + 1}/{retries} after error: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise


def build_search_query(spice: dict) -> str:
    """Build a PubMed search query for a spice using its names and synonyms."""
    terms = []

    # Main name
    terms.append(f'"{spice["main_name"]}"')

    # Scientific name (most specific, always include)
    if spice["scientific_name"]:
        terms.append(f'"{spice["scientific_name"]}"')

    # Add synonyms (skip very short or ambiguous ones)
    for syn in spice["synonyms"]:
        if len(syn) >= 4:  # Skip very short synonyms to avoid noise
            terms.append(f'"{syn}"')

    query = " OR ".join(terms)
    return query


def search_pmids_for_spice(spice: dict, max_results: int = MAX_PMIDS_PER_SPICE) -> list[str]:
    """Search PubMed for a spice and return list of PMIDs."""
    query = build_search_query(spice)

    # Fetch PMIDs directly in batches of up to 10000
    pmids = []
    fetch_count = max_results
    retmax_per_call = min(10000, fetch_count)

    for start in range(0, fetch_count, retmax_per_call):
        batch_size = min(retmax_per_call, fetch_count - start)
        params = {
            "db": "pubmed",
            "term": query,
            "retstart": start,
            "retmax": batch_size,
        }
        data = http_get(ESEARCH_URL, params)
        time.sleep(RATE_LIMIT_DELAY)

        root = ET.fromstring(data)

        # On first call, check total count to avoid unnecessary requests
        if start == 0:
            count_el = root.find("Count")
            if count_el is None or int(count_el.text) == 0:
                return []
            total_available = int(count_el.text)
            fetch_count = min(total_available, max_results)

        id_list = root.find("IdList")
        if id_list is not None:
            for id_el in id_list.findall("Id"):
                pmids.append(id_el.text)

        # Stop if we got fewer than requested (no more results)
        if id_list is None or len(id_list.findall("Id")) < batch_size:
            break

    return pmids[:max_results]


def collect_all_pmids(spice_dict: list[dict]) -> dict:
    """Search PubMed for all spices. Returns {spice_name: [pmids]}."""
    # Check for cached results
    if PMIDS_CACHE.exists():
        print(f"Loading cached PMIDs from {PMIDS_CACHE.name}...")
        with open(PMIDS_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  Loaded {len(cached)} spices from cache")
        return cached

    spice_pmids = {}
    total = len(spice_dict)

    for i, spice in enumerate(spice_dict):
        name = spice["main_name"]
        print(f"  [{i+1:3d}/{total}] Searching: {name:<25s}", end="", flush=True)

        try:
            pmids = search_pmids_for_spice(spice)
            spice_pmids[name] = pmids
            print(f" -> {len(pmids):5d} PMIDs")
        except Exception as e:
            print(f" -> ERROR: {e}")
            spice_pmids[name] = []

    # Save cache
    with open(PMIDS_CACHE, "w", encoding="utf-8") as f:
        json.dump(spice_pmids, f)
    print(f"  Saved PMID cache to {PMIDS_CACHE.name}")

    return spice_pmids


def fetch_abstracts_batch(pmids: list[str]) -> dict:
    """Fetch article data for a batch of PMIDs. Returns {pmid: {title, abstract, date}}."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    data = http_post(EFETCH_URL, params)
    time.sleep(RATE_LIMIT_DELAY)

    articles = {}
    # Sanitize XML: remove control characters that cause parse errors
    text = data.decode("utf-8", errors="replace")
    text = "".join(ch if ch == "\n" or ch == "\r" or ch == "\t" or ord(ch) >= 32 else " " for ch in text)
    root = ET.fromstring(text)

    for article_el in root.findall(".//PubmedArticle"):
        # PMID
        pmid_el = article_el.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text

        # Title
        title_el = article_el.find(".//ArticleTitle")
        title = ""
        if title_el is not None:
            title = "".join(title_el.itertext()).strip()

        # Abstract - may have multiple AbstractText elements (structured abstract)
        abstract_parts = []
        for abs_el in article_el.findall(".//AbstractText"):
            label = abs_el.get("Label", "")
            text = "".join(abs_el.itertext()).strip()
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Publication date - try multiple locations
        pub_date = ""
        # Try PubDate first
        date_el = article_el.find(".//PubDate")
        if date_el is not None:
            year = date_el.findtext("Year", "")
            month = date_el.findtext("Month", "")
            day = date_el.findtext("Day", "")
            medline_date = date_el.findtext("MedlineDate", "")

            if year:
                pub_date = year
                if month:
                    pub_date += f"-{month}"
                    if day:
                        pub_date += f"-{day}"
            elif medline_date:
                pub_date = medline_date

        articles[pmid] = {
            "title": title,
            "abstract": abstract,
            "publication_date": pub_date,
        }

    return articles


def fetch_all_abstracts(all_pmids: list[str]) -> dict:
    """Fetch abstracts for all PMIDs in batches with resume support."""
    # Load cache if exists (for resume)
    all_articles = {}
    if ARTICLES_CACHE.exists():
        print(f"  Loading cached articles from {ARTICLES_CACHE.name}...")
        with open(ARTICLES_CACHE, encoding="utf-8") as f:
            all_articles = json.load(f)
        print(f"  Loaded {len(all_articles)} cached articles")

    # Filter out already-fetched PMIDs
    remaining = [p for p in all_pmids if p not in all_articles]
    total = len(all_pmids)
    print(f"  {len(remaining)} PMIDs remaining to fetch ({total - len(remaining)} already cached)")

    if not remaining:
        return all_articles

    total_batches = (len(remaining) + EFETCH_BATCH_SIZE - 1) // EFETCH_BATCH_SIZE
    save_interval = 50  # Save cache every 50 batches

    for i in range(0, len(remaining), EFETCH_BATCH_SIZE):
        batch = remaining[i:i + EFETCH_BATCH_SIZE]
        batch_num = i // EFETCH_BATCH_SIZE + 1
        print(f"  Batch {batch_num:4d}/{total_batches} ({len(all_articles):6d}/{total} total)", end="\r", flush=True)

        try:
            articles = fetch_abstracts_batch(batch)
            all_articles.update(articles)
        except Exception as e:
            print(f"\n  ERROR in batch {batch_num}: {e}")

        # Periodic save
        if batch_num % save_interval == 0:
            with open(ARTICLES_CACHE, "w", encoding="utf-8") as f:
                json.dump(all_articles, f)

    # Final save
    with open(ARTICLES_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_articles, f)

    print(f"  Fetched {len(all_articles)} articles from {total} PMIDs" + " " * 30)
    return all_articles


def run_ner_and_write_csv(
    spice_pmids: dict,
    articles: dict,
    spice_dict: list[dict],
) -> None:
    """Run NER on all fetched articles and write results to CSV."""
    # Build a reverse map: pmid -> set of spice names that found it via search
    pmid_to_search_spices = {}
    for spice_name, pmids in spice_pmids.items():
        for pmid in pmids:
            if pmid not in pmid_to_search_spices:
                pmid_to_search_spices[pmid] = set()
            pmid_to_search_spices[pmid].add(spice_name)

    rows = []
    total = len(articles)
    processed = 0

    for pmid, article in articles.items():
        processed += 1
        if processed % 1000 == 0:
            print(f"  NER processing: {processed:6d}/{total}", end="\r", flush=True)

        title = article["title"]
        abstract = article["abstract"]

        # Run NER on title and abstract separately
        title_matches = find_spices_in_text(title, spice_dict) if title else []
        abstract_matches = find_spices_in_text(abstract, spice_dict) if abstract else []

        # Collect unique spice names found in each
        title_spices = {m["main_name"] for m in title_matches}
        abstract_spices = {m["main_name"] for m in abstract_matches}
        all_found_spices = title_spices | abstract_spices

        # Create one row per spice found
        for spice_name in all_found_spices:
            in_title = spice_name in title_spices
            in_abstract = spice_name in abstract_spices

            if in_title and in_abstract:
                location = "title;abstract"
            elif in_title:
                location = "title"
            else:
                location = "abstract"

            rows.append({
                "pubmed_id": pmid,
                "spice": spice_name,
                "location": location,
                "publication_date": article["publication_date"],
                "title": title,
                "abstract": abstract,
            })

    print(f"  NER processing: {processed:6d}/{total} done" + " " * 20)

    # Write CSV
    fieldnames = ["pubmed_id", "spice", "location", "publication_date", "title", "abstract"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_stats(rows: list[dict], spice_pmids: dict) -> None:
    """Print summary statistics."""
    unique_pmids = {r["pubmed_id"] for r in rows}
    unique_spices = {r["spice"] for r in rows}
    spices_with_results = sum(1 for pmids in spice_pmids.values() if pmids)

    # Location breakdown
    title_only = sum(1 for r in rows if r["location"] == "title")
    abstract_only = sum(1 for r in rows if r["location"] == "abstract")
    both = sum(1 for r in rows if r["location"] == "title;abstract")

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows in CSV:           {len(rows):,}")
    print(f"  Unique PubMed articles:      {len(unique_pmids):,}")
    print(f"  Total spices in dictionary:  {len(spice_pmids)}")
    print(f"  Spices with PubMed results:  {spices_with_results}")
    print(f"  Unique spices found by NER:  {len(unique_spices)}")
    print(f"")
    print(f"  Location breakdown:")
    print(f"    Title only:                {title_only:,}")
    print(f"    Abstract only:             {abstract_only:,}")
    print(f"    Title + Abstract:          {both:,}")
    print(f"")
    print(f"  Output file: {OUTPUT_CSV}")
    print(f"{'='*60}")

    # Top 20 spices by row count
    from collections import Counter
    spice_counts = Counter(r["spice"] for r in rows)
    print(f"\n  Top 20 spices by number of articles:")
    for spice, count in spice_counts.most_common(20):
        print(f"    {spice:<25s} {count:6,}")


def main():
    print("=" * 60)
    print("PubMed Spice Mining Pipeline")
    print("=" * 60)

    # Step 1: Load spice dictionary
    print(f"\n[Step 1] Loading spice NER dictionary...")
    spice_dict = load_spice_dictionary(str(DICT_PATH))
    print(f"  Loaded {len(spice_dict)} spices with {sum(len(s['all_names']) for s in spice_dict)} name variants")

    # Step 2: Search PubMed for each spice
    print(f"\n[Step 2] Searching PubMed for spice mentions...")
    spice_pmids = collect_all_pmids(spice_dict)

    # Step 3: Deduplicate and collect all unique PMIDs
    print(f"\n[Step 3] Deduplicating PMIDs...")
    all_pmids_set = set()
    for pmids in spice_pmids.values():
        all_pmids_set.update(pmids)
    all_pmids = sorted(all_pmids_set)
    print(f"  Total unique PMIDs: {len(all_pmids):,}")

    # Step 4: Fetch abstracts
    print(f"\n[Step 4] Fetching abstracts from PubMed...")
    articles = fetch_all_abstracts(all_pmids)

    # Step 5: Run NER and write CSV
    print(f"\n[Step 5] Running spice NER on fetched articles...")
    rows = run_ner_and_write_csv(spice_pmids, articles, spice_dict)

    # Step 6: Print stats
    print_stats(rows, spice_pmids)


if __name__ == "__main__":
    main()
