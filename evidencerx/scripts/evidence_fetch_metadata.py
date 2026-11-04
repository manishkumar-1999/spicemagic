"""
EvidenceRx — Phase E1: PubMed metadata fetcher.

Fetches PublicationType tags and MeSH descriptors for every unique PubMed ID
in the spice-disease dataset. These two fields are the free weak-supervision
signal for study-type classification:

  - PublicationType  -> RCT, Meta-Analysis, Systematic Review, Observational, ...
  - MeSH descriptors -> in-vitro vs animal split (Animals, Mice, In Vitro
                        Techniques, Cells Cultured, Humans, ...)

Output: pubmed_metadata_cache.json  (resumable, keyed by PMID)
    { "12345": {"pub_types": [...], "mesh": [...]}, ... }

Usage:
    python evidence_fetch_metadata.py            # full run (resumes from cache)
    python evidence_fetch_metadata.py --sample 20   # test on 20 PMIDs only
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
ASSOC_CSV = BASE_DIR / "spice_disease_associations.csv"
META_CACHE = BASE_DIR / "pubmed_metadata_cache.json"

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI allows 3 req/sec without a key, 10 req/sec with one.
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "").strip()
BATCH_SIZE = 300
RATE_LIMIT_DELAY = 0.11 if NCBI_API_KEY else 0.35
MAX_RETRIES = 4
SAVE_EVERY = 20  # flush cache to disk every N batches


def http_post(url: str, params: dict, retries: int = MAX_RETRIES) -> bytes:
    """POST with exponential-backoff retry (id lists can be long)."""
    post_data = urllib.parse.urlencode(params).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=post_data, method="POST")
            req.add_header("User-Agent", "SpiceDisease-EvidenceRx/1.0 (research; python)")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    retry {attempt + 1}/{retries} after {e} (wait {wait}s)")
                time.sleep(wait)
            else:
                raise


def parse_efetch_xml(xml_bytes: bytes) -> dict:
    """Extract {pmid: {pub_types, mesh}} from an efetch PubMed XML response."""
    out = {}
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"    XML parse error: {e}")
        return out

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        pmid = pmid_el.text.strip()

        pub_types = [
            (pt.text or "").strip()
            for pt in art.findall(".//PublicationTypeList/PublicationType")
            if (pt.text or "").strip()
        ]
        mesh = [
            (d.text or "").strip()
            for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if (d.text or "").strip()
        ]
        out[pmid] = {"pub_types": pub_types, "mesh": mesh}
    return out


def load_cache() -> dict:
    if META_CACHE.exists():
        try:
            with open(META_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print("  (cache unreadable, starting fresh)")
    return {}


def save_cache(cache: dict) -> None:
    tmp = META_CACHE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(META_CACHE)


def get_target_pmids(sample: int = 0) -> list:
    df = pd.read_csv(ASSOC_CSV, usecols=["PubMed ID"])
    pmids = (
        df["PubMed ID"].dropna().astype("int64").astype(str).unique().tolist()
    )
    pmids.sort()
    if sample:
        pmids = pmids[:sample]
    return pmids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="fetch only the first N PMIDs (for testing)")
    args = ap.parse_args()

    pmids = get_target_pmids(args.sample)
    cache = load_cache()

    todo = [p for p in pmids if p not in cache]
    print(f"Target PMIDs: {len(pmids):,}")
    print(f"Already cached: {len(pmids) - len(todo):,}")
    print(f"To fetch: {len(todo):,}")
    if NCBI_API_KEY:
        print("Using NCBI API key (10 req/sec).")
    if not todo:
        print("Nothing to do.")
        return

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        batch = todo[bi * BATCH_SIZE:(bi + 1) * BATCH_SIZE]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        xml_bytes = http_post(EFETCH_URL, params)
        parsed = parse_efetch_xml(xml_bytes)

        # Record every requested PMID, even ones PubMed returned nothing for,
        # so we don't re-request them forever.
        for p in batch:
            cache[p] = parsed.get(p, {"pub_types": [], "mesh": []})

        got = sum(1 for p in batch if cache[p]["pub_types"] or cache[p]["mesh"])
        print(f"  batch {bi + 1}/{n_batches}: requested {len(batch)}, "
              f"with metadata {got}")

        if (bi + 1) % SAVE_EVERY == 0:
            save_cache(cache)
            print(f"    ...cache saved ({len(cache):,} records)")
        time.sleep(RATE_LIMIT_DELAY)

    save_cache(cache)
    print(f"\nDone. Cache holds {len(cache):,} records at {META_CACHE.name}")


if __name__ == "__main__":
    main()
