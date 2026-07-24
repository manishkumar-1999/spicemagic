"""
SpiceDisease: Spice-Disease Association Mining from PubMed
Based on the SpiceRx methodology (bioRxiv 273599v1)

Pipeline:
1. Load spice/herb NER dictionary (167 entries with synonyms)
2. Search PubMed abstracts for spice mentions (dictionary-based NER)
3. Tag diseases using MeSH terms
4. Extract spice-disease co-occurrences
"""

import csv
import re
from pathlib import Path


def load_spice_dictionary(filepath: str = "spice_ner_dictionary.csv") -> list[dict]:
    """Load the spice NER dictionary from CSV.

    Returns a list of dicts with keys: Main_Name, Synonyms (list), Scientific_Name,
    and a compiled regex pattern for matching.
    """
    spices = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            synonyms = [s.strip() for s in row["Synonyms"].split(";") if s.strip()]
            # Build all name variants for NER matching
            all_names = [row["Main_Name"]] + synonyms
            if row["Scientific_Name"]:
                all_names.append(row["Scientific_Name"])

            # Sort by length descending so longer names match first
            all_names.sort(key=len, reverse=True)

            # Compile a regex pattern for this spice (case-insensitive, whole word)
            escaped = [re.escape(name) for name in all_names]
            pattern = re.compile(
                r"\b(?:" + "|".join(escaped) + r")\b",
                re.IGNORECASE,
            )

            spices.append({
                "main_name": row["Main_Name"],
                "synonyms": synonyms,
                "scientific_name": row["Scientific_Name"],
                "all_names": all_names,
                "pattern": pattern,
            })
    return spices


def find_spices_in_text(text: str, spice_dict: list[dict]) -> list[dict]:
    """Find all spice mentions in a given text using dictionary-based NER.

    Uses longest-match-first disambiguation: when matches overlap, keeps the
    longest match and discards shorter substrings. For same-span matches from
    different spice entries (e.g., shared scientific names), keeps all.

    Returns list of dicts with main_name, matched_term, start, end positions.
    """
    raw_matches = []
    for spice in spice_dict:
        for match in spice["pattern"].finditer(text):
            raw_matches.append({
                "main_name": spice["main_name"],
                "scientific_name": spice["scientific_name"],
                "matched_term": match.group(),
                "start": match.start(),
                "end": match.end(),
            })

    # Sort by start position, then by match length descending (longest first)
    raw_matches.sort(key=lambda m: (m["start"], -(m["end"] - m["start"])))

    # Longest-match-first: discard matches that are strict subspans of a longer match
    filtered = []
    for m in raw_matches:
        is_subspan = False
        for kept in filtered:
            if m["start"] >= kept["start"] and m["end"] <= kept["end"] and m["main_name"] != kept["main_name"]:
                # This match is inside a longer match from a different spice
                # e.g. "basil" inside "holy basil" -> discard "basil"
                if (m["end"] - m["start"]) < (kept["end"] - kept["start"]):
                    is_subspan = True
                    break
        if not is_subspan:
            filtered.append(m)

    # Deduplicate same-span same-spice matches
    seen = set()
    deduped = []
    for m in filtered:
        key = (m["main_name"], m["start"], m["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(m)

    deduped.sort(key=lambda m: m["start"])
    return deduped


def print_dictionary_stats(spice_dict: list[dict]) -> None:
    """Print summary statistics of the loaded dictionary."""
    total_spices = len(spice_dict)
    total_synonyms = sum(len(s["synonyms"]) for s in spice_dict)
    with_scientific = sum(1 for s in spice_dict if s["scientific_name"])
    total_name_variants = sum(len(s["all_names"]) for s in spice_dict)

    print(f"Spice NER Dictionary Statistics:")
    print(f"  Total spices/herbs:     {total_spices}")
    print(f"  Total synonyms:         {total_synonyms}")
    print(f"  With scientific names:  {with_scientific}")
    print(f"  Total name variants:    {total_name_variants}")
    print()


def demo():
    """Demonstrate NER on sample text."""
    dict_path = Path(__file__).parent / "spice_ner_dictionary.csv"
    spice_dict = load_spice_dictionary(str(dict_path))
    print_dictionary_stats(spice_dict)

    # Demo: find spices in a sample PubMed-like abstract
    sample_text = (
        "Curcumin, the active compound in turmeric (Curcuma longa), has shown "
        "anti-inflammatory properties. Black pepper (Piper nigrum) enhances "
        "curcumin bioavailability. Ginger and cinnamon also exhibit antioxidant "
        "effects. A combination of fenugreek, holy basil (tulsi), and long pepper "
        "was studied for its effects on diabetes and cardiovascular disease. "
        "Saffron (Crocus sativus) has been investigated for Alzheimer's disease."
    )

    print("Sample text:")
    print(f"  {sample_text[:100]}...")
    print()

    matches = find_spices_in_text(sample_text, spice_dict)
    print(f"Found {len(matches)} spice mentions:")
    for m in matches:
        print(f"  [{m['start']:3d}-{m['end']:3d}] {m['matched_term']:<25s} -> {m['main_name']} ({m['scientific_name']})")


if __name__ == "__main__":
    demo()
