"""
Transformer-based Relation Classifier for Spice-Disease Associations

Uses a zero-shot NLI (Natural Language Inference) model to classify
spice-disease relationships as positive (therapeutic), negative (adverse),
or neutral (mere co-occurrence).

Model: cross-encoder/nli-deberta-v3-small (~140MB, fits in 2GB VRAM)
Approach: For each evidence sentence, classify against hypothesis templates:
  - "This describes a therapeutic or beneficial effect" -> positive
  - "This describes an adverse or harmful effect" -> negative
  - "This is a neutral observation" -> neutral

Input:  spice_disease_associations_enhanced.csv
Output: spice_disease_associations_biobert.csv
"""

import csv
import json
import time
import re
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent
INPUT_CSV = BASE_DIR / "spice_disease_associations_enhanced.csv"
OUTPUT_CSV = BASE_DIR / "spice_disease_associations_biobert.csv"
SPICE_CSV = BASE_DIR / "pubmed_spice_results_clean.csv"
CACHE_FILE = BASE_DIR / "biobert_cache.json"

# Model selection — small enough for 2GB VRAM
MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# NLI hypothesis templates
HYPOTHESES = {
    "positive": "This text describes a therapeutic, beneficial, or protective effect of a substance on a disease or condition.",
    "negative": "This text describes an adverse, harmful, or toxic effect of a substance causing or worsening a disease or condition.",
    "neutral": "This text is a neutral scientific observation without clear therapeutic or adverse claims.",
}

# NLI label mapping: model outputs [contradiction, neutral, entailment]
# We check entailment score for each hypothesis
ENTAILMENT_IDX = 2  # Index for "entailment" in NLI output

BATCH_SIZE = 16  # Sentences per GPU batch (small for 2GB VRAM)
SAVE_INTERVAL = 1000  # Save cache every N sentences


def load_model():
    """Load the NLI model and tokenizer."""
    print(f"  Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    print(f"  Device: {device}")
    print(f"  VRAM used by model: {vram:.0f} MB")

    return tokenizer, model, device


def classify_batch(sentences: list[str], tokenizer, model, device) -> list[dict]:
    """Classify a batch of sentences using NLI zero-shot approach.

    For each sentence, tests it against each hypothesis and picks
    the one with the highest entailment score.

    Returns list of {nature, scores} dicts.
    """
    results = []

    for sentence in sentences:
        if not sentence.strip():
            results.append({"nature": "neutral", "scores": {"positive": 0, "negative": 0, "neutral": 1}})
            continue

        # Truncate very long sentences
        sentence = sentence[:512]

        best_label = "neutral"
        best_score = -1
        scores = {}

        for label, hypothesis in HYPOTHESES.items():
            # NLI input: premise (sentence) + hypothesis
            inputs = tokenizer(
                sentence, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=0)
                entailment_score = probs[ENTAILMENT_IDX].item()

            scores[label] = round(entailment_score, 4)

            if entailment_score > best_score:
                best_score = entailment_score
                best_label = label

        results.append({"nature": best_label, "scores": scores})

    return results


def get_best_sentence(row: dict, abstracts: dict) -> str:
    """Get the best sentence to classify for a given row.

    Priority:
    1. evidence_sentence (contains both spice + disease)
    2. Abstract sentence with both mentions
    3. Abstract sentence with disease mention
    4. First 2 sentences of abstract
    5. Title
    """
    # Priority 1: evidence sentence
    ev = row.get("evidence_sentence", "").strip()
    if ev:
        return ev[:512]

    pmid = row["pubmed_id"]
    abstract = abstracts.get(pmid, "")
    spice = row["spice"].lower()
    disease = row["disease"].lower()

    if abstract:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', abstract) if s.strip()]

        # Priority 2: sentence with both mentions
        for sent in sentences:
            if spice in sent.lower() and disease in sent.lower():
                return sent[:512]

        # Priority 3: sentence with disease mention
        for sent in sentences:
            if disease in sent.lower():
                return sent[:512]

        # Priority 4: first 2 sentences
        return ' '.join(sentences[:2])[:512]

    # Priority 5: title
    return (row.get("title", "") or "")[:512]


def main():
    print("=" * 65)
    print("  Transformer-based Relation Classifier (DeBERTa-v3 NLI)")
    print("=" * 65)

    # Step 1: Load data
    print(f"\n[Step 1] Loading data...")
    rows = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    print(f"  Loaded {len(rows):,} association rows")

    # Load abstracts
    abstracts = {}
    with open(SPICE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pmid = row["pubmed_id"]
            if pmid not in abstracts:
                abstracts[pmid] = row.get("abstract", "")
    print(f"  Loaded abstracts for {len(abstracts):,} PMIDs")

    # Step 2: Load model
    print(f"\n[Step 2] Loading transformer model...")
    tokenizer, model, device = load_model()

    # Step 3: Extract and deduplicate sentences
    print(f"\n[Step 3] Extracting and deduplicating sentences...")
    row_sentences = []  # (row_idx, sentence) for each row
    unique_sentences = {}  # sentence_key -> sentence

    for i, row in enumerate(rows):
        sentence = get_best_sentence(row, abstracts)
        key = sentence[:200]  # Cache key
        row_sentences.append((i, key))
        if key not in unique_sentences:
            unique_sentences[key] = sentence

    print(f"  Total rows: {len(rows):,}")
    print(f"  Unique sentences to classify: {len(unique_sentences):,}")
    print(f"  Dedup ratio: {len(unique_sentences)/len(rows)*100:.1f}%")

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached classifications")

    # Step 4: Classify unique sentences only
    to_classify = [(k, s) for k, s in unique_sentences.items() if k not in cache]
    already_cached = len(unique_sentences) - len(to_classify)
    print(f"\n[Step 4] Running transformer classification...")
    print(f"  Already cached: {already_cached:,}")
    print(f"  Need to classify: {len(to_classify):,}")
    print(f"  Device: {device}")

    start_time = time.time()
    total = len(to_classify)

    for i, (key, sentence) in enumerate(to_classify):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate / 60
            print(f"    {i+1:,}/{total:,} "
                  f"({elapsed/60:.1f}min elapsed, ~{remaining:.0f}min remaining, "
                  f"{rate:.1f} sent/sec)", end="\r", flush=True)

        results = classify_batch([sentence], tokenizer, model, device)
        cache[key] = results[0]

        # Periodic save
        if (i + 1) % SAVE_INTERVAL == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)

    # Final cache save
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    elapsed = time.time() - start_time
    print(f"\n  Classified {len(to_classify):,} unique sentences in {elapsed/60:.1f} minutes")

    # Step 4b: Apply results to all rows
    print(f"\n  Applying results to all {len(rows):,} rows...")
    for row_idx, key in row_sentences:
        result = cache.get(key, {"nature": "neutral", "scores": {"positive": 0, "negative": 0, "neutral": 1}})
        rows[row_idx]["association_nature_transformer"] = result["nature"]
        rows[row_idx]["transformer_score_positive"] = result["scores"].get("positive", 0)
        rows[row_idx]["transformer_score_negative"] = result["scores"].get("negative", 0)
        rows[row_idx]["transformer_score_neutral"] = result["scores"].get("neutral", 0)

    # Step 5: Merge — use transformer result as primary, keep all other columns
    print(f"\n[Step 5] Writing output...")

    # Determine final association_nature using confidence-weighted ensemble:
    # - If transformer is confident (score >= 0.5), trust it
    # - If both agree, boost confidence
    # - If they disagree and transformer is low-confidence, trust enhanced
    TRANSFORMER_CONFIDENCE_THRESHOLD = 0.5

    for row in rows:
        trans_nature = row.get("association_nature_transformer", "neutral")
        enhanced_nature = row.get("association_nature", "neutral")

        trans_score = max(
            float(row.get("transformer_score_positive", 0)),
            float(row.get("transformer_score_negative", 0)),
            float(row.get("transformer_score_neutral", 0)),
        )
        original_conf = float(row.get("confidence", 0.5))

        row["classifier_agreement"] = "agree" if trans_nature == enhanced_nature else "disagree"

        if trans_nature == enhanced_nature:
            # Both agree — high confidence, use shared label
            row["association_nature_final"] = trans_nature
            row["confidence_final"] = round(min(1.0, 0.5 * trans_score + 0.5 * original_conf + 0.1), 3)
        elif trans_score >= TRANSFORMER_CONFIDENCE_THRESHOLD:
            # Transformer is confident — trust it over enhanced
            row["association_nature_final"] = trans_nature
            row["confidence_final"] = round(0.6 * trans_score + 0.4 * original_conf, 3)
        else:
            # Transformer is low-confidence — trust enhanced classifier
            row["association_nature_final"] = enhanced_nature
            row["confidence_final"] = round(0.4 * trans_score + 0.6 * original_conf, 3)

    fieldnames = [
        "pubmed_id", "spice", "disease", "mesh_id", "disease_category",
        "association_nature_final", "association_nature", "association_nature_original",
        "association_nature_transformer", "classifier_agreement",
        "confidence_final", "confidence", "confidence_original",
        "transformer_score_positive", "transformer_score_negative", "transformer_score_neutral",
        "co_occurrence_level", "spice_location", "disease_location",
        "is_negated", "is_hedged", "study_type",
        "evidence_sentence", "publication_date", "title",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written {len(rows):,} rows to {OUTPUT_CSV.name}")

    # Step 6: Stats
    print(f"\n{'='*65}")
    print(f"  TRANSFORMER CLASSIFICATION RESULTS")
    print(f"{'='*65}")

    # Nature comparison across all 3 methods
    orig_counts = Counter(r.get("association_nature_original", "") for r in rows)
    enhanced_counts = Counter(r.get("association_nature", "") for r in rows)
    trans_counts = Counter(r.get("association_nature_transformer", "") for r in rows)
    final_counts = Counter(r.get("association_nature_final", "") for r in rows)

    print(f"\n  Classification comparison:")
    print(f"  {'Category':<12s} {'Original':>10s} {'Enhanced':>10s} {'Transformer':>12s} {'Final':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    for cat in ["positive", "negative", "neutral"]:
        print(f"  {cat:<12s} {orig_counts.get(cat,0):>10,} "
              f"{enhanced_counts.get(cat,0):>10,} "
              f"{trans_counts.get(cat,0):>12,} "
              f"{final_counts.get(cat,0):>10,}")

    # Agreement rate
    agree = sum(1 for r in rows if r.get("classifier_agreement") == "agree")
    total = len(rows)
    print(f"\n  Enhanced vs Transformer agreement: {agree:,}/{total:,} ({agree/total*100:.1f}%)")

    # Confidence comparison
    final_confs = [float(r["confidence_final"]) for r in rows]
    avg_final = sum(final_confs) / len(final_confs)
    high = sum(1 for c in final_confs if c >= 0.7)
    med = sum(1 for c in final_confs if 0.4 <= c < 0.7)
    low = sum(1 for c in final_confs if c < 0.4)

    print(f"\n  Final confidence: avg={avg_final:.3f}")
    print(f"    High (>=0.7):   {high:>10,}")
    print(f"    Medium (0.4-0.7): {med:>8,}")
    print(f"    Low (<0.4):     {low:>10,}")

    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
