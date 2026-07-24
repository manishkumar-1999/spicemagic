"""
SpiceRx-style CNN Relation Classifier for Spice-Disease Associations

Replicates the SpiceRx (Rakhi et al., PLOS One 2018) methodology:
  - Sentence-level CNN with word + position + POS embeddings
  - Trained on SpiceRx's annotated sentences (downloaded from GitHub)
  - Entities marked as ANN-FOOD and ANN-DISEASE in sentences
  - 3-class output: Positive / Negative / Neutral

Key differences from SpiceRx:
  - Uses spaCy for POS tags instead of GeniaTagger (cross-platform)
  - PyTorch CNN instead of Keras (more modern)
  - Combined dataset: SpiceRx 1,046 + our high-confidence associations

Input:  spice_disease_associations_enhanced.csv
Output: spice_disease_associations_spicerx_cnn.csv

Usage:
  python spicerx_cnn_classifier.py
"""

import csv
import json
import re
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
INPUT_CSV  = BASE_DIR / "spice_disease_associations_enhanced.csv"
OUTPUT_CSV = BASE_DIR / "spice_disease_associations_spicerx_cnn.csv"
SPICE_CSV  = BASE_DIR / "pubmed_spice_results_clean.csv"
SPICERX_DATA_FILE = BASE_DIR / "spicerx_annotated.tsv"
MODEL_SAVE = BASE_DIR / "spicerx_cnn_model.pt"
VOCAB_SAVE = BASE_DIR / "spicerx_vocab.json"

# SpiceRx GitHub annotated data URL
SPICERX_TSV_URL = (
    "https://raw.githubusercontent.com/cosylabiiit/"
    "spice-disease-associations/master/data/annotated-sentences.tsv"
)

# Model hyperparameters (matching SpiceRx)
MAX_SEQ_LEN    = 100   # max tokens per sentence
EMBED_DIM      = 100   # word embedding dimension
POS_EMBED_DIM  = 25    # POS tag embedding dimension
DIST_EMBED_DIM = 25    # position distance embedding dimension
NUM_FILTERS    = 100   # CNN filters per kernel size
KERNEL_SIZES   = [2, 3, 4]
DROPOUT        = 0.5
NUM_CLASSES    = 3
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 1e-3
MAX_DIST       = 50    # clip position distances at ±50

# Labels
LABEL2IDX = {"Positive": 0, "Negative": 1, "Neutral": 2}
IDX2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

# POS tag vocabulary (universal)
POS_TAGS = [
    "PAD", "UNK", "NOUN", "VERB", "ADJ", "ADV", "ADP",
    "DET", "PRON", "NUM", "PUNCT", "CONJ", "PART", "X", "SPACE"
]
POS2IDX = {p: i for i, p in enumerate(POS_TAGS)}


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD SPICERX ANNOTATED DATA
# ═══════════════════════════════════════════════════════════════════

def download_spicerx_data() -> list[dict]:
    """Download SpiceRx annotated sentences from GitHub."""
    if SPICERX_DATA_FILE.exists():
        print(f"  Loading cached SpiceRx data from {SPICERX_DATA_FILE.name}...")
    else:
        print(f"  Downloading SpiceRx annotated sentences from GitHub...")
        req = urllib.request.Request(
            SPICERX_TSV_URL,
            headers={"User-Agent": "SpiceDisease/1.0 (research)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        with open(SPICERX_DATA_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved to {SPICERX_DATA_FILE.name}")

    rows = []
    with open(SPICERX_DATA_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("Labels", "").strip()
            sentence = row.get("Sentence", "").strip()
            if label in LABEL2IDX and sentence:
                rows.append({
                    "sentence": sentence,
                    "label": label,
                    "spice": row.get("Food Text", "").strip(),
                    "disease": row.get("Disease Text", "").strip(),
                    "source": "spicerx",
                })
    print(f"  Loaded {len(rows):,} annotated sentences from SpiceRx")
    return rows


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — LOAD OUR HIGH-CONFIDENCE ASSOCIATIONS
# ═══════════════════════════════════════════════════════════════════

def load_our_high_confidence(min_confidence: float = 0.85,
                              max_samples: int = 5000) -> list[dict]:
    """
    Load high-confidence associations from our dataset where all 3
    classifiers agree. Use as additional training data.
    """
    if not INPUT_CSV.exists():
        print("  Input CSV not found, skipping our data")
        return []

    rows = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only use rows where all 3 classifiers agree
            orig   = row.get("association_nature_original", "")
            enh    = row.get("association_nature", "")
            trans  = row.get("association_nature_transformer", "")
            conf   = float(row.get("confidence", 0))
            sent   = row.get("evidence_sentence", "").strip()
            spice  = row.get("spice", "").strip()
            disease = row.get("disease", "").strip()

            if not sent or not spice or not disease:
                continue
            if conf < min_confidence:
                continue

            # All 3 must agree
            nature_map = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}
            orig_std  = nature_map.get(orig, "")
            enh_std   = nature_map.get(enh, "")
            trans_std = nature_map.get(trans, "")

            if orig_std == enh_std == trans_std and orig_std in LABEL2IDX:
                # Replace actual spice/disease names with ANN-FOOD/ANN-DISEASE
                ann_sent = re.sub(re.escape(spice), "ANN-FOOD", sent, flags=re.IGNORECASE)
                ann_sent = re.sub(re.escape(disease), "ANN-DISEASE", ann_sent, flags=re.IGNORECASE)

                rows.append({
                    "sentence": ann_sent,
                    "label": orig_std,
                    "spice": spice,
                    "disease": disease,
                    "source": "ours",
                })

    # Balance classes and cap total
    random.seed(42)
    by_label = {"Positive": [], "Negative": [], "Neutral": []}
    for r in rows:
        by_label[r["label"]].append(r)

    # Cap each class
    per_class = min(max_samples // 3, min(len(v) for v in by_label.values() if v))
    balanced = []
    for label, items in by_label.items():
        balanced.extend(random.sample(items, min(per_class, len(items))))

    random.shuffle(balanced)
    print(f"  Loaded {len(balanced):,} high-confidence associations from our dataset")
    print(f"    Positive: {sum(1 for r in balanced if r['label']=='Positive'):,}")
    print(f"    Negative: {sum(1 for r in balanced if r['label']=='Negative'):,}")
    print(f"    Neutral:  {sum(1 for r in balanced if r['label']=='Neutral'):,}")
    return balanced


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — TOKENIZATION AND FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens


def get_pos_tags(tokens: list[str]) -> list[str]:
    """
    Simple rule-based POS tagger (no external dependency).
    Approximates POS tags for CNN training.
    Falls back to spaCy if available.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(" ".join(tokens))
        return [token.pos_ for token in doc]
    except Exception:
        pass

    # Fallback: simple rule-based tagger
    pos = []
    for tok in tokens:
        if tok in ("ann-food", "ann-disease"):
            pos.append("NOUN")
        elif tok in ("the", "a", "an", "this", "these", "that", "those"):
            pos.append("DET")
        elif tok in ("in", "of", "for", "on", "at", "by", "with", "from", "to"):
            pos.append("ADP")
        elif tok in ("and", "or", "but", "nor", "so"):
            pos.append("CONJ")
        elif tok in ("is", "are", "was", "were", "be", "been",
                     "show", "showed", "shows", "suggest", "suggests",
                     "reduce", "reduced", "reduces", "inhibit", "inhibits",
                     "cause", "causes", "caused", "induce", "induces",
                     "treat", "treats", "treated", "prevent", "prevents"):
            pos.append("VERB")
        elif tok in ("significant", "clinical", "therapeutic", "adverse",
                     "positive", "negative", "anti", "high", "low"):
            pos.append("ADJ")
        elif tok in ("not", "no", "never", "neither"):
            pos.append("ADV")
        elif tok.isdigit():
            pos.append("NUM")
        elif tok in (".", ",", ";", ":", "!", "?", "(", ")", "-"):
            pos.append("PUNCT")
        else:
            pos.append("NOUN")  # default guess
    return pos


def get_position_distances(tokens: list[str],
                            entity: str) -> list[int]:
    """
    Compute distance of each token from the nearest occurrence
    of the entity token (ANN-FOOD or ANN-DISEASE).
    Clips distances to [-MAX_DIST, MAX_DIST].
    """
    entity_lower = entity.lower()
    entity_indices = [i for i, t in enumerate(tokens) if t == entity_lower]

    distances = []
    for i in range(len(tokens)):
        if not entity_indices:
            dist = MAX_DIST
        else:
            dist = min(abs(i - idx) for idx in entity_indices)
        distances.append(min(dist, MAX_DIST))
    return distances


def build_vocab(sentences: list[list[str]], min_freq: int = 1) -> dict:
    """Build word vocabulary from tokenized sentences."""
    counter = Counter()
    for tokens in sentences:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_sentence(tokens: list[str], pos_tags: list[str],
                    food_dists: list[int], dis_dists: list[int],
                    vocab: dict, max_len: int):
    """
    Encode a sentence into padded index arrays.
    Returns: (word_ids, pos_ids, food_dist_ids, dis_dist_ids)
    """
    # Truncate
    tokens    = tokens[:max_len]
    pos_tags  = pos_tags[:max_len]
    food_dists = food_dists[:max_len]
    dis_dists  = dis_dists[:max_len]

    # Pad
    pad_len = max_len - len(tokens)
    word_ids     = [vocab.get(t, 1) for t in tokens] + [0] * pad_len
    pos_ids      = [POS2IDX.get(p, 1) for p in pos_tags] + [0] * pad_len
    # Distance IDs: shift by MAX_DIST so range is [0, 2*MAX_DIST]
    food_dist_ids = [d + MAX_DIST for d in food_dists] + [MAX_DIST] * pad_len
    dis_dist_ids  = [d + MAX_DIST for d in dis_dists]  + [MAX_DIST] * pad_len

    return word_ids, pos_ids, food_dist_ids, dis_dist_ids


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════

class SpiceRxDataset(Dataset):
    """PyTorch Dataset for SpiceRx-style CNN training."""

    def __init__(self, samples: list[dict], vocab: dict, max_len: int = MAX_SEQ_LEN):
        self.data = []
        for s in samples:
            tokens     = simple_tokenize(s["sentence"])
            pos_tags   = get_pos_tags(tokens)
            food_dists = get_position_distances(tokens, "ann-food")
            dis_dists  = get_position_distances(tokens, "ann-disease")

            word_ids, pos_ids, food_dist_ids, dis_dist_ids = encode_sentence(
                tokens, pos_tags, food_dists, dis_dists, vocab, max_len
            )

            label = LABEL2IDX[s["label"]]
            self.data.append((word_ids, pos_ids, food_dist_ids, dis_dist_ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word_ids, pos_ids, food_dist_ids, dis_dist_ids, label = self.data[idx]
        return (
            torch.tensor(word_ids,      dtype=torch.long),
            torch.tensor(pos_ids,       dtype=torch.long),
            torch.tensor(food_dist_ids, dtype=torch.long),
            torch.tensor(dis_dist_ids,  dtype=torch.long),
            torch.tensor(label,         dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — CNN MODEL (SpiceRx architecture)
# ═══════════════════════════════════════════════════════════════════

class SpiceRxCNN(nn.Module):
    """
    CNN Relation Classifier — replicates SpiceRx architecture.

    4 embedding streams:
      1. Word embeddings
      2. POS tag embeddings
      3. Distance from ANN-FOOD embeddings
      4. Distance from ANN-DISEASE embeddings

    All streams concatenated → Conv1D → MaxPool → Dropout → Softmax
    """

    def __init__(self, vocab_size: int, num_pos: int = len(POS_TAGS),
                 num_dist: int = 2 * MAX_DIST + 1):
        super().__init__()

        # Embedding layers
        self.word_embed  = nn.Embedding(vocab_size, EMBED_DIM,    padding_idx=0)
        self.pos_embed   = nn.Embedding(num_pos,    POS_EMBED_DIM, padding_idx=0)
        self.food_embed  = nn.Embedding(num_dist,   DIST_EMBED_DIM, padding_idx=MAX_DIST)
        self.dis_embed   = nn.Embedding(num_dist,   DIST_EMBED_DIM, padding_idx=MAX_DIST)

        # Total input dim per token
        total_dim = EMBED_DIM + POS_EMBED_DIM + DIST_EMBED_DIM + DIST_EMBED_DIM

        # Parallel CNN filters for different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=total_dim,
                out_channels=NUM_FILTERS,
                kernel_size=k,
                padding=k // 2,
            )
            for k in KERNEL_SIZES
        ])

        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), NUM_CLASSES)

    def forward(self, word_ids, pos_ids, food_dist_ids, dis_dist_ids):
        # Embed all features — shape: (batch, seq_len, embed_dim)
        w = self.word_embed(word_ids)
        p = self.pos_embed(pos_ids)
        f = self.food_embed(food_dist_ids)
        d = self.dis_embed(dis_dist_ids)

        # Concatenate along embedding dim
        x = torch.cat([w, p, f, d], dim=-1)  # (batch, seq_len, total_dim)

        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Apply each conv filter + ReLU + global max pool
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))             # (batch, num_filters, seq_len')
            c = torch.max(c, dim=-1).values     # (batch, num_filters)
            pooled.append(c)

        # Concatenate all filter outputs
        x = torch.cat(pooled, dim=-1)           # (batch, num_filters * len(kernels))
        x = self.dropout(x)
        logits = self.fc(x)                     # (batch, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(train_data: list[dict], val_data: list[dict],
                vocab: dict, device: torch.device) -> SpiceRxCNN:
    """Train the SpiceRx CNN model."""

    train_dataset = SpiceRxDataset(train_data, vocab)
    val_dataset   = SpiceRxDataset(val_data,   vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    model = SpiceRxCNN(vocab_size=len(vocab)).to(device)

    # Class weights to handle imbalance (negative class is rare)
    label_counts = Counter(s["label"] for s in train_data)
    total = sum(label_counts.values())
    weights = torch.tensor([
        total / (label_counts.get("Positive", 1) * NUM_CLASSES),
        total / (label_counts.get("Negative", 1) * NUM_CLASSES),
        total / (label_counts.get("Neutral",  1) * NUM_CLASSES),
    ], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_val_acc = 0.0
    best_state   = None

    print(f"\n  Training SpiceRx CNN...")
    print(f"  {'Epoch':<6} {'Train Loss':<12} {'Val Acc':<10} {'Status'}")
    print(f"  {'-'*45}")

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            word_ids, pos_ids, food_dist, dis_dist, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(word_ids, pos_ids, food_dist, dis_dist)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                word_ids, pos_ids, food_dist, dis_dist, labels = [b.to(device) for b in batch]
                logits = model(word_ids, pos_ids, food_dist, dis_dist)
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(1 - val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            status = "<-- best"

        print(f"  {epoch:<6} {avg_loss:<12.4f} {val_acc:<10.4f} {status}")

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    return model


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(model: SpiceRxCNN, test_data: list[dict],
                   vocab: dict, device: torch.device):
    """Print detailed evaluation report."""
    test_dataset = SpiceRxDataset(test_data, vocab)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            word_ids, pos_ids, food_dist, dis_dist, labels = [b.to(device) for b in batch]
            logits = model(word_ids, pos_ids, food_dist, dis_dist)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    label_names = ["Positive", "Negative", "Neutral"]
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS (Test Set)")
    print(f"{'='*55}")
    print(f"\n  Overall Accuracy: {accuracy_score(all_labels, all_preds)*100:.1f}%")
    print(f"\n  Per-class Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=3
    ))
    print(f"\n  SpiceRx baseline (CNN):    86.7%")
    print(f"  Our zero-shot (DeBERTa):  ~89.0%")
    print(f"  This model:               {accuracy_score(all_labels, all_preds)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — PREDICT ON ALL OUR ASSOCIATIONS
# ═══════════════════════════════════════════════════════════════════

def predict_all_associations(model: SpiceRxCNN, vocab: dict,
                              device: torch.device):
    """Run trained CNN on all our 250K associations."""
    if not INPUT_CSV.exists():
        print("  Input CSV not found, skipping prediction")
        return

    print(f"\n[Step 8] Predicting on all associations...")
    rows = []
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows):,} associations")

    model.eval()
    predictions = []
    batch_size  = 512

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]
        batch_tensors = []

        for row in batch_rows:
            sent   = row.get("evidence_sentence", "").strip()
            spice  = row.get("spice", "").strip()
            disease = row.get("disease", "").strip()

            # Replace entity names with ANN tokens
            if sent and spice:
                sent = re.sub(re.escape(spice), "ANN-FOOD", sent, flags=re.IGNORECASE)
            if sent and disease:
                sent = re.sub(re.escape(disease), "ANN-DISEASE", sent, flags=re.IGNORECASE)

            tokens     = simple_tokenize(sent) if sent else ["ann-food", "ann-disease"]
            pos_tags   = get_pos_tags(tokens)
            food_dists = get_position_distances(tokens, "ann-food")
            dis_dists  = get_position_distances(tokens, "ann-disease")

            word_ids, pos_ids, food_dist_ids, dis_dist_ids = encode_sentence(
                tokens, pos_tags, food_dists, dis_dists, vocab, MAX_SEQ_LEN
            )
            batch_tensors.append((word_ids, pos_ids, food_dist_ids, dis_dist_ids))

        # Stack into tensors
        word_t = torch.tensor([b[0] for b in batch_tensors], dtype=torch.long).to(device)
        pos_t  = torch.tensor([b[1] for b in batch_tensors], dtype=torch.long).to(device)
        food_t = torch.tensor([b[2] for b in batch_tensors], dtype=torch.long).to(device)
        dis_t  = torch.tensor([b[3] for b in batch_tensors], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(word_t, pos_t, food_t, dis_t)
            probs  = torch.softmax(logits, dim=-1).cpu()
            preds  = logits.argmax(dim=-1).cpu().tolist()

        for j, pred in enumerate(preds):
            predictions.append({
                "label": IDX2LABEL[pred],
                "score_positive": round(probs[j][0].item(), 3),
                "score_negative": round(probs[j][1].item(), 3),
                "score_neutral":  round(probs[j][2].item(), 3),
            })

        if (i // batch_size + 1) % 50 == 0:
            print(f"    Processed {i+len(batch_rows):,}/{len(rows):,}", end="\r", flush=True)

    print(f"\n  Predictions complete")

    # Write output CSV
    new_fieldnames = list(fieldnames) + [
        "association_nature_spicerx_cnn",
        "cnn_score_positive",
        "cnn_score_negative",
        "cnn_score_neutral",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, predictions):
            row["association_nature_spicerx_cnn"] = pred["label"]
            row["cnn_score_positive"] = pred["score_positive"]
            row["cnn_score_negative"] = pred["score_negative"]
            row["cnn_score_neutral"]  = pred["score_neutral"]
            writer.writerow(row)

    # Stats
    nature_counts = Counter(p["label"] for p in predictions)
    print(f"\n  CNN Prediction Distribution:")
    for label, count in nature_counts.most_common():
        print(f"    {label:<12} {count:>8,}  ({count/len(predictions)*100:.1f}%)")
    print(f"\n  Output: {OUTPUT_CSV.name}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SpiceRx-style CNN Classifier")
    print("  Replicating Rakhi et al. PLOS One 2018")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Step 1: Load SpiceRx annotated data
    print(f"\n[Step 1] Loading SpiceRx annotated data...")
    spicerx_data = download_spicerx_data()

    label_dist = Counter(s["label"] for s in spicerx_data)
    print(f"  Label distribution:")
    for label, count in label_dist.most_common():
        print(f"    {label:<12} {count:>5,}")

    # Step 2: Load our high-confidence associations
    print(f"\n[Step 2] Loading our high-confidence associations...")
    our_data = load_our_high_confidence(min_confidence=0.85, max_samples=3000)

    # Combine datasets
    all_data = spicerx_data + our_data
    random.seed(42)
    random.shuffle(all_data)
    print(f"\n  Combined dataset: {len(all_data):,} samples")

    # Step 3: Build vocabulary
    print(f"\n[Step 3] Building vocabulary...")
    all_tokens = [simple_tokenize(s["sentence"]) for s in all_data]
    vocab = build_vocab(all_tokens, min_freq=1)
    print(f"  Vocabulary size: {len(vocab):,} tokens")

    # Save vocab for later inference
    with open(VOCAB_SAVE, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(f"  Saved vocab to {VOCAB_SAVE.name}")

    # Step 4: Train/val/test split (70/15/15)
    print(f"\n[Step 4] Splitting data...")
    train_data, temp_data = train_test_split(all_data, test_size=0.30, random_state=42,
                                              stratify=[s["label"] for s in all_data])
    val_data, test_data   = train_test_split(temp_data, test_size=0.50, random_state=42,
                                              stratify=[s["label"] for s in temp_data])
    print(f"  Train: {len(train_data):,}  Val: {len(val_data):,}  Test: {len(test_data):,}")

    # Step 5: Train model
    print(f"\n[Step 5] Training CNN model (SpiceRx architecture)...")
    model = train_model(train_data, val_data, vocab, device)

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"  Model saved to {MODEL_SAVE.name}")

    # Step 6: Evaluate on test set
    print(f"\n[Step 6] Evaluating on test set...")
    evaluate_model(model, test_data, vocab, device)

    # Step 7: Predict on all our associations
    print(f"\n[Step 7] Running CNN on all our 250K associations...")
    predict_all_associations(model, vocab, device)

    print(f"\n{'='*60}")
    print(f"  Done! SpiceRx CNN classification complete.")
    print(f"  Output: {OUTPUT_CSV.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
