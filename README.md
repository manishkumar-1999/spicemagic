# 🌿 SpiceMagic — Spice-Disease Association Classifier

An AI-powered interactive web application for exploring **228,750 spice-disease
associations** mined from PubMed biomedical literature, classified by a
fine-tuned **PubMedBERT** model that achieves **90.00% accuracy** —
outperforming the SpiceRx CNN baseline (86.70%) by **3.30%**.

---

## 🎯 Project Overview

SpiceMagic mines PubMed abstracts to build a large, high-confidence database
of spice-disease relationships, each classified as:

- 🟢 **Positive** — spice benefits the condition (53.4%)
- 🟡 **Neutral** — no clear effect / study aim (37.1%)
- 🔴 **Negative** — spice may cause harm (9.6%)

The final dataset covers **164 unique spices**, **37,871 unique diseases**,
and **47,858 unique PubMed papers**.

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **90.00%** |
| F1 Score | 89.95% |
| MCC | 0.818 |
| Sensitivity | 90.00% |
| Specificity | 94.30% |

**+3.30% improvement over the SpiceRx CNN published baseline (86.70%).**

### Top 5 Models Compared

| Rank | Model | Accuracy | F1 | MCC |
|------|-------|----------|-----|-----|
| 🥇 1 | PubMedBERT Multi-Seed (Refined 90%) | **90.00%** | 89.95% | 0.818 |
| 🥈 2 | PubMedBERT Multi-Seed Ensemble | 89.33% | 89.25% | 0.805 |
| 🥉 3 | K-Fold + Multi-Seed (3 seeds) | 88.49% | 88.45% | 0.790 |
| 4 | PubMedBERT (Cleaned 4500) | 88.36% | 88.39% | 0.741 |
| 5 | K-Fold + Multi-Seed (5s Tuned) | 88.36% | 88.35% | 0.788 |

See `results/model_comparisonfinal.xlsx` for the full 14-model comparison.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the web app locally
```bash
python -m streamlit run spicerx_app.py
```
Opens at `http://localhost:8501` — full interactive dashboard.

### 3. Deploy to Streamlit Cloud
1. Push this repo to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud) → New app
3. Point to your repo, main file: `spicerx_app.py`
4. Deploy — done.

---

## 🌟 Web App Features (11 Pages)

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | Database overview with key metrics + sentiment distribution |
| 🧠 **Live Prediction** | Quick Search: pick spice + disease, see instant results |
| 🔎 **Search & Explore** | Autocomplete filters + timeline slider + CSV download |
| 🌶️ **Spice Profile** | Full analytics per spice with PubMed links |
| 🧬 **Disease Profile** | Full analytics per disease |
| ⚖️ **Compare** | Side-by-side spice-vs-spice comparison |
| 🌊 **Sankey Flow** | Publication-quality flow: Spice → Sentiment → Disease |
| 📈 **Analytics** | Beneficial spices, time trends, confidence analysis |
| 💬 **AI Chat** | Natural-language queries over the database |
| ✉️ **Contact** | Real contact form (via Formsubmit) with social links |
| 💡 **About** | Project overview + author bio + methodology |

**UI polish:** Dark mode, visitor counter, rotating "Did you know?" facts, PubMed links in all tables.

---

## 📁 Repository Structure

```
Spicemagic/
│
├── spicerx_app.py                          # Streamlit web application
├── requirements.txt                        # Python dependencies
├── README.md                               # This file
├── .gitignore                              # Files excluded from git
│
├── final_database_v2.csv          (88 MB)  # ⭐ Main dataset (228,750 labeled)
├── spice_disease_associations.csv (64 MB)  # Raw associations (pre-labeling)
├── annotation_cleaned.xlsx                 # 4,500-row training set (refined)
├── disease_annotations.csv                 # PubTator3 disease NER output
├── spice_ner_dictionary.csv                # Spice NER lookup table
│
├── Assets (used by the app):
│   ├── cosylab_logo.png                    # Top nav logo
│   ├── bagler_photo.jpg                    # Author photo (About page)
│   ├── iiitd_logo.png                      # Footer institution logo
│   └── cc_badge.png                        # Creative Commons badge
│
├── scripts/                                # Data pipeline (Python)
│   ├── main.py                             # Pipeline entry point
│   ├── pubmed_search.py                    # PubMed API collection
│   ├── disease_ner.py                      # Disease entity recognition
│   ├── association_mining.py               # Spice-disease pairing
│   ├── phase4_analysis.py                  # Extract insights → insights.xlsx
│   └── phase5_visualizations.py            # Generate 11 analysis charts
│
├── notebooks/                              # Model training
│   ├── MultiSeed_Ensemble_Colab.ipynb      # ⭐ Trains the 90% champion
│   └── Phase3_v2_Predict_250k_Kaggle.ipynb # Predicts on 250k rows
│
├── baselines/                              # Classical model comparisons
│   ├── spicerx_cnn_classifier.py           # SpiceRx CNN (2018 baseline)
│   ├── biobert_classifier.py               # BioBERT alternative
│   ├── enhanced_classifier.py              # Enhanced ensemble
│   ├── cross_validation.py                 # SVM / LR / RF / XGBoost
│   ├── cross_validation_advanced.py        # Stacking + voting
│   └── cross_validation_smote.py           # SMOTE + GridSearch
│
├── results/                                # Analysis outputs
│   ├── model_comparisonfinal.xlsx          # 14-model comparison table
│   ├── confusion_matrices.xlsx             # Confusion matrices of top 5 model
│   ├── insights.xlsx                       # 9-sheet insight report
│   ├── top_positive_pairs.csv              # Top positive associations
│   ├── top_negative_pairs.csv              # Top negative associations
│   ├── top_neutral_pairs.csv               # Top neutral associations
│   ├── spice_stats.csv                     # Per-spice statistics
│   ├── disease_stats.csv                   # Per-disease statistics
│   └── time_trends.csv                     # Research growth over time
│
├── visualizations/                         # 11 publication-quality charts
│   ├── 00_master_summary.png               # 4-panel overview
│   ├── 01_sentiment_pie.png                # Sentiment distribution
│   ├── 02_top_positive.png                 # Top positive pairs
│   ├── 03_top_negative.png                 # Top negative pairs
│   ├── 04_top_spices.png                   # Most-studied spices
│   ├── 05_top_diseases.png                 # Most-studied diseases
│   ├── 06_most_beneficial.png              # Highest positive-% spices
│   ├── 07_disease_categories.png           # MeSH category analysis
│   ├── 08_time_trends.png                  # Research growth over time
│   ├── 09_heatmap.png                      # Spice × Category heatmap
│   └── 10_confidence.png                   # Model confidence distribution
│
└── phase3_v2_summary.txt                   # Prediction run summary
└── pubtator3_diagram.txt                   # Pipeline architecture diagram
```

---

## 🧪 Methodology

### 1. Data Collection (`scripts/pubmed_search.py`)
- Queried PubMed for **164 spice terms** across biomedical abstracts
- Collected **47,858 unique papers**

### 2. Entity Recognition (`scripts/disease_ner.py`)
- Used **PubTator3 API** to annotate disease mentions
- Produced `disease_annotations.csv`

### 3. Association Mining (`scripts/association_mining.py`)
- Paired co-occurring spice + disease mentions
- Classified diseases by **MeSH tree number** (categories: Neoplasms, Cardiovascular, etc.)
- Output: `spice_disease_associations.csv` (250,255 rows)

### 4. Manual Annotation
- **4,500 rows** manually labeled (`annotation_cleaned.xlsx`)

### 5. Model Training (`notebooks/MultiSeed_Ensemble_Colab.ipynb`)
- Base: **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
- Fine-tuned with **weighted CrossEntropy** for class imbalance
- **5-fold Cross-Validation** for rigorous evaluation
- **Multi-Seed Ensemble** — 5 random seeds, soft voting
- **90.00% accuracy** on held-out test set

### 6. Full Prediction (`notebooks/Phase3_v2_Predict_250k_Kaggle.ipynb`)
- Predicted labels for all 228,750 associations

### 7. Analysis & Visualization
- `phase4_analysis.py` → `results/insights.xlsx` (9 sheets)
- `phase5_visualizations.py` → 11 publication-quality PNG charts

### 8. Evidence grading — EvidenceRx (`evidencerx/`)
The classifier above says *whether* an association is beneficial. It does not
say whether the claim rests on a randomised trial or a petri dish. EvidenceRx
grades every paper by study design and recomputes the headline numbers.

| | % of associations reported beneficial |
|---|---|
| All on-topic evidence | 63.1% |
| Animal + lab only | 74.9% |
| **Human evidence only** | **28.2%** |

A second PubMedBERT model assigns one of seven evidence tiers (off-topic,
synthesis, RCT, human-observational, animal, in-vitro, review) at **86.4%
accuracy** (95% CI ±5.5 pp) on a frozen 147-paper human-labelled test set.
Inter-annotator agreement κ = 0.903; a 45-pair manual audit found 0/45 tier
errors and 76% of pairs usable as reported.

The beneficial-spice ranking reorders completely: **8 of the raw top-15 drop
out** once human evidence is required — they are famous only from cell culture.

See [`evidencerx/README.md`](evidencerx/README.md) for the tier scheme, the
ablation (weak supervision helped at 706 gold labels and *hurt* at 1,045), the
reproduction order, and the stated limitations.

---

## 📥 Large Files Not Included

To stay within GitHub's 100MB per-file hard limit, one file is excluded:

| File | Size | How to obtain |
|------|------|---------------|
| `pubmedbert_final.pt` | 418 MB | Retrain by running `notebooks/MultiSeed_Ensemble_Colab.ipynb` on Colab (free T4 GPU, ~45 min) |
| `final_database_v3.csv` | 93 MB | Run `evidencerx/scripts/evidence_build_v3.py` |
| `e3_predict.csv`, `e3_weak.csv` | 170 MB | Run `evidencerx/scripts/evidence_prep_e3_data.py` |

**The web app does NOT need any of these to run.** It reads predictions from
`final_database_v2.csv`, and merges evidence tiers from `evidence_tiers.csv` —
a 1.3 MB PubMed-ID → tier lookup that carries the same information as the 93 MB
tiered database at 1.4% of the size.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Streamlit 1.28+
- Plotly 5.15+
- Pandas, NumPy, scikit-learn

See [`requirements.txt`](requirements.txt) for the full list with pinned versions.

---

## 🎓 Academic Background

Built on top of and compared against:

- **Rakhi et al. (2018)** — [SpiceRx CNN](https://journals.plos.org/plosone/) (PLOS ONE): 86.70% baseline
- **Microsoft PubMedBERT** — biomedical transformer language model
- **BioBERT** — alternative biomedical transformer
- **Multi-Seed Ensembling** — classic technique for reducing variance

---

## 👤 Author

**Manish Kumar**
**Guide: Dr. Ganesh Bagler and Madhvi Kumari Sinha**
Research Intern, Complex Systems Laboratory (CoSyLab)
Indraprastha Institute of Information Technology, Delhi (IIIT-Delhi)

Supervised by **Prof. Ganesh Bagler**
- 🎓 [Google Scholar](https://scholar.google.com/citations?user=qyth_0QAAAAJ)
- 💼 [LinkedIn](https://in.linkedin.com/in/ganeshbagler)
- 🌐 [CoSyLab Website](https://cosylab.iiitd.edu.in/)

---

## 🙏 Acknowledgments

- **Microsoft** for the PubMedBERT model
- **Rakhi et al.** for the SpiceRx CNN methodology and dataset seed
- **NCBI / PubMed** for open biomedical literature access
- **PubTator3** for disease entity annotations
- **Kaggle / Google Colab** for free GPU compute
- **CoSyLab, IIIT-Delhi** for research infrastructure and mentorship

---

## 📝 License

This work is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)** — free for academic and non-commercial use with attribution.

---
