# EvidenceRx — evidence grading for the SpiceMagic database

SpiceMagic mines 228,750 spice–disease associations from PubMed and labels each
one positive, neutral or negative. It does not distinguish a meta-analysis of
randomised trials from a result in a petri dish. EvidenceRx adds that
distinction: every paper is graded by study design, and the headline numbers
are recomputed with the grade taken into account.

The short version of what it found:

> The raw literature reports a **beneficial** association **63.1%** of the time.
> Restricted to human evidence, that falls to **28.2%** — a drop of 34.8
> percentage points. Animal and lab studies alone report benefit **74.9%** of
> the time.

Same spices, same diseases, same papers. The answer more than halves once you
require human evidence.

---

## The tier scheme

| tier | class | what it is |
|------|-------|------------|
| 0 | off-topic | not about a spice health effect at all — mining noise |
| 1 | synthesis | meta-analysis or systematic review of human trials |
| 2 | rct | randomised controlled trial in humans |
| 3 | human-observational | cohort, case-control, cross-sectional, case report |
| 4 | animal | in-vivo animal study |
| 5 | in-vitro | cell line, biochemical or microbiology assay |
| 6 | review | narrative review, editorial, or in-silico only |

Tier 0 is a **relevance** judgement; tiers 1–6 are **study-design** judgements.
They are deliberately in one classifier because the mining produces both kinds
of error, but they are not the same question — see `docs/OFFTOPIC_RULES.md`.

---

## Results

**Evidence-tier classifier** (PubMedBERT, 7 classes, title + abstract), measured
on a frozen 147-paper human-labelled test set:

| model | test accuracy | macro-F1 |
|-------|--------------|----------|
| rule baseline (no ML) | 78.5% precision* | — |
| weak→gold, 706 gold labels | 80.3% | 0.796 |
| weak→gold, 1,045 gold labels | 83.0% | 0.828 |
| **gold-only, 3-seed ensemble** | **86.4%** | **0.865** |
| gold-only + PubMed metadata, 5 seeds | 84.4% | 0.834 |

\* on-topic precision of the hand-written rules, not an accuracy.

**86.4% carries a 95% confidence interval of ±5.5 pp → [80.9, 91.9].** At 147
test papers the four rows above are not all statistically separable; the
defensible claim is a measured +6.1 pp improvement with a stated interval.

### The ablation worth reading

| model | test accuracy |
|-------|--------------|
| weak-only (40,158 rule labels, no gold) | 59.9% |
| weak→gold (3-seed ensemble) | 81.0% |
| gold-only (3-seed ensemble) | **87.8%** |

Weak supervision **helped at 706 gold labels and hurt at 1,045.** Gold-only beat
weak→gold by 5.7 pp with complete seed separation (gold-only's worst seed, 0.837,
beat weak→gold's best, 0.810). Pre-training on noisy rule labels is not free once
the clean set is large enough.

*(The ablation table reports 87.8% while the shipped predictions score 86.4%.
Cause: `new_model()` ran before `set_seed()`, so the final inference refit built
different models than the ones scored. Quote 86.4% — that is the model the
published tiers came from. Fixed in the v3 notebook.)*

### Annotation

1,400 gold papers: 1,400 human-labelled. Inter-annotator agreement
measured on an 84-paper blind calibration set: **Cohen's κ = 0.903**.

The test split is **100% human-labelled**. Provenance for every paper is in `results/gold_provenance.csv`.

### Reliability audit

45 spice–disease pairs hand-checked across three strata (preclinical-hype,
most-contested, random control):

| layer checked | error rate |
|---------------|-----------|
| evidence tier assignment | **0 / 45** |
| disease term (from the v2 mining) | 11% |
| association direction (from the v2 sentiment model) | 13% |

76% of pairs were usable as reported (95% CI 61–86). Pairs surfaced by the
controversy analysis were **no less reliable than a random control** (27% vs 20%
unusable), which is what keeps the Phase 6/7 findings defensible.

Two specific defects the audit found, both in pre-existing layers:

- *"Allium vegetables are associated with **reduced** risk of colorectal cancer"*
  is labelled **negative** — the sentiment model conflates an inverse
  association with a harmful one.
- Safflower/breast cancer: safflower oil is the **n-6 control diet** in those
  rodent studies, not the intervention.

---

## Reproducing

Scripts run in this order. Steps 3 and 6 need a GPU (Kaggle notebooks in
`notebooks/`); everything else runs locally in minutes.

```
1  evidence_fetch_metadata.py     PublicationType + MeSH for 49,499 PMIDs
2  evidence_label_studytype.py    rule labels, 6 tiers, 83.5% coverage
3  evidence_gold_sample.py        1,000-paper blind annotation sheet
   evidence_gold_finalize.py      score rules vs gold, build the splits
   evidence_gold_audit.py         quality check on the gold labels
4  evidence_prep_e3_data.py       two-stage 7-class training data
5  notebooks/..._v2.ipynb         train the classifier            [GPU]
6  evidence_round2_sample.py      active-learning round 2
   evidence_round2_apply_labels.py / _audit.py / gold_merge_rounds.py
7  evidence_build_v3.py           attach tiers to the association table
8  evidence_aggregate.py          tier-corrected re-analysis
9  evidence_controversy.py        disagreement + preclinical-hype motif
10 evidence_confounder_audit.py   build the 45-pair audit sheet
   evidence_confounder_score.py   score it
```

`evidence_validate.py` checks the whole chain — split integrity, leakage,
recomputed accuracy, and staleness of every derived artefact. Run it before
trusting any downstream number.

### What is not in this folder

Large regenerable intermediates are excluded: `final_database_v3.csv` (93 MB),
`e3_predict.csv`, `e3_weak.csv`, `e3_tier_predictions.csv`. The app does not
need them — it merges `../evidence_tiers.csv`, a 1.3 MB PubMed-ID → tier
lookup, which carries the same information at 1.4% of the size.

---

## Honest limitations

- **147-paper test set.** One paper is 0.68 pp; seed standard deviation is
  1.7 pp. Differences smaller than ~3 pp are not measurable here.
- **~17% of association rows carry a method word as the "disease"**
  (`cytotoxicity`, `proliferation`, `oxidative stress`). `evidence_controversy.py`
  filters these, but they remain in the underlying database.
