"""
5-Fold Cross Validation for Spice-Disease Association Classification
Dataset: annotation_4000.xlsx (4000 rows, 800 per fold)
Features: TF-IDF on 'Info (English)' column
Classifiers: Logistic Regression, SVM, Random Forest
Labels: positive, negative, neutral
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from collections import defaultdict


# ── Load data ─────────────────────────────────────────────────────────
print("=" * 65)
print("  5-Fold Cross Validation — Spice-Disease Classification")
print("=" * 65)

print("\n[Step 1] Loading data...")
df = pd.read_excel("annotation_4000.xlsx")
print(f"  Loaded {len(df)} rows")
print(f"  Columns: {list(df.columns)}")

# Use Info (English) as feature, Association Nature as label
df = df.dropna(subset=['Info (English)', 'Association Nature'])
df = df[df['Association Nature'].str.strip() != '']
df['text'] = df['Info (English)'].astype(str).str.strip()
df['label'] = df['Association Nature'].astype(str).str.strip().str.lower()

print(f"  Valid rows: {len(df)}")
print(f"  Label distribution:")
for label, count in df['label'].value_counts().items():
    print(f"    {label:<12}: {count:>5} ({count/len(df)*100:.1f}%)")

X = df['text'].values
y = df['label'].values

# ── Classifiers ────────────────────────────────────────────────────────
classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', multi_class='multinomial', random_state=42
    ),
    "SVM (LinearSVC)": LinearSVC(
        C=1.0, max_iter=2000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
}

# ── 5-Fold Cross Validation ────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n[Step 2] Running 5-Fold Stratified Cross Validation...")
print(f"  Folds: 5  |  Train per fold: {int(len(df)*0.8)}  |  Test per fold: {int(len(df)*0.2)}")

all_results = {}

for clf_name, clf in classifiers.items():
    print(f"\n{'─'*65}")
    print(f"  Classifier: {clf_name}")
    print(f"{'─'*65}")

    fold_accuracies = []
    fold_precisions = []
    fold_recalls    = []
    fold_f1s        = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # TF-IDF vectorization (fit only on train to avoid data leakage)
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf  = tfidf.transform(X_test)

        # Train
        clf.fit(X_train_tfidf, y_train)

        # Predict
        y_pred = clf.predict(X_test_tfidf)

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold}: Acc={acc*100:.1f}%  Prec={prec*100:.1f}%  "
              f"Rec={rec*100:.1f}%  F1={f1*100:.1f}%  "
              f"(train={len(X_train)}, test={len(X_test)})")

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  Average Results:")
    print(f"    Accuracy:  {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")
    print(f"    Precision: {np.mean(fold_precisions)*100:.2f}% ± {np.std(fold_precisions)*100:.2f}%")
    print(f"    Recall:    {np.mean(fold_recalls)*100:.2f}% ± {np.std(fold_recalls)*100:.2f}%")
    print(f"    F1 Score:  {np.mean(fold_f1s)*100:.2f}% ± {np.std(fold_f1s)*100:.2f}%")

    # Per-class report
    print(f"\n  Per-Class Report (all folds combined):")
    print(classification_report(all_y_true, all_y_pred,
                                 target_names=['negative','neutral','positive'],
                                 digits=3, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=['positive','negative','neutral'])
    print(f"  Confusion Matrix (positive / negative / neutral):")
    print(f"               Pred-pos  Pred-neg  Pred-neu")
    for i, label in enumerate(['positive','negative','neutral']):
        print(f"  True-{label:<10} {cm[i][0]:>8}  {cm[i][1]:>8}  {cm[i][2]:>8}")

    all_results[clf_name] = {
        'accuracy': np.mean(fold_accuracies),
        'precision': np.mean(fold_precisions),
        'recall': np.mean(fold_recalls),
        'f1': np.mean(fold_f1s),
        'std': np.std(fold_accuracies),
    }

# ── Final Comparison ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL COMPARISON — ALL CLASSIFIERS")
print(f"{'='*65}")
print(f"  {'Classifier':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
best_clf = max(all_results, key=lambda k: all_results[k]['f1'])
for name, res in all_results.items():
    marker = ' <-- BEST' if name == best_clf else ''
    print(f"  {name:<25} {res['accuracy']*100:>9.2f}% {res['precision']*100:>9.2f}% "
          f"{res['recall']*100:>9.2f}% {res['f1']*100:>9.2f}%{marker}")

print(f"\n  Best Classifier: {best_clf}")
print(f"  Best F1 Score:   {all_results[best_clf]['f1']*100:.2f}%")
print(f"{'='*65}")
