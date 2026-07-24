"""
5-Fold Cross Validation with SMOTE + Hyperparameter Tuning
Dataset: annotation_4000_final.xlsx
Method: SMOTE for class balancing + GridSearchCV for best parameters
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  SMOTE + Hyperparameter Tuning — 5-Fold Cross Validation")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────────────
print("\n[Step 1] Loading data...")
df = pd.read_excel("annotation_4000_final.xlsx")
df = df.dropna(subset=['Info (English)', 'Association Nature'])
df = df[df['Association Nature'].str.strip() != '']
df['text']  = df['Info (English)'].astype(str).str.strip()
df['label'] = df['Association Nature'].astype(str).str.strip().str.lower()

print(f"  Valid rows: {len(df)}")
for label, count in df['label'].value_counts().items():
    print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

X = df['text'].values
y = df['label'].values

# ── Grid Search Parameters ────────────────────────────────────────────
param_grid = {
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range':  [(1, 1), (1, 2), (1, 3)],
    'tfidf__min_df':       [1, 2],
    'svm__C':              [0.1, 1, 10, 100],
}

print(f"\n[Step 2] Grid Search Parameter Space:")
total_combos = 2 * 3 * 2 * 4
print(f"  max_features: [10000, 20000]")
print(f"  ngram_range:  [(1,1), (1,2), (1,3)]")
print(f"  min_df:       [1, 2]")
print(f"  C:            [0.1, 1, 10, 100]")
print(f"  Total combinations: {total_combos}")

# ── 5-Fold Cross Validation with SMOTE ───────────────────────────────
print(f"\n[Step 3] Running 5-Fold CV with SMOTE + Grid Search...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc=[];fold_prec=[];fold_rec=[];fold_f1=[]
all_true=[];all_pred=[]
best_params_list = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n  --- Fold {fold}/5 ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    from collections import Counter
    print(f"  Train distribution: {dict(Counter(y_train))}")

    # Step 1: TF-IDF on train set first (needed for SMOTE)
    # Use inner cross validation for grid search
    inner_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('svm',   LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)),
    ])

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        inner_pipeline,
        param_grid,
        cv=inner_cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0,
    )

    print(f"  Running Grid Search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"  Best params: C={best_params['svm__C']}, "
          f"ngram={best_params['tfidf__ngram_range']}, "
          f"max_feat={best_params['tfidf__max_features']}, "
          f"min_df={best_params['tfidf__min_df']}")
    print(f"  Best CV score: {grid_search.best_score_*100:.2f}%")

    # Step 2: Apply SMOTE with best TF-IDF params
    best_tfidf = TfidfVectorizer(
        max_features=best_params['tfidf__max_features'],
        ngram_range=best_params['tfidf__ngram_range'],
        min_df=best_params['tfidf__min_df'],
        sublinear_tf=True,
    )
    X_train_tfidf = best_tfidf.fit_transform(X_train)
    X_test_tfidf  = best_tfidf.transform(X_test)

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42, k_neighbors=min(4, min(Counter(y_train).values())-1))
    X_train_sm, y_train_sm = smote.fit_resample(X_train_tfidf, y_train)
    print(f"  After SMOTE: {dict(Counter(y_train_sm))}")

    # Step 3: Train best SVM on SMOTE-balanced data
    best_svm = LinearSVC(
        C=best_params['svm__C'],
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
    )
    best_svm.fit(X_train_sm, y_train_sm)
    y_pred = best_svm.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    fold_acc.append(acc); fold_prec.append(prec)
    fold_rec.append(rec); fold_f1.append(f1)
    all_true.extend(y_test); all_pred.extend(y_pred)

    print(f"  Fold {fold} Result: Acc={acc*100:.1f}%  Prec={prec*100:.1f}%  "
          f"Rec={rec*100:.1f}%  F1={f1*100:.1f}%")

# ── Final Results ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  FINAL RESULTS — SMOTE + Hyperparameter Tuning")
print(f"{'='*65}")

print(f"\n  Fold-wise Accuracy:")
for i, acc in enumerate(fold_acc, 1):
    print(f"    Fold {i}: {acc*100:.2f}%")

print(f"\n  Average Results:")
print(f"    Accuracy:  {np.mean(fold_acc)*100:.2f}% ± {np.std(fold_acc)*100:.2f}%")
print(f"    Precision: {np.mean(fold_prec)*100:.2f}%")
print(f"    Recall:    {np.mean(fold_rec)*100:.2f}%")
print(f"    F1 Score:  {np.mean(fold_f1)*100:.2f}%")

print(f"\n  Per-Class Report:")
print(classification_report(all_true, all_pred, digits=3, zero_division=0))

cm = confusion_matrix(all_true, all_pred, labels=['positive','negative','neutral'])
print(f"  Confusion Matrix:")
print(f"               Pred-pos  Pred-neg  Pred-neu")
for i, label in enumerate(['positive','negative','neutral']):
    print(f"  True-{label:<10} {cm[i][0]:>8}  {cm[i][1]:>8}  {cm[i][2]:>8}")

print(f"\n  Most common best params across folds:")
from collections import Counter
c_vals = Counter(p['svm__C'] for p in best_params_list)
ng_vals = Counter(p['tfidf__ngram_range'] for p in best_params_list)
print(f"    C: {dict(c_vals)}")
print(f"    ngram_range: {dict(ng_vals)}")

print(f"\n  Comparison:")
print(f"    Before (SVM no SMOTE):        74.27%")
print(f"    After  (SMOTE + Grid Search): {np.mean(fold_acc)*100:.2f}%")
print(f"    Improvement:                  {(np.mean(fold_acc)-0.7427)*100:+.2f}%")
print(f"{'='*65}")
