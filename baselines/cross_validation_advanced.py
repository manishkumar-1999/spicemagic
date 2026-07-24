"""
Advanced 5-Fold Cross Validation
Methods: XGBoost + Ensemble (SVM + LR + XGBoost voting)
Dataset: annotation_4000_final.xlsx
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

print("=" * 65)
print("  Advanced ML — XGBoost + Ensemble Voting")
print("  5-Fold Cross Validation on annotation_4000_final.xlsx")
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

# Encode labels for XGBoost
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── Classifiers ────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_results = {}

# ── 1. XGBoost ────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Classifier: XGBoost + TF-IDF")
print(f"{'─'*65}")

fold_acc=[];fold_prec=[];fold_rec=[];fold_f1=[]
all_true=[];all_pred=[]

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_enc = le.transform(y_train)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                             sublinear_tf=True, min_df=2)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    # Class weights for imbalance
    counts = Counter(y_train)
    total = len(y_train)
    scale = {le.transform([k])[0]: total/(3*v) for k,v in counts.items()}

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1,
    )
    xgb.fit(X_tr, y_train_enc)
    y_pred_enc = xgb.predict(X_te)
    y_pred = le.inverse_transform(y_pred_enc)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    fold_acc.append(acc); fold_prec.append(prec)
    fold_rec.append(rec); fold_f1.append(f1)
    all_true.extend(y_test); all_pred.extend(y_pred)

    print(f"  Fold {fold}: Acc={acc*100:.1f}%  Prec={prec*100:.1f}%  "
          f"Rec={rec*100:.1f}%  F1={f1*100:.1f}%")

print(f"\n  Average:")
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

all_results['XGBoost'] = {
    'accuracy': np.mean(fold_acc), 'precision': np.mean(fold_prec),
    'recall': np.mean(fold_rec), 'f1': np.mean(fold_f1)
}

# ── 2. Ensemble Voting (SVM + LR + XGBoost) ──────────────────────────
print(f"\n{'─'*65}")
print(f"  Classifier: Ensemble Voting (SVM + LR + XGBoost)")
print(f"{'─'*65}")

fold_acc=[];fold_prec=[];fold_rec=[];fold_f1=[]
all_true=[];all_pred=[]

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_enc = le.transform(y_train)

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                             sublinear_tf=True, min_df=2)
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)

    # Individual classifiers
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    )
    lr  = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1)

    # Train SVM and LR on string labels
    svm.fit(X_tr, y_train)
    lr.fit(X_tr, y_train)

    # Train XGBoost on encoded labels
    xgb.fit(X_tr, y_train_enc)

    # Get probabilities from each
    svm_prob = svm.predict_proba(X_te)   # shape (n, 3)
    lr_prob  = lr.predict_proba(X_te)    # shape (n, 3)
    xgb_prob = xgb.predict_proba(X_te)  # shape (n, 3) — encoded order

    # Align XGBoost class order with SVM/LR class order
    svm_classes = list(svm.classes_)
    xgb_classes = list(le.classes_)

    # Reorder xgb_prob to match svm_classes order
    xgb_prob_aligned = np.zeros_like(xgb_prob)
    for i, cls in enumerate(svm_classes):
        xgb_idx = xgb_classes.index(cls)
        xgb_prob_aligned[:, i] = xgb_prob[:, xgb_idx]

    # Average probabilities (soft voting)
    avg_prob = (svm_prob + lr_prob + xgb_prob_aligned) / 3
    y_pred_idx = np.argmax(avg_prob, axis=1)
    y_pred = np.array(svm_classes)[y_pred_idx]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    fold_acc.append(acc); fold_prec.append(prec)
    fold_rec.append(rec); fold_f1.append(f1)
    all_true.extend(y_test); all_pred.extend(y_pred)

    print(f"  Fold {fold}: Acc={acc*100:.1f}%  Prec={prec*100:.1f}%  "
          f"Rec={rec*100:.1f}%  F1={f1*100:.1f}%")

print(f"\n  Average:")
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

all_results['Ensemble (SVM+LR+XGB)'] = {
    'accuracy': np.mean(fold_acc), 'precision': np.mean(fold_prec),
    'recall': np.mean(fold_rec), 'f1': np.mean(fold_f1)
}

# ── Final Comparison ───────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  COMPLETE COMPARISON — ALL METHODS")
print(f"{'='*65}")

all_results['SVM (previous best)'] = {'accuracy': 0.7427, 'precision': 0.7380, 'recall': 0.7427, 'f1': 0.7388}
all_results['SMOTE+GridSearch']    = {'accuracy': 0.7377, 'precision': 0.7406, 'recall': 0.7377, 'f1': 0.7373}

print(f"  {'Method':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

best = max(all_results, key=lambda k: all_results[k]['f1'])
for name, res in all_results.items():
    marker = ' <-- BEST' if name == best else ''
    print(f"  {name:<28} {res['accuracy']*100:>9.2f}% {res['precision']*100:>9.2f}% "
          f"{res['recall']*100:>9.2f}% {res['f1']*100:>9.2f}%{marker}")

print(f"\n  Best Method: {best}")
print(f"  Best Accuracy: {all_results[best]['accuracy']*100:.2f}%")
print(f"{'='*65}")
