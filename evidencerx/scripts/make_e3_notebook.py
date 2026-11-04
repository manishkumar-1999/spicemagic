"""
Generates Evidence_TierClassifier_Kaggle.ipynb -- the evidence-tier classifier
(PubMedBERT, 7 classes, two-stage weak->gold with ablation).
Run locally to emit the notebook; upload e3_*.csv as a Kaggle dataset, run on GPU.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# EvidenceRx -- Evidence-Tier Classifier (PubMedBERT, two-stage)\n\n"
    "Predicts an evidence tier for every paper from **title + abstract**.\n\n"
    "**7 classes (index = tier):** 0 off-topic, 1 synthesis, 2 rct, "
    "3 human-observational, 4 animal, 5 in-vitro, 6 review.\n\n"
    "**Two-stage (weak-supervision):** train on ~40k noisy rule labels "
    "(tiers 1-6), then fine-tune on the clean gold set (tiers 0-6, incl "
    "off-topic). We report an **ablation** -- weak-only vs gold-only vs "
    "weak->gold -- all judged on the held-out gold test split (the honest "
    "metric).\n\n"
    "Upload `e3_weak.csv`, `e3_gold_train.csv`, `e3_gold_val.csv`, "
    "`e3_gold_test.csv`, `e3_predict.csv` as a Kaggle dataset; set `INPUT_DIR`; "
    "enable GPU."
))

c.append(nbf.v4.new_code_cell(
    "import os, numpy as np, pandas as pd, torch, random\n"
    "import torch.nn as nn\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from transformers import (AutoTokenizer,\n"
    "                          AutoModelForSequenceClassification, set_seed)\n"
    "from sklearn.metrics import (accuracy_score, f1_score,\n"
    "                             classification_report, confusion_matrix)\n\n"
    "import glob\n"
    "# auto-detect the dataset folder (works whatever you named it)\n"
    "_hits = glob.glob('/kaggle/input/**/e3_weak.csv', recursive=True)\n"
    "assert _hits, 'e3_weak.csv not found under /kaggle/input -- click Add Input and attach your dataset'\n"
    "INPUT_DIR = os.path.dirname(_hits[0])\n"
    "print('INPUT_DIR =', INPUT_DIR)\n"
    "MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'\n"
    "CLASSES = ['off-topic','synthesis','rct','human-observational',\n"
    "           'animal','in-vitro','review']   # index == tier\n"
    "NUM = len(CLASSES)\n"
    "MAX_LEN = 384\n"
    "BATCH = 16\n"
    "WEAK_EPOCHS = 3\n"
    "GOLD_EPOCHS = 6\n"
    "LR = 2e-5\n"
    "FT_LR = 1e-5           # gentler LR for the gold fine-tune\n"
    "SEED = 42\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "os.environ['PYTHONHASHSEED'] = str(SEED)\n"
    "set_seed(SEED)\n"
    "g = torch.Generator().manual_seed(SEED)\n"
    "torch.backends.cudnn.deterministic = True\n"
    "torch.backends.cudnn.benchmark = False\n"
    "print('device:', device, '| classes:', NUM)"
))

c.append(nbf.v4.new_code_cell(
    "weak = pd.read_csv(f'{INPUT_DIR}/e3_weak.csv')\n"
    "gtr  = pd.read_csv(f'{INPUT_DIR}/e3_gold_train.csv')\n"
    "gva  = pd.read_csv(f'{INPUT_DIR}/e3_gold_val.csv')\n"
    "gte  = pd.read_csv(f'{INPUT_DIR}/e3_gold_test.csv')\n"
    "pred = pd.read_csv(f'{INPUT_DIR}/e3_predict.csv')\n"
    "for d in (weak, gtr, gva, gte, pred):\n"
    "    d['text'] = d['text'].fillna('').astype(str)\n"
    "print('weak', len(weak), '| gold tr/va/te', len(gtr), len(gva), len(gte),\n"
    "      '| predict', len(pred))\n"
    "print('weak label balance :', weak['label'].value_counts().sort_index().to_dict())\n"
    "print('gold train balance :', gtr['label'].value_counts().sort_index().to_dict())"
))

c.append(nbf.v4.new_code_cell(
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n\n"
    "class DS(Dataset):\n"
    "    def __init__(self, texts, labels=None):\n"
    "        self.texts=list(texts); self.labels=list(labels) if labels is not None else None\n"
    "    def __len__(self): return len(self.texts)\n"
    "    def __getitem__(self, i):\n"
    "        enc = tokenizer(self.texts[i], truncation=True, padding='max_length',\n"
    "                        max_length=MAX_LEN, return_tensors='pt')\n"
    "        item = {k: v.squeeze(0) for k, v in enc.items()}\n"
    "        if self.labels is not None:\n"
    "            item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)\n"
    "        return item\n\n"
    "def loader(df, shuffle=False):\n"
    "    ds = DS(df['text'], df['label'] if 'label' in df else None)\n"
    "    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,\n"
    "                      generator=g if shuffle else None)\n\n"
    "gold_test_dl = loader(gte)"
))

c.append(nbf.v4.new_code_cell(
    "def class_weights(labels):\n"
    "    counts = np.bincount(labels, minlength=NUM).astype(float)\n"
    "    w = counts.sum() / (NUM * np.clip(counts, 1, None))\n"
    "    w[counts == 0] = 1.0            # classes absent in this stage\n"
    "    return torch.tensor(w, dtype=torch.float).to(device)\n\n"
    "def new_model():\n"
    "    return AutoModelForSequenceClassification.from_pretrained(\n"
    "        MODEL, num_labels=NUM).to(device)\n\n"
    "@torch.no_grad()\n"
    "def evaluate(model, dl, with_report=False):\n"
    "    model.eval(); P, G = [], []\n"
    "    for b in dl:\n"
    "        y = b.pop('labels').to(device)\n"
    "        b = {k: v.to(device) for k, v in b.items()}\n"
    "        P += model(**b).logits.argmax(1).cpu().tolist(); G += y.cpu().tolist()\n"
    "    acc = accuracy_score(G, P); f1 = f1_score(G, P, average='macro')\n"
    "    if with_report:\n"
    "        present = sorted(set(G) | set(P))\n"
    "        print(classification_report(G, P, labels=present,\n"
    "              target_names=[CLASSES[i] for i in present], digits=3, zero_division=0))\n"
    "        print('confusion (rows=true, cols=pred):')\n"
    "        print(pd.DataFrame(confusion_matrix(G, P, labels=list(range(NUM))),\n"
    "              index=CLASSES, columns=CLASSES))\n"
    "    return acc, f1\n\n"
    "def train(model, df, epochs, lr, tag):\n"
    "    dl = loader(df, shuffle=True)\n"
    "    loss_fn = nn.CrossEntropyLoss(weight=class_weights(df['label'].values))\n"
    "    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)\n"
    "    for ep in range(epochs):\n"
    "        model.train(); tot=0\n"
    "        for b in dl:\n"
    "            y = b.pop('labels').to(device)\n"
    "            b = {k: v.to(device) for k, v in b.items()}\n"
    "            opt.zero_grad(); loss = loss_fn(model(**b).logits, y)\n"
    "            loss.backward(); opt.step(); tot += loss.item()\n"
    "        acc, f1 = evaluate(model, gold_test_dl)\n"
    "        print(f'  [{tag}] epoch {ep+1}/{epochs} loss={tot/len(dl):.3f} '\n"
    "              f'gold-test acc={acc:.3f} macroF1={f1:.3f}')\n"
    "    return model"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Stage 1: WEAK-only (tiers 1-6; off-topic unseen here) =====\n"
    "print('--- Stage 1: train on weak labels ---')\n"
    "m_weak = train(new_model(), weak, WEAK_EPOCHS, LR, 'weak')\n"
    "weak_acc, weak_f1 = evaluate(m_weak, gold_test_dl)"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Ablation: GOLD-only (from scratch, gold train split only) =====\n"
    "print(f'--- Ablation: gold-only ({len(gtr)} rows) ---')\n"
    "m_gold = train(new_model(), gtr, GOLD_EPOCHS, LR, 'gold-only')\n"
    "gold_acc, gold_f1 = evaluate(m_gold, gold_test_dl)"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Stage 2: WEAK -> GOLD fine-tune (the main model) =====\n"
    "print('--- Stage 2: fine-tune weak model on gold ---')\n"
    "m_final = train(m_weak, gtr, GOLD_EPOCHS, FT_LR, 'weak->gold')\n"
    "final_acc, final_f1 = evaluate(m_final, gold_test_dl)"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Ablation table + final report on gold test =====\n"
    "print('\\n===== ABLATION (gold test) =====')\n"
    "print(pd.DataFrame({\n"
    "    'model': ['weak-only','gold-only','weak->gold'],\n"
    "    'accuracy': [weak_acc, gold_acc, final_acc],\n"
    "    'macro_F1': [weak_f1, gold_f1, final_f1]}).round(3).to_string(index=False))\n"
    "print('\\n===== FINAL MODEL (weak->gold) per-class report =====')\n"
    "evaluate(m_final, gold_test_dl, with_report=True)"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Full inference: tier for every paper =====\n"
    "@torch.no_grad()\n"
    "def predict_probs(model, df):\n"
    "    model.eval(); out=[]\n"
    "    dl = DataLoader(DS(df['text']), batch_size=BATCH)\n"
    "    for b in dl:\n"
    "        b = {k: v.to(device) for k, v in b.items()}\n"
    "        out.append(torch.softmax(model(**b).logits, 1).cpu().numpy())\n"
    "    return np.concatenate(out)\n\n"
    "Pr = predict_probs(m_final, pred)\n"
    "res = pred[['pmid']].copy()\n"
    "for i, cname in enumerate(CLASSES):\n"
    "    res[f'p_{i}_{cname}'] = Pr[:, i]\n"
    "res['pred_tier'] = Pr.argmax(1)\n"
    "res['pred_class'] = [CLASSES[i] for i in Pr.argmax(1)]\n"
    "res['confidence'] = Pr.max(1)\n"
    "res.to_csv('e3_tier_predictions.csv', index=False)\n"
    "print('wrote e3_tier_predictions.csv', res.shape)\n"
    "print(res['pred_class'].value_counts())\n"
    "print('mean confidence:', round(res['confidence'].mean(), 3))"
))

c.append(nbf.v4.new_code_cell(
    "m_final.save_pretrained('evidence_tier_pubmedbert')\n"
    "tokenizer.save_pretrained('evidence_tier_pubmedbert')\n"
    "print('saved model -> evidence_tier_pubmedbert/')"
))

c.append(nbf.v4.new_markdown_cell(
    "## After running\n"
    "Download `e3_tier_predictions.csv` into the project folder, then locally:\n"
    "```\n"
    "python evidence_build_v3.py   # attach tier + off-topic flag -> final_database_v3.csv\n"
    "python evidence_aggregate.py  # tier-corrected re-analysis (Phase 6)\n"
    "```"
))

nb["cells"] = c
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}}
with open("Evidence_TierClassifier_Kaggle.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote Evidence_TierClassifier_Kaggle.ipynb with", len(c), "cells")
