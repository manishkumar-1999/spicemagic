"""
Generates Evidence_TierClassifier_Kaggle_v3.ipynb.

v3 vs v2 -- five changes, each traceable to something the v2 run showed:

  1. WEAK STAGE DROPPED. v2 settled it: gold-only beat weak->gold by 5.7 pp
     on test with complete seed separation (gold-only worst 0.837 >
     weak->gold best 0.810). Pre-training on 40k noisy rule labels is
     actively harmful at 1,045 gold labels. Dropping it also cuts ~1 h.

  2. EARLY STOPPING on val. v2 logs peaked at epoch 4 (val 0.861) and fell
     to 0.837 by epoch 6 -- it shipped an overfit model. Now trains up to
     MAX_EPOCHS and restores the best-val checkpoint.

  3. SEED BUG FIXED. v2 called new_model() before train() ran set_seed(),
     so the classifier head was initialised from arbitrary RNG state. That
     is why the ablation said 0.878 but the shipped predictions scored
     0.864 -- they were different models. set_seed() now runs first.

  4. INFERENCE INSIDE THE SEED LOOP. v2 refitted the winner afterwards,
     which is what let the reported and shipped ensembles diverge. Now the
     model that is scored is the model that predicts. They cannot differ.

  5. LR warmup + linear decay, and 5 seeds instead of 3.

Paired with WITH_METADATA=True in evidence_prep_e3_data.py, so the input now
carries [PUBTYPE] and [MESH] -- PublicationType covers 100% of the corpus and
is near-decisive for tiers 1/2/6, which is where 7 of v2's 20 test errors sat.

Run locally to emit the notebook. v1/v2 notebooks are left untouched.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# EvidenceRx -- Evidence-Tier Classifier v3 (PubMedBERT)\n\n"
    "**7 classes (index = tier):** 0 off-topic, 1 synthesis, 2 rct, "
    "3 human-observational, 4 animal, 5 in-vitro, 6 review.\n\n"
    "### Changes from v2 (test 0.864)\n"
    "| | v2 | v3 |\n|---|---|---|\n"
    "| weak stage | trained, then discarded | **gone** (cost 5.7 pp) |\n"
    "| epochs | fixed 6 | **early stopping on val** |\n"
    "| seeding | head init before set_seed | **fixed** |\n"
    "| inference | refit afterwards | **inside the seed loop** |\n"
    "| LR | constant | **warmup + linear decay** |\n"
    "| seeds | 3 | **5** |\n"
    "| input | title+abstract | **+ [PUBTYPE] + [MESH]** |\n\n"
    "The scored model and the predicting model are now the same object, so "
    "the reported accuracy and the shipped tiers cannot disagree -- that is "
    "what went wrong in v2.\n\n"
    "Attach the dataset with the regenerated `e3_*.csv` (metadata version); "
    "GPU + Internet on. Expect ~50-60 min."
))

c.append(nbf.v4.new_code_cell(
    "import os, glob, copy, numpy as np, pandas as pd, torch\n"
    "import torch.nn as nn\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
    "                          get_linear_schedule_with_warmup, set_seed)\n"
    "from sklearn.metrics import (accuracy_score, f1_score,\n"
    "                             classification_report, confusion_matrix)\n\n"
    "_hits = glob.glob('/kaggle/input/**/e3_gold_train.csv', recursive=True)\n"
    "assert _hits, 'e3_gold_train.csv not found -- click Add Input'\n"
    "INPUT_DIR = os.path.dirname(_hits[0])\n"
    "print('INPUT_DIR =', INPUT_DIR)\n"
    "if len(_hits) > 1:\n"
    "    print('WARNING: found in more than one input; using the first:')\n"
    "    for h in _hits:\n"
    "        print('   ', h)\n\n"
    "MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'\n"
    "CLASSES = ['off-topic','synthesis','rct','human-observational',\n"
    "           'animal','in-vitro','review']   # index == tier\n"
    "NUM = len(CLASSES)\n"
    "MAX_LEN = 512\n"
    "BATCH = 16\n"
    "MAX_EPOCHS = 10          # early stopping decides the real number\n"
    "PATIENCE = 3\n"
    "LR = 2e-5\n"
    "WARMUP_FRAC = 0.1\n"
    "SEEDS = [42, 1337, 2024, 7, 2718]\n"
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print('device:', device, '| seeds:', SEEDS)"
))

c.append(nbf.v4.new_code_cell(
    "gtr  = pd.read_csv(f'{INPUT_DIR}/e3_gold_train.csv')\n"
    "gva  = pd.read_csv(f'{INPUT_DIR}/e3_gold_val.csv')\n"
    "gte  = pd.read_csv(f'{INPUT_DIR}/e3_gold_test.csv')\n"
    "pred = pd.read_csv(f'{INPUT_DIR}/e3_predict.csv')\n"
    "for d in (gtr, gva, gte, pred):\n"
    "    d['text'] = d['text'].fillna('').astype(str)\n"
    "y_val, y_test = gva['label'].values, gte['label'].values\n"
    "print('gold tr/va/te', len(gtr), len(gva), len(gte), '| predict', len(pred))\n"
    "print('train balance:', gtr['label'].value_counts().sort_index().to_dict())\n"
    "has_meta = gtr['text'].str.contains(r'\\[PUBTYPE\\]').mean()\n"
    "print(f'inputs carrying [PUBTYPE]: {100*has_meta:.0f}%')\n"
    "assert has_meta > 0.99, ('These are the OLD title+abstract inputs. '\n"
    "    'Re-run evidence_prep_e3_data.py with WITH_METADATA=True and '\n"
    "    're-upload the dataset.')"
))

c.append(nbf.v4.new_code_cell(
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n\n"
    "class DS(Dataset):\n"
    "    def __init__(self, texts, labels=None):\n"
    "        self.texts = list(texts)\n"
    "        self.labels = list(labels) if labels is not None else None\n"
    "    def __len__(self):\n"
    "        return len(self.texts)\n"
    "    def __getitem__(self, i):\n"
    "        enc = tokenizer(self.texts[i], truncation=True, padding='max_length',\n"
    "                        max_length=MAX_LEN, return_tensors='pt')\n"
    "        item = {k: v.squeeze(0) for k, v in enc.items()}\n"
    "        if self.labels is not None:\n"
    "            item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)\n"
    "        return item\n\n"
    "def class_weights(labels):\n"
    "    counts = np.bincount(labels, minlength=NUM).astype(float)\n"
    "    w = counts.sum() / (NUM * np.clip(counts, 1, None))\n"
    "    w[counts == 0] = 1.0\n"
    "    return torch.tensor(w, dtype=torch.float).to(device)\n\n"
    "@torch.no_grad()\n"
    "def probs_on(model, df):\n"
    "    model.eval()\n"
    "    out = []\n"
    "    for b in DataLoader(DS(df['text']), batch_size=BATCH):\n"
    "        b = {k: v.to(device) for k, v in b.items()}\n"
    "        out.append(torch.softmax(model(**b).logits, 1).cpu().numpy())\n"
    "    return np.concatenate(out)\n\n"
    "def score(P, y):\n"
    "    yh = P.argmax(1)\n"
    "    return accuracy_score(y, yh), f1_score(y, yh, average='macro')\n\n"
    "torch.backends.cudnn.deterministic = True\n"
    "torch.backends.cudnn.benchmark = False"
))

c.append(nbf.v4.new_code_cell(
    "def fit_one(seed):\n"
    "    \"\"\"Train one gold-only model with early stopping on val.\n"
    "    set_seed BEFORE from_pretrained -- otherwise the classifier head is\n"
    "    initialised from whatever RNG state happens to be current, which is\n"
    "    the v2 bug that made the ablation and the shipped model disagree.\"\"\"\n"
    "    set_seed(seed)\n"
    "    model = AutoModelForSequenceClassification.from_pretrained(\n"
    "        MODEL, num_labels=NUM).to(device)\n"
    "    g = torch.Generator().manual_seed(seed)\n"
    "    dl = DataLoader(DS(gtr['text'], gtr['label']), batch_size=BATCH,\n"
    "                    shuffle=True, generator=g)\n"
    "    loss_fn = nn.CrossEntropyLoss(weight=class_weights(gtr['label'].values))\n"
    "    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)\n"
    "    total = len(dl) * MAX_EPOCHS\n"
    "    sched = get_linear_schedule_with_warmup(\n"
    "        opt, int(total * WARMUP_FRAC), total)\n\n"
    "    best_acc, best_ep, best_state, bad = -1.0, 0, None, 0\n"
    "    for ep in range(1, MAX_EPOCHS + 1):\n"
    "        model.train()\n"
    "        tot = 0.0\n"
    "        for b in dl:\n"
    "            y = b.pop('labels').to(device)\n"
    "            b = {k: v.to(device) for k, v in b.items()}\n"
    "            opt.zero_grad()\n"
    "            loss = loss_fn(model(**b).logits, y)\n"
    "            loss.backward()\n"
    "            opt.step()\n"
    "            sched.step()\n"
    "            tot += loss.item()\n"
    "        va, vf = score(probs_on(model, gva), y_val)\n"
    "        flag = ''\n"
    "        if va > best_acc:\n"
    "            best_acc, best_ep, bad = va, ep, 0\n"
    "            best_state = {k: v.detach().cpu().clone()\n"
    "                          for k, v in model.state_dict().items()}\n"
    "            flag = '  <-- best'\n"
    "        else:\n"
    "            bad += 1\n"
    "        print(f'  [s{seed}] ep {ep}/{MAX_EPOCHS} loss={tot/len(dl):.3f} '\n"
    "              f'val_acc={va:.3f} val_F1={vf:.3f}{flag}')\n"
    "        if bad >= PATIENCE:\n"
    "            print(f'  [s{seed}] early stop (no val gain in {PATIENCE})')\n"
    "            break\n"
    "    model.load_state_dict(best_state)      # restore the best epoch\n"
    "    print(f'  [s{seed}] restored epoch {best_ep} (val_acc={best_acc:.3f})')\n"
    "    return model, best_ep, best_acc"
))

c.append(nbf.v4.new_code_cell(
    "# ===== 5 seeds. Each model is scored AND used for inference before it\n"
    "# is freed, so the reported ensemble is exactly the shipped ensemble. =====\n"
    "Pv, Pt, Pc, rows = [], [], np.zeros((len(pred), NUM)), []\n"
    "for s in SEEDS:\n"
    "    print(f'\\n===== seed {s} =====')\n"
    "    m, ep, bacc = fit_one(s)\n"
    "    pv, pt = probs_on(m, gva), probs_on(m, gte)\n"
    "    Pv.append(pv); Pt.append(pt)\n"
    "    print('  running corpus inference ...')\n"
    "    Pc += probs_on(m, pred)\n"
    "    rows.append((s, ep, *score(pv, y_val), *score(pt, y_test)))\n"
    "    del m\n"
    "    torch.cuda.empty_cache()\n\n"
    "seeds_df = pd.DataFrame(rows, columns=['seed','best_epoch','val_acc',\n"
    "                                       'val_F1','test_acc','test_F1'])\n"
    "print('\\n===== per-seed =====')\n"
    "print(seeds_df.round(3).to_string(index=False))\n"
    "print('\\nbest epoch chosen by val:', sorted(seeds_df.best_epoch.tolist()))"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Ensemble =====\n"
    "Ev, Et = np.mean(Pv, axis=0), np.mean(Pt, axis=0)\n"
    "ens_val, ens_test = score(Ev, y_val), score(Et, y_test)\n"
    "tbl = pd.DataFrame([\n"
    "    ['single (mean of %d)' % len(SEEDS), seeds_df.val_acc.mean(),\n"
    "     seeds_df.val_F1.mean(), seeds_df.test_acc.mean(), seeds_df.test_F1.mean()],\n"
    "    ['ENSEMBLE of %d' % len(SEEDS), *ens_val, *ens_test]],\n"
    "    columns=['mode','val_acc','val_F1','test_acc','test_F1'])\n"
    "print('===== v3 RESULT =====')\n"
    "print(tbl.round(3).to_string(index=False))\n"
    "print(f'\\nseed sd: val {seeds_df.val_acc.std():.3f} | '\n"
    "      f'test {seeds_df.test_acc.std():.3f}')\n"
    "print('\\nbaselines on the same frozen test split:')\n"
    "print('  round 1 (weak->gold, 706 gold) 0.803')\n"
    "print('  v2     (gold-only ensemble)    0.864')\n"
    "print(f'  v3     (this run)              {ens_test[0]:.3f}  '\n"
    "      f'[{ens_test[0]-0.864:+.3f} vs v2]')"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Final report on the frozen test split =====\n"
    "yh = Et.argmax(1)\n"
    "print(classification_report(y_test, yh, labels=list(range(NUM)),\n"
    "      target_names=CLASSES, digits=3, zero_division=0))\n"
    "cm = pd.DataFrame(confusion_matrix(y_test, yh, labels=list(range(NUM))),\n"
    "                  index=CLASSES, columns=CLASSES)\n"
    "print('confusion (rows=true, cols=pred):')\n"
    "print(cm)\n"
    "err = int(cm.values.sum() - np.trace(cm.values))\n"
    "print(f'\\ntotal errors: {err}   (v2 had 20)')\n"
    "print('errors by TRUE class:')\n"
    "for k, v in (cm.sum(1) - np.diag(cm)).sort_values(ascending=False).items():\n"
    "    if v:\n"
    "        print(f'   {k:22s} {v:3d}  ({100*v/max(err,1):.0f}%)')\n"
    "print('\\nremaining confusions (true -> pred):')\n"
    "for i, ti in enumerate(CLASSES):\n"
    "    for j, tj in enumerate(CLASSES):\n"
    "        if i != j and cm.iloc[i, j]:\n"
    "            print(f'   {ti:22s} -> {tj:22s} {cm.iloc[i, j]}')"
))

c.append(nbf.v4.new_code_cell(
    "# ===== Write outputs =====\n"
    "Pr = Pc / len(SEEDS)\n"
    "res = pred[['pmid']].copy()\n"
    "for i, cname in enumerate(CLASSES):\n"
    "    res[f'p_{i}_{cname}'] = Pr[:, i]\n"
    "res['pred_tier'] = Pr.argmax(1)\n"
    "res['pred_class'] = [CLASSES[i] for i in Pr.argmax(1)]\n"
    "res['confidence'] = Pr.max(1)\n"
    "res.to_csv('e3_tier_predictions.csv', index=False)\n"
    "seeds_df.to_csv('e3_seed_results.csv', index=False)\n"
    "tbl.to_csv('e3_ablation.csv', index=False)\n"
    "print('wrote e3_tier_predictions.csv', res.shape)\n"
    "print(res['pred_class'].value_counts())\n"
    "print('mean confidence:', round(res['confidence'].mean(), 3))\n\n"
    "from IPython.display import FileLink, display\n"
    "for f in ['e3_tier_predictions.csv','e3_ablation.csv','e3_seed_results.csv']:\n"
    "    display(FileLink(f))"
))

c.append(nbf.v4.new_markdown_cell(
    "## After running\n"
    "Download the three files (links above), move them into the project folder "
    "**replacing** the old ones -- Windows will otherwise save them as "
    "`e3_tier_predictions (1).csv` and the pipeline will silently keep using "
    "the stale file. Then:\n"
    "```\n"
    "python evidence_build_v3.py\n"
    "python evidence_aggregate.py\n"
    "```"
))

nb["cells"] = c
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}}
with open("Evidence_TierClassifier_Kaggle_v3.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote Evidence_TierClassifier_Kaggle_v3.ipynb with", len(c), "cells")
