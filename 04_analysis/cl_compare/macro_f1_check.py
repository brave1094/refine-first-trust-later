#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""macro_f1_check.py — Exp1 vs Exp3 macro-F1 on the refined test set (sizectrl, seed 42), from the per-session wrong
lists (99_documents/results/wrong_lists/sizectrl/{model}/{ds}/exp1_wrong.csv, exp3_wrong.csv) and the class counts of
the test set actually evaluated (01_dataset/{ds}/sizectrl/xgboost/test_denoised/features.csv, Label column).
Per class: FN = wrong rows with true=c, FP = wrong rows with pred=c, TP = n_c - FN.  Accuracy is recomputed as a check.
out: 99_documents/results/analysis/09_cl_compare/macro_f1_check.csv (dataset, model, n_test, acc1, acc3, f1_1, f1_3,
d_acc_pp, d_f1_pp) and a dataset summary.
run: python3 macro_f1_check.py"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv, os, statistics as st
from collections import Counter
import pandas as pd
import cl_compare as C

WL = str(RP.WRONG_LIST / "sizectrl")
MODELS = ["rf", "xgboost", "2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
ORDER = ["vpn16", "tor16", "iot23", "ustc16", "cic18", "tls1.3", "cispec", "cic17"]


def macro(n_c, wrong):
    fn = Counter(t for t, p in wrong)
    fp = Counter(p for t, p in wrong)
    f1s = []
    for c, n in n_c.items():
        tp = n - fn[c]
        den = 2 * tp + fp[c] + fn[c]
        f1s.append(2 * tp / den if den else 0.0)
    return 100 * sum(f1s) / len(f1s), 100 * (1 - len(wrong) / sum(n_c.values()))


def wrong(p):
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return list(zip(df["true_label"], df["pred_label"]))


rows = []
for ds in ORDER:
    fp_ = os.path.join(C.DSROOT, ds, "sizectrl", "xgboost", "test_denoised", "features.csv")
    n_c = Counter(pd.read_csv(fp_, usecols=["Label"], dtype=str)["Label"])
    for m in MODELS:
        p1, p3 = os.path.join(WL, m, ds, "exp1_wrong.csv"), os.path.join(WL, m, ds, "exp3_wrong.csv")
        if not (os.path.exists(p1) and os.path.exists(p3)):
            continue
        w1, w3 = wrong(p1), wrong(p3)
        unk = {t for t, _ in w1 + w3} - set(n_c)
        f1, a1 = macro(n_c, w1)
        f3, a3 = macro(n_c, w3)
        rows.append([ds, m, sum(n_c.values()), round(a1, 2), round(a3, 2), round(f1, 2), round(f3, 2),
                     round(a1 - a3, 2), round(f1 - f3, 2), len(unk)])
        C.log(f"[f1] {ds:8} {m:13} acc {a1:6.2f}->{a3:6.2f}  macroF1 {f1:6.2f}->{f3:6.2f}  unknown labels {len(unk)}")
os.makedirs(C.OUT, exist_ok=True)
out = os.path.join(C.OUT, "macro_f1_check.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "model", "n_test", "acc1", "acc3", "f1_1", "f1_3", "d_acc_pp", "d_f1_pp", "unknown_labels"])
    w.writerows(rows)
    w.writerow([])
    w.writerow(["dataset", "mean_d_acc_pp", "mean_d_f1_pp", "models_f1_positive"])
    for ds in ORDER:
        r = [x for x in rows if x[0] == ds]
        if r:
            w.writerow([ds, round(st.mean(x[7] for x in r), 2), round(st.mean(x[8] for x in r), 2), f"{sum(x[8] > 0 for x in r)}/{len(r)}"])
C.log(f"[f1] -> {out}")
