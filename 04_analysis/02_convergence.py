#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 2 learning dynamics (convergence delay) — sizectrl: DL model exp1 vs exp3 epoch curves (train_loss/acc, test_acc).
   Output: 99_documents/results/analysis/02_convergence/csv/{ds}/{model}.csv + figure/{ds}/{model}.png"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK = RP.ROOT
RES = WORK / "03_model" / "results" / "sizectrl"
OUT = RP.ANALYSIS_OUT / "02_convergence"
DL = ["2dcnn","etbert","netmamba","trafficformer","yatc"]
DSS = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]

def epochs(m, ds, e):
    p = RES / m / ds / f"{m}_{ds}_exp{e}.csv"
    if not p.exists(): return []
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        def g(k):
            try: return float(r[k])
            except: return None
        out.append((int(r["epoch"]), g("train_loss"), g("train_accuracy"), g("test_accuracy")))
    return out

n_fig = 0
for ds in DSS:
    for m in DL:
        e1, e3 = epochs(m, ds, 1), epochs(m, ds, 3)
        if not e1 and not e3: continue
        (OUT / "csv" / ds).mkdir(parents=True, exist_ok=True)
        (OUT / "figure" / ds).mkdir(parents=True, exist_ok=True)
        # merged CSV
        d1 = {x[0]: x for x in e1}; d3 = {x[0]: x for x in e3}
        eps = sorted(set(d1) | set(d3))
        with open(OUT / "csv" / ds / f"{m}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["epoch","e1_loss","e1_train_acc","e1_test_acc","e3_loss","e3_train_acc","e3_test_acc"])
            for ep in eps:
                a = d1.get(ep, (ep,None,None,None)); b = d3.get(ep, (ep,None,None,None))
                w.writerow([ep, a[1],a[2],a[3], b[1],b[2],b[3]])
        # figure: loss + acc
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        x1 = [x[0] for x in e1]; x3 = [x[0] for x in e3]
        ax[0].plot(x1, [x[1] for x in e1], "-o", ms=3, label="exp1 denoised", color="#1F3864")
        ax[0].plot(x3, [x[1] for x in e3], "-s", ms=3, label="exp3 noisy", color="#C0504D")
        ax[0].set_title("train loss"); ax[0].set_xlabel("epoch"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
        ax[1].plot(x1, [x[2] for x in e1], "-o", ms=3, label="exp1 train_acc", color="#1F3864")
        ax[1].plot(x3, [x[2] for x in e3], "-s", ms=3, label="exp3 train_acc", color="#C0504D")
        t1 = [(x[0], x[3]) for x in e1 if x[3] is not None]
        t3 = [(x[0], x[3]) for x in e3 if x[3] is not None]
        if t1: ax[1].plot([p[0] for p in t1], [p[1] for p in t1], "--^", ms=4, label="exp1 test_acc", color="#2E75B6")
        if t3: ax[1].plot([p[0] for p in t3], [p[1] for p in t3], "--v", ms=4, label="exp3 test_acc", color="#E8A33D")
        ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylim(0,1); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
        fig.suptitle(f"{ds} / {m}  —  convergence (sizectrl)", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / "figure" / ds / f"{m}.png", dpi=130); plt.close(fig)
        n_fig += 1
print(f"[02 done] figure {n_fig} files → {OUT}")
