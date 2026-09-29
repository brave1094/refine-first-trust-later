#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 1 target-signal suppression — sizectrl/full: exp1 (refined) vs exp3 (noisy) accuracy.
   Output: 99_documents/results/analysis/01_target_signal/{variant}/{ds}.csv + _summary.csv + figure/{variant}/{ds}.png"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os, csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK = RP.ROOT                      # 02_SCIE_NOISE
M03 = WORK / "03_model"; OUT = RP.ANALYSIS_OUT / "01_target_signal"
MODELS = ["rf","xgboost","2dcnn","etbert","netmamba","trafficformer","yatc"]
DSS = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]

def load(variant):
    acc = {}
    for m in MODELS:
        p = RP.RESULT_TABLES / variant / f"{m}.csv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                try: acc[(m, r["dataset"])] = {k: float(r[k]) for k in ("exp1","exp2","exp3","exp4")}
                except: pass
    return acc

for variant in ("sizectrl", "full"):
    acc = load(variant)
    if not acc: print(f"[skip] {variant}: no result_table"); continue
    (OUT / variant).mkdir(parents=True, exist_ok=True)
    (OUT / "figure" / variant).mkdir(parents=True, exist_ok=True)
    summ = []
    for ds in DSS:
        rows, deltas = [], []
        for m in MODELS:
            a = acc.get((m, ds))
            if not a: continue
            d = (a["exp1"] - a["exp3"]) * 100
            rows.append([m, f"{a['exp1']*100:.2f}", f"{a['exp3']*100:.2f}", f"{d:+.2f}"])
            deltas.append((m, a["exp1"]*100, a["exp3"]*100, d))
        if not rows: continue
        with open(OUT / variant / f"{ds}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["model","exp1_refined(%)","exp3_noisy(%)","Δ(e1-e3,%p)"]); w.writerows(rows)
        # figure: grouped bar exp1 vs exp3
        ms = [x[0] for x in deltas]; e1 = [x[1] for x in deltas]; e3 = [x[2] for x in deltas]
        xs = range(len(ms)); wbar = 0.38
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar([x - wbar/2 for x in xs], e1, wbar, label="exp1 (denoised-train)", color="#1F3864")
        ax.bar([x + wbar/2 for x in xs], e3, wbar, label="exp3 (noisy-train)", color="#C0504D")
        ax.set_xticks(list(xs)); ax.set_xticklabels(ms, rotation=20, fontsize=8)
        ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100)
        avg = sum(x[3] for x in deltas) / len(deltas)
        ax.set_title(f"{ds} / {variant}  —  exp1 vs exp3 (mean Δ = {avg:+.1f}%p)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "figure" / variant / f"{ds}.png", dpi=130); plt.close(fig)
        pos = sum(1 for x in deltas if x[3] > 0)
        summ.append([ds, f"{avg:+.2f}", pos, len(deltas)])
    with open(OUT / variant / "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["dataset","mean Δ(e1-e3,%p)","n_models_refined_better","total"]); w.writerows(summ)
    # overall summary bars
    if summ:
        fig, ax = plt.subplots(figsize=(9, 4))
        vals = [float(s[1]) for s in summ]; cols = ["#1F3864" if v > 0 else "#C0504D" for v in vals]
        ax.bar([s[0] for s in summ], vals, color=cols)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("mean Δ exp1−exp3 (%p)")
        ax.set_title(f"Target-signal interference ({variant}) — higher = clean training wins", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "figure" / f"summary_delta_{variant}.png", dpi=130); plt.close(fig)
    print(f"[01] {variant}: {len(summ)} datasets → {OUT}")
print("[01 done]")
