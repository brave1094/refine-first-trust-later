#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 3 class-boundary collapse — sizectrl: penultimate embedding silhouette + t-SNE (exp1 vs exp3).
   Input: 99_documents/results/embeddings/a3_emb_long_sizectrl.csv, 99_documents/results/embeddings/a3_tsne/sizectrl/{ds}/{model}_exp{1,3}.csv
   Output: 99_documents/results/analysis/03_boundary/silhouette.csv + _summary.csv + figure/{ds}/{model}.png"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK = RP.ROOT
M03 = WORK / "03_model"; OUT = RP.ANALYSIS_OUT / "03_boundary"
EMB = RP.EMB_DIR / "a3_emb_long_sizectrl.csv"; TSNE = RP.EMB_DIR / "a3_tsne" / "sizectrl"
MODELS = ["2dcnn", "yatc", "netmamba", "etbert", "trafficformer"]  # the 2 UER models are
#   extracted additionally with a3_extract_emb_uer.py (if missing, sil.get returns None → auto skip)
NPY = MODELS
DSS = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]
OUT.mkdir(parents=True, exist_ok=True)

# ── silhouette table ──
sil = {}
if EMB.exists():
    for r in csv.DictReader(open(EMB, encoding="utf-8-sig")):
        sil[(r["model"], r["dataset"], r["exp"])] = (float(r["silhouette"]), float(r.get("sep_ratio", 0) or 0))
rows, summ = [], []
for ds in DSS:
    ds_d = []
    for m in NPY:
        a = sil.get((m, ds, "1")); b = sil.get((m, ds, "3"))
        if not a or not b: continue
        d = a[0] - b[0]
        rows.append([ds, m, f"{a[0]:.3f}", f"{b[0]:.3f}", f"{d:+.3f}", f"{a[1]:.2f}", f"{b[1]:.2f}"])
        ds_d.append(d)
    if ds_d:
        md = sum(ds_d) / len(ds_d)
        summ.append([ds, f"{md:+.3f}", sum(1 for x in ds_d if x > 0), len(ds_d),
                     "collapse" if md > 0.02 else ("⚠️reversed" if md < 0 else "neutral")])
with open(OUT / "silhouette.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["dataset","model","sil_e1","sil_e3","silΔ(e1-e3)","sep_e1","sep_e3"]); w.writerows(rows)
with open(OUT / "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["dataset","mean silΔ","n_collapsed_models","total","verdict"]); w.writerows(summ)

# ── t-SNE figure (exp1 vs exp3) ──
def coords(ds, m, e):
    p = TSNE / ds / f"{m}_exp{e}.csv"
    if not p.exists(): return None
    xs, ys, ls = [], [], []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        xs.append(float(r["x"])); ys.append(float(r["y"])); ls.append(int(r["label"]))
    return xs, ys, ls
n = 0
for ds in DSS:
    for m in NPY:
        c1, c3 = coords(ds, m, 1), coords(ds, m, 3)
        if not c1 or not c3: continue
        (OUT / "figure" / ds).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for j, (e, c) in enumerate(((1, c1), (3, c3))):
            ax[j].scatter(c[0], c[1], c=c[2], s=7, cmap="tab20", alpha=0.75)
            s = sil.get((m, ds, str(e)))
            tag = "exp1 denoised-train" if e == 1 else "exp3 noisy-train"
            ax[j].set_title(f"{tag}" + (f"  sil={s[0]:.3f}" if s else "")); ax[j].set_xticks([]); ax[j].set_yticks([])
        sd = (sil.get((m,ds,"1"),(0,0))[0] - sil.get((m,ds,"3"),(0,0))[0])
        fig.suptitle(f"{ds} / {m}  penultimate t-SNE  (silΔ = {sd:+.3f})", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / "figure" / ds / f"{m}.png", dpi=130); plt.close(fig); n += 1
print(f"[03 done] silhouette {len(rows)} rows, t-SNE fig {n} → {OUT}")
print("     (full-version embeddings and etbert/trafficformer need additional extraction)")
