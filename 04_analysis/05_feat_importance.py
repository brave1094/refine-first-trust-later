#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 5 feature importance distortion — [distortion] rf-impurity exp1 (refined) vs exp3 (noisy): rankρ·artifact massΔ·top-k.
   (TreeSHAP·DL SHAP 3-view byte→field profiles are a separate pipeline — only rf-impurity distortion here)
   Output: 99_documents/results/analysis/05_feat_importance/{variant}/{ds}_rf_top25.csv + _summary.csv + figure/"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = RP.ROOT
M03 = WORK / "03_model"; OUT = RP.ANALYSIS_OUT / "05_feat_importance"
DSS = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]
ART = re.compile(r"(ip_id|checksum|ip_ttl|dsfield|ip_flags|_frag|port|addr|src|dst|ip_ver|ip_total_len)", re.I)

def imp(variant, ds, exp):
    p = RP.PARAM / variant / "rf" / ds / f"exp{exp}" / "feature_importance.csv"
    if not p.exists(): return None
    df = pd.read_csv(p); col = "importance" if "importance" in df.columns else "gain"
    s = df.set_index(df.columns[0])[col]
    return s / s.sum() if s.sum() > 0 else s

for variant in ("full", "sizectrl"):
    (OUT / variant).mkdir(parents=True, exist_ok=True)
    (OUT / "figure").mkdir(parents=True, exist_ok=True)
    summ = []
    for ds in DSS:
        a, b = imp(variant, ds, 1), imp(variant, ds, 3)
        if a is None or b is None: continue
        idx = a.index.union(b.index); a = a.reindex(idx, fill_value=0); b = b.reindex(idx, fill_value=0)
        rho = a.rank().corr(b.rank())                      # Pearson on ranks = Spearman
        art1 = a[a.index.to_series().str.contains(ART, regex=True)].sum()
        art3 = b[b.index.to_series().str.contains(ART, regex=True)].sum()
        out = pd.DataFrame({"exp1_refined": a, "exp3_noisy": b}); out["delta"] = out["exp3_noisy"] - out["exp1_refined"]
        out.sort_values("exp3_noisy", ascending=False).head(25).round(6).to_csv(
            OUT / variant / f"{ds}_rf_top25.csv", encoding="utf-8-sig")
        summ.append([ds, f"{rho:.3f}", f"{art1*100:.1f}", f"{art3*100:.1f}", f"{(art3-art1)*100:+.1f}",
                     a.idxmax(), b.idxmax(), ("⚠️reversed" if (art3-art1)<0 else ("strong_distortion" if (art3-art1)*100>5 else "weak"))])
    with open(OUT / variant / "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset","rankρ(e1,e3)","artifact_mass_e1(%)","_e3(%)","Δ(%p)","top1_e1","top1_e3","verdict"]); w.writerows(summ)
    # figure: artifact mass Δ bars
    if summ:
        fig, ax = plt.subplots(figsize=(9,4))
        vals = [float(s[4]) for s in summ]; cols = ["#C0504D" if v<0 else "#1F3864" for v in vals]
        ax.bar([s[0] for s in summ], vals, color=cols)
        ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("artifact-mass Δ exp3−exp1 (%p)")
        ax.set_title(f"Importance distortion ({variant}) — noise shifts mass to artifacts (ip_id …)", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "figure" / f"artifact_delta_{variant}.png", dpi=130); plt.close(fig)
    print(f"[05] {variant}: {len(summ)} items (rf-impurity) → {OUT}")
print("[05 done]")
