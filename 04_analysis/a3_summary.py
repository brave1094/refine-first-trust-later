#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a3_summary.py — a3_emb_long.csv → (model,dataset) exp1 vs exp3 sil/sep delta pivot.
   Run: python3 a3_summary.py   →  a3_emb_summary.csv (+ console output)"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(str(RP.EMB_DIR), "a3_emb_long.csv")
OUT = os.path.join(str(RP.EMB_DIR), "a3_emb_summary.csv")

d = {}
with open(LONG, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        d[(r["model"], r["dataset"], r["exp"])] = r

keys = sorted({(m, ds) for (m, ds, e) in d})
rows = []
for m, ds in keys:
    a, b = d.get((m, ds, "1")), d.get((m, ds, "3"))
    if not a or not b:
        continue
    s1, s3 = float(a["silhouette"]), float(b["silhouette"])
    p1, p3 = float(a["sep_ratio"]), float(b["sep_ratio"])
    rows.append([m, ds, round(s1, 4), round(s3, 4), round(s1 - s3, 4),
                 round(p1, 4), round(p3, 4), round(p1 - p3, 4)])

rows.sort()
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["model", "dataset", "sil_exp1", "sil_exp3", "sil_Δ(1-3)",
                "sep_exp1", "sep_exp3", "sep_Δ(1-3, +=refined_separates_better)"])
    w.writerows(rows)

print(f"{'model':<14}{'dataset':<9}{'silΔ':>8}{'sepΔ':>8}")
for m, ds, s1, s3, sd, p1, p3, pd in rows:
    print(f"{m:<14}{ds:<9}{sd:>+8.3f}{pd:>+8.2f}")
print(f"\n[a3] → {OUT} ({len(rows)} rows)")
