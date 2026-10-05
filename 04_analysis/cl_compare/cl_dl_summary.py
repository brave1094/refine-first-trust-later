#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cl_dl_summary.py — aggregates the DL results of the CL comparison. Can be run midway (uses only finished cells).
Accuracy on the refined test set (Exp3 evaluation). raw = existing sizectrl Exp3, e1 = existing sizectrl Exp1
(independently refined sample, for reference).
Output: dl_long.csv, dl_summary.csv in 99_documents/results/analysis/09_cl_compare/ + console tables."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import numpy as np
import pandas as pd
from scipy import stats

RT = str(RP.RESULT_TABLES)
OUT = str(RP.ANALYSIS_OUT / "09_cl_compare")
SEEDS = [42, 1, 7, 2024, 31337]
MODELS = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
DSS = ["vpn16", "tor16", "iot23", "cic17"]


def get(variant, seed, model, ds, exp):
    v = variant + ("" if seed == 42 else f"_seed{seed}")
    p = os.path.join(RT, v, f"{model}.csv")
    if not os.path.exists(p):
        return np.nan
    t = pd.read_csv(p, dtype={"dataset": str})
    r = t[t["dataset"] == ds]
    if r.empty or pd.isna(r.iloc[0].get(f"exp{exp}")):
        return np.nan
    return float(r.iloc[0][f"exp{exp}"]) * 100


rows = []
for ds in DSS:
    for m in MODELS:
        for s in SEEDS:
            rows.append(dict(dataset=ds, model=m, seed=s,
                             e1=get("sizectrl", s, m, ds, 1), raw=get("sizectrl", s, m, ds, 3),
                             rule=get("cl_rule", s, m, ds, 3), cl_budget=get("cl_budget", s, m, ds, 3),
                             random=get(f"cl_random_s{s}", s, m, ds, 3), cl_default=get("cl_default", s, m, ds, 3)))
L = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
L.to_csv(os.path.join(OUT, "dl_long.csv"), index=False)

C = ["e1", "raw", "rule", "cl_budget", "random", "cl_default"]
pd.set_option("display.width", 200)
print("=== mean accuracy per model (refined test set, %, finished seeds only; n = number of seeds) ===")
g = L.groupby(["dataset", "model"])
tab = g[C].mean().round(2)
tab["n_rule"] = g["rule"].count()
print(tab.to_string())


def paired(a, b, sub):
    d = (sub[a] - sub[b]).dropna()
    if len(d) < 2:
        return len(d), (d.mean() if len(d) else np.nan), np.nan, np.nan
    ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_1samp(d, 0).pvalue
    return len(d), d.mean(), ci, p


print("\n=== paired comparison (same model, same seed, pp; models × seeds pooled per dataset) ===")
out = []
for ds in DSS:
    sub = L[L["dataset"] == ds]
    for a, b in (("rule", "cl_budget"), ("rule", "random"), ("cl_budget", "random"), ("rule", "raw"),
                 ("cl_budget", "raw"), ("random", "raw"), ("cl_default", "raw")):
        n, mu, ci, p = paired(a, b, sub)
        if n:
            pos = int(((sub[a] - sub[b]).dropna() > 0).sum())
            out.append(dict(dataset=ds, comparison=f"{a} - {b}", n=n, mean_pp=round(mu, 2),
                            ci95_pp=round(ci, 2) if ci == ci else "", p=round(p, 4) if p == p else "",
                            positive=f"{pos}/{n}"))
S = pd.DataFrame(out)
S.to_csv(os.path.join(OUT, "dl_summary.csv"), index=False)
print(S.to_string(index=False))
