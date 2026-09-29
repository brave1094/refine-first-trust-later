#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 4 evaluation objectivity collapse — main exp1 (refinement pipeline) vs exp4 (unrefined raw-split practice).
   + mechanism: exp2 wrong-answer noise rate / Lift (over-representation vs base rate). sizectrl+full.
   Output: 99_documents/results/analysis/04_fail_to_eval/{variant}/{ds}.csv + _summary.csv(Lift) + figure/"""
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
M03 = WORK / "03_model"; DSROOT = WORK / "01_dataset"; OUT = RP.ANALYSIS_OUT / "04_fail_to_eval"
MODELS = ["rf","xgboost","2dcnn","etbert","netmamba","trafficformer","yatc"]
DSS = ["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]
# variant → filelist folder for the noise definition (sizectrl=00_filelist_sm, full=00_filelist)
FL = {"sizectrl": "00_filelist_sm", "full": "00_filelist"}

def _num(s): m = re.search(r"\d+", str(s)); return m.group(0) if m else ""

def load(variant):
    acc = {}
    for m in MODELS:
        p = RP.RESULT_TABLES / variant / f"{m}.csv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                try: acc[(m, r["dataset"])] = {k: float(r[k]) for k in ("exp1","exp2","exp3","exp4")}
                except: pass
    return acc

def noise_keys(ds, variant):
    fl = DSROOT / ds / FL[variant]
    fn, fd = fl / "list_test_noisy.csv", fl / "list_test_denoised.csv"
    if not (fn.exists() and fd.exists()): return None
    n = pd.read_csv(fn, dtype=str, na_filter=False, encoding="utf-8-sig")
    d = pd.read_csv(fd, dtype=str, na_filter=False, encoding="utf-8-sig")
    n.columns = [c.strip().lstrip("﻿") for c in n.columns]; d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    nk = set(zip(n["filename"], n["session_id"])); dk = set(zip(d["filename"], d["session_id"]))
    noise = nk - dk
    return dict(noise=noise, noise_num=set((f,_num(s)) for f,s in noise),
                total=nk, total_num=set((f,_num(s)) for f,s in nk), base=len(noise)/max(len(nk),1))

for variant in ("sizectrl", "full"):
    acc = load(variant)
    if not acc: print(f"[skip] {variant}"); continue
    (OUT / variant).mkdir(parents=True, exist_ok=True)
    (OUT / "figure").mkdir(parents=True, exist_ok=True)
    summ = []
    for ds in DSS:
        rows = []
        for m in MODELS:
            a = acc.get((m, ds))
            if not a: continue
            rows.append([m, f"{a['exp1']*100:.2f}", f"{a['exp4']*100:.2f}", f"{(a['exp1']-a['exp4'])*100:+.2f}",
                         f"{a['exp2']*100:.2f}", f"{(a['exp1']-a['exp2'])*100:+.2f}"])
        if not rows: continue
        with open(OUT / variant / f"{ds}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["model","exp1_refined(%)","exp4_unrefined_practice(%)","Δ(e1-e4,%p)","exp2_noisy_test(%)","Δ(e1-e2,%p)"]); w.writerows(rows)
        # Lift (exp2 wrong-answer noise rate / base rate)
        k = noise_keys(ds, variant); lift = base = wr = None
        if k:
            base = k["base"]; ratios = []
            for m in MODELS:
                p = RP.WRONG_LIST / variant / m / ds / "exp2_wrong.csv"
                if not p.exists(): continue
                w = pd.read_csv(p, dtype=str, na_filter=False); nn = len(w)
                if nn == 0: continue
                hit = sum(((fn,sid) in k["noise"]) if re.search("[a-zA-Z]", sid) else ((fn,_num(sid)) in k["noise_num"])
                          for fn,sid in zip(w["filename"], w["session_id"]))
                ratios.append(hit/nn)
            if ratios and base > 0: wr = sum(ratios)/len(ratios); lift = wr/base
        d14 = sum((acc[(m,ds)]["exp1"]-acc[(m,ds)]["exp4"]) for m in MODELS if (m,ds) in acc)/max(1,sum(1 for m in MODELS if (m,ds) in acc))*100
        summ.append([ds, f"{d14:+.2f}", f"{base*100:.1f}" if base is not None else "—",
                     f"{wr*100:.1f}" if wr is not None else "—", f"{lift:.2f}" if lift else ("—" if base is None else "0.00"),
                     ("⚠️Lift<1" if (lift is not None and lift<1) else ("control" if base==0 or base is None else "holds"))])
    with open(OUT / variant / "_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["dataset","exp1-exp4(%p)","base_noise_rate(%)","wrong_noise_rate(%)","Lift","verdict"]); w.writerows(summ)
    # figure: Lift bars
    lv = [(s[0], float(s[4])) for s in summ if s[4] not in ("—","0.00")]
    if lv:
        fig, ax = plt.subplots(figsize=(9,4))
        cols = ["#1F3864" if v>=1.3 else ("#7F9DB9" if v>=1 else "#C0504D") for _,v in lv]
        ax.bar([x[0] for x in lv], [x[1] for x in lv], color=cols)
        ax.axhline(1, color="k", lw=0.8, ls="--"); ax.set_ylabel("Lift (wrong-noise / base)")
        ax.set_title(f"Evaluation distortion Lift ({variant}) — >1 = noise over-represented in errors", fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "figure" / f"lift_{variant}.png", dpi=130); plt.close(fig)
    print(f"[04] {variant}: {len(summ)} items → {OUT}")
print("[04 done]")
