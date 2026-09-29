#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_sz_strat.py — head-to-head comparison of sizectrl vs strat (runs completed so far, fair matching).
   Principle: compare only cells where 'the same (model,ds) is completed in both variants' (matched).
        exp1(den/den) is shared by both variants (sizectrl reused). Differences are only exp3/exp4/noise definition.
   Rows = claim1·4·5 (directly comparable). claim2=curves/claim3=no strat embeddings → excluded (separate).
   Whether a claim 'survived more' = the larger the absolute metric, the stronger the refinement effect."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv, re
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent; WORK = RP.ROOT
M03=WORK/"03_model"; DSROOT=WORK/"01_dataset"
MODELS=["rf","xgboost","2dcnn","etbert","netmamba","trafficformer","yatc"]
DSS=["vpn16","tor16","tls1.3","cispec","ustc16","cic17","cic18","iot23"]

def _f(x):
    try: return float(x)
    except: return None
def read(variant):
    d={}
    for m in MODELS:
        p=RP.RESULT_TABLES/variant/f"{m}.csv"
        if not p.exists(): continue
        for r in csv.DictReader(open(p,encoding="utf-8-sig")):
            d[(m,r["dataset"])]={k:_f(r.get(k,"")) for k in ("exp1","exp2","exp3","exp4")}
    return d
SZ=read("sizectrl"); ST=read("strat")
# exp1 is shared by both variants (sizectrl). strat has its own values only for exp2/3/4.
def e1(k): return SZ.get(k,{}).get("exp1")

# ── claim1: refined-training advantage = exp1 − exp3 (%p) ─────────────────────────────
def claim1():
    cells=[]
    for m in MODELS:
        for ds in DSS:
            k=(m,ds); a1=e1(k)
            e3s=SZ.get(k,{}).get("exp3"); e3t=ST.get(k,{}).get("exp3")
            if a1 is None or e3s is None or e3t is None: continue  # only cells completed in both
            cells.append((k,(a1-e3s)*100,(a1-e3t)*100))
    if not cells: return None
    n=len(cells)
    sz=sum(c[1] for c in cells)/n; st=sum(c[2] for c in cells)/n
    win_sz=sum(1 for c in cells if c[1]>0); win_st=sum(1 for c in cells if c[2]>0)
    return dict(n=n, sz=sz, st=st, win_sz=win_sz, win_st=win_st,
                cells=cells)

# ── claim4a: non-refined-practice gap = exp1 − exp4 (%p) ──────────────────────────
def claim4a():
    cells=[]
    for m in MODELS:
        for ds in DSS:
            k=(m,ds); a1=e1(k)
            e4s=SZ.get(k,{}).get("exp4"); e4t=ST.get(k,{}).get("exp4")
            if a1 is None or e4s is None or e4t is None: continue
            cells.append((k,(a1-e4s)*100,(a1-e4t)*100))
    if not cells: return None
    n=len(cells); sz=sum(c[1] for c in cells)/n; st=sum(c[2] for c in cells)/n
    return dict(n=n, sz=sz, st=st, cells=cells)

# ── claim4b: Lift (noise rate among errors / base) — per dataset ──────────────────
FL={"sizectrl":"00_filelist_sm","strat":"00_filelist_strat"}
def _num(s): m=re.search(r"\d+",str(s)); return m.group(0) if m else ""
def _noise(ds,variant):
    fl=DSROOT/ds/FL[variant]; fn,fd=fl/"list_test_noisy.csv",fl/"list_test_denoised.csv"
    if not (fn.exists() and fd.exists()): return None
    n=pd.read_csv(fn,dtype=str,na_filter=False,encoding="utf-8-sig"); d=pd.read_csv(fd,dtype=str,na_filter=False,encoding="utf-8-sig")
    n.columns=[c.strip().lstrip("﻿") for c in n.columns]; d.columns=[c.strip().lstrip("﻿") for c in d.columns]
    nk=set(zip(n["filename"],n["session_id"])); dk=set(zip(d["filename"],d["session_id"])); noise=nk-dk
    if len(nk)==0: return None
    return dict(noise=noise,noise_num=set((f,_num(s)) for f,s in noise),base=len(noise)/len(nk))
def _lift(ds,variant):
    k=_noise(ds,variant)
    if not k or k["base"]<=0: return None
    ratios=[]
    for m in MODELS:
        p=RP.WRONG_LIST/variant/m/ds/"exp2_wrong.csv"
        if not p.exists(): continue
        w=pd.read_csv(p,dtype=str,na_filter=False); nn=len(w)
        if nn==0: continue
        hit=sum(((fn,sid) in k["noise"]) if re.search("[a-zA-Z]",sid) else ((fn,_num(sid)) in k["noise_num"])
                for fn,sid in zip(w["filename"],w["session_id"]))
        ratios.append(hit/nn)
    if not ratios: return None
    return (sum(ratios)/len(ratios))/k["base"]
def claim4b():
    rows=[]
    for ds in DSS:
        ls=_lift(ds,"sizectrl"); lt=_lift(ds,"strat")
        if ls is None or lt is None: continue        # only datasets computed in both
        rows.append((ds,ls,lt))
    return rows

# ── claim5: rf-impurity artifact mass Δ = art(exp3) − art(exp1) (%p) ──────
ART=re.compile(r"(?:ip_id|checksum|ip_ttl|dsfield|ip_flags|_frag|port|addr|src|dst|ip_ver|ip_total_len)",re.I)
def _imp(variant,ds,exp):
    p=RP.PARAM/variant/"rf"/ds/f"exp{exp}"/"feature_importance.csv"
    if not p.exists(): return None
    df=pd.read_csv(p); col="importance" if "importance" in df.columns else "gain"
    s=df.set_index(df.columns[0])[col]; return s/s.sum() if s.sum()>0 else s
def _artmass(s):
    return s[s.index.to_series().str.contains(ART,regex=True)].sum()
def claim5():
    rows=[]
    for ds in DSS:
        a1=_imp("sizectrl",ds,1)                    # exp1 shared (sizectrl)
        b_sz=_imp("sizectrl",ds,3); b_st=_imp("strat",ds,3)
        if a1 is None or b_sz is None or b_st is None: continue
        art1=_artmass(a1)*100
        d_sz=_artmass(b_sz.reindex(a1.index.union(b_sz.index),fill_value=0))*100 - art1
        d_st=_artmass(b_st.reindex(a1.index.union(b_st.index),fill_value=0))*100 - art1
        rows.append((ds,d_sz,d_st))
    return rows

def verdict(sz,st,bigger_is_stronger=True):
    if sz is None or st is None: return "?"
    diff=st-sz
    if abs(diff)<1e-9: return "≈ same"
    stronger = (st>sz) if bigger_is_stronger else (st<sz)
    return f"strat {'stronger↑' if stronger else 'weaker↓'} ({diff:+.2f})"

c1=claim1(); c4a=claim4a(); c4b=claim4b(); c5=claim5()
print("="*78)
print(" sizectrl vs strat  —  claim survival comparison (matched completed cells)")
print("="*78)

print("\n[claim1] target signal: refined-training advantage = exp1 − exp3  (larger = stronger refinement effect)")
if c1:
    print(f"   matched {c1['n']} cells |  sizectrl mean {c1['sz']:+.2f}%p (refined wins {c1['win_sz']}/{c1['n']})"
          f"  vs  strat mean {c1['st']:+.2f}%p (refined wins {c1['win_st']}/{c1['n']})")
    print(f"   → {verdict(c1['sz'],c1['st'])}")
else: print("   (no matched cells)")

print("\n[claim4a] evaluation objectivity (accuracy axis): exp1 − exp4  (larger = more overestimation by non-refined practice)")
if c4a:
    print(f"   matched {c4a['n']} cells |  sizectrl mean {c4a['sz']:+.2f}%p  vs  strat mean {c4a['st']:+.2f}%p")
    print(f"   → {verdict(c4a['sz'],c4a['st'])}")
else: print("   (no matched cells)")

print("\n[claim4b] evaluation objectivity (Lift): error noise rate/base  (>1 & larger = noise dominates errors)")
if c4b:
    print(f"   {'ds':<9} sizectrl   strat    verdict")
    for ds,ls,lt in c4b:
        print(f"   {ds:<9} {ls:6.2f}    {lt:6.2f}    {verdict(ls,lt)}")
    ms=sum(x[1] for x in c4b)/len(c4b); mt=sum(x[2] for x in c4b)/len(c4b)
    print(f"   {'mean':<9} {ms:6.2f}    {mt:6.2f}    {verdict(ms,mt)}   (matched {len(c4b)} datasets)")
else: print("   (no matched datasets)")

print("\n[claim5] feature distortion (rf-impurity): artifact mass Δ = art(exp3)−art(exp1)  (larger = stronger distortion)")
if c5:
    print(f"   {'ds':<9} sizectrl   strat    verdict")
    for ds,ds_sz,ds_st in c5:
        print(f"   {ds:<9} {ds_sz:+6.1f}    {ds_st:+6.1f}    {verdict(ds_sz,ds_st)}")
    ms=sum(x[1] for x in c5)/len(c5); mt=sum(x[2] for x in c5)/len(c5)
    print(f"   {'mean':<9} {ms:+6.1f}    {mt:+6.1f}    {verdict(ms,mt)}   (matched {len(c5)})")
else: print("   (no matched datasets)")

print("\n[claim2] convergence delay = epoch curves (not a single number) → no comparison table, inspect visually (02_convergence/strat/figure)")
print("[claim3] boundary collapse = no strat penultimate embedding dump → server dump code needed (below)")
print("="*78)
