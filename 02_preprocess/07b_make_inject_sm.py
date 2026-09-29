#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07b_make_inject_sm.py — rewrites the noisy list of sizectrl (00_filelist_sm)
                        as a 'paired + native noise + injection (+p)' stratified sample.
─────────────────────────────────────────────────────────────────────────────
Design (honesty):
  · denoised (refined set, exp1) keeps the existing 00_filelist_sm as is (unchanged).
  · Only the noisy list is rewritten:  N per class =
        (N - n_noise) benign  [base shared with denoised = paired]
      +  n_noise      noise  [drawn from the actual noise sessions of that class]
    n_noise = round( min(1.0, native_ratio + INJECT) × N )
      native_ratio = actual noise ratio of that class (out_34/perclass_{ds}.csv: removed/before)
      INJECT       = additional injection (default 0.05 = 5%p). ★Must be stated as 'controlled injection' in the paper.
  · Effect: shared benign base → lower variance, suppressed reversals.  Low-noise sets also measurable with +5%p.
  · train/test role decided by the same session-identity hash (crc32) as 05 (zero cross-leakage).

Noise session = at least one of the dataset default_clean_rules (file as is, incl. Aa_7) is 1
  → exactly the complement of the denoised definition (all 0).

Usage:
  python3 07b_make_inject_sm.py                       # 10 datasets, inject 0.05
  python3 07b_make_inject_sm.py --dataset cic17 ustc16 --inject 0.05 --procs 8
  python3 07b_make_inject_sm.py --dataset vpn16 --dry_run     # print counts only
Then:  rebuild with 06_make_dataset.py --variant sizectrl → retrain.
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, ast, csv, glob, os, re, shutil, subprocess, sys, zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
WORK = CODE.parent
DS_BASE = WORK / "01_dataset"
NRDIR = CODE / "noise_rule"
DATA8T = str(RP.DATASETS)
PERCLASS = RP.REFINEMENT_COUNTS
TEST_RATIO = 0.2
SEED = 42
SPLIT_SEED = 42                      # must match the session hash seed of 05

DSMAP = {  # tag: (04 parent, noise_rule file)
 "vpn16":("21_ISCX-VPN-2016","21_noise_rule_vpn16.py"),
 "tor16":("22_ISCX-TOR-2016","22_noise_rule_tor16.py"),
 "tls1.3":("23_CSTNET_TLS1.3","23_noise_rule_tls1.3.py"),
 "cispec":("24_CipherSpectrum","24_noise_rule_cispec.py"),
 
 "ustc16":("01_USTC-TFC_2016","01_noise_rule_ustc16.py"),
 "cic17":("02_CIC-IDS-2017","02_noise_rule_cic17.py"),
 "cic18":("03_CIC-IDS-2018","03_noise_rule_cic18.py"),
 "iot23":("04_CIC_IoT_Dataset_2023","04_noise_rule_iot23.py"),
}
ALL = list(DSMAP)
TRUE = {"1","True","TRUE","true"}
META = ["filename","session_id","task1","task2","task3"]

def default_rules(nrfile):                        # incl. Aa_7 (pipeline definition as is)
    s = open(NRDIR/nrfile, encoding="utf-8").read()
    s = re.sub(r"#.*","",s)
    m = re.search(r"default_clean_rules\s*=\s*\[", s); j=m.end()-1; d=0
    for k in range(j,len(s)):
        if s[k]=="[":d+=1
        elif s[k]=="]":
            d-=1
            if d==0: end=k+1;break
    return ast.literal_eval(s[j:end])

def is_test(fn, sid):                              # same absolute hash threshold as 05
    key=f"{SPLIT_SEED}|{fn}|{sid}".encode("utf-8")
    return (zlib.crc32(key)&0xFFFFFFFF)/0xFFFFFFFF < TEST_RATIO

def proto_stream(sid):
    if sid=="-" or "_" not in sid: return "-","-"
    p,s = sid.split("_",1); return p,s

# ── Worker: extract only 'noise sessions' from one file, up to cap per class ──
def _worker(job):
    fp, rules, need = job                          # need: {class: total required}
    cols = META + rules
    got = defaultdict(list); seen = defaultdict(int)
    for ch in pd.read_csv(fp, usecols=cols, dtype=str, keep_default_na=False, chunksize=300_000):
        mark = ch[rules].isin(TRUE).any(axis=1).to_numpy()
        if not mark.any(): continue
        sub = ch.loc[mark, META]
        for fn,sid,t1,t2,t3 in sub.itertuples(index=False, name=None):
            c=t3
            cap=need.get(c,0)
            if cap and seen[c]<cap*3:              # collect generously up to 3x to allow for the hash split
                got[c].append((fn,sid,t1,t2,t3)); seen[c]+=1
    return {c:v for c,v in got.items()}

def collect_noise(tag, subdir, rules, need, procs):
    files = sorted(glob.glob(os.path.join(DATA8T, subdir, "04_session_noisy_labeled","*.csv")))
    pool_rows = defaultdict(list)
    done = {c:False for c in need}
    jobs = [(f, rules, need) for f in files]
    with Pool(procs) as pool:
        for res in pool.imap_unordered(_worker, jobs):
            for c, rows in res.items():
                if len(pool_rows[c]) < need.get(c,0)*3:
                    pool_rows[c].extend(rows)
            # early stop: stop once every class has required*3
            if need and all(len(pool_rows[c])>=need[c]*3 or need[c]==0 for c in need):
                pool.terminate(); break
    return pool_rows

def load_list(p):
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)

def build(tag, subdir, nrfile, inject, procs, dry):
    fdir = DS_BASE/tag/"00_filelist_sm"
    if not (fdir/"list_train_denoised.csv").exists():
        print(f"[SKIP] {tag}: 00_filelist_sm not found — run 05/07 first"); return
    rules = default_rules(nrfile)
    # native ratio
    native = {}
    pc = PERCLASS/f"perclass_{tag}.csv"
    if pc.exists():
        for r in csv.DictReader(open(pc,encoding="utf-8-sig")):
            b=int(r["before"]); native[r["class"]] = (int(r["removed"])/b if b else 0.0)
    den_tr = load_list(fdir/"list_train_denoised.csv")
    den_te = load_list(fdir/"list_test_denoised.csv")
    # N, n_noise per class (train/test each)
    ntr = den_tr.groupby("group_key").size().to_dict()
    nte = den_te.groupby("group_key").size().to_dict()
    frac = {c: min(1.0, native.get(c,0.0)+inject) for c in set(ntr)|set(nte)}
    nn_tr = {c: int(round(frac[c]*ntr.get(c,0))) for c in ntr}
    nn_te = {c: int(round(frac[c]*nte.get(c,0))) for c in nte}
    need = {c: nn_tr.get(c,0)+nn_te.get(c,0) for c in set(nn_tr)|set(nn_te)}
    need = {c:v for c,v in need.items() if v>0}
    # collect noise sessions
    if dry:
        print(f"[{tag}] inject={inject} · classes {len(ntr)} · noise required (train+test) top:")
        for c in sorted(need, key=lambda x:-need[x])[:8]:
            print(f"    {c:24s} N(tr/te)={ntr.get(c,0)}/{nte.get(c,0)} native={native.get(c,0)*100:4.1f}% → n_noise={nn_tr.get(c,0)}/{nn_te.get(c,0)}")
        return
    pool = collect_noise(tag, subdir, rules, need, procs)
    rng = np.random.default_rng(SEED)
    def make(split, den, nn):
        out=[]
        short=[]
        # split noise into train/test by hash
        for c in den.groupby("group_key").groups if False else set(list(nn)+list(den["group_key"].unique())):
            base = den[den["group_key"]==c]
            N = len(base); k = nn.get(c,0)
            # noise candidates (for this split)
            cand = [row for row in pool.get(c,[]) if is_test(row[0],row[1])==(split=="test")]
            if len(cand) < k:
                short.append((c,k,len(cand))); k=len(cand)
            nc = N - k
            # nc from the benign base (shared), k noise
            b_idx = base.sample(n=min(nc,N), random_state=int(rng.integers(1e9))).copy() if nc>0 else base.iloc[0:0].copy()
            for _,r in b_idx.iterrows():
                out.append([r["filename"],r["session_id"],r["proto"],r["stream"],c,r["task1"],r["task2"],r["task3"]])
            pick = rng.permutation(len(cand))[:k]
            for pi in pick:
                fn,sid,t1,t2,t3=cand[pi]; p,s=proto_stream(sid)
                out.append([fn,sid,p,s,c,t1,t2,t3])
        return out, short
    tr_rows, sh1 = make("train", den_tr, nn_tr)
    te_rows, sh2 = make("test",  den_te, nn_te)
    hdr=["filename","session_id","proto","stream","group_key","task1","task2","task3"]
    # new variant 00_filelist_strat = sizectrl denoised (copy) + noisy (=stratified). The original sm is kept.
    sdir = DS_BASE/tag/"00_filelist_strat"; sdir.mkdir(parents=True, exist_ok=True)
    for f in ("list_train_denoised.csv","list_test_denoised.csv","label_map.json"):
        if (fdir/f).exists(): shutil.copyfile(fdir/f, sdir/f)
    for split, rows in (("train",tr_rows),("test",te_rows)):
        for tgt in (sdir/f"list_{split}_noisy.csv",              # for the pipeline
                    fdir/f"list_{split}_noisy_stratified.csv"):  # reference copy inside sm (original noisy kept)
            with open(tgt,"w",newline="",encoding="utf-8-sig") as fp:
                w=csv.writer(fp); w.writerow(hdr); w.writerows(rows)
    print(f"[{tag}] strat created · train {len(tr_rows):,} / test {len(te_rows):,} "
          f"· inject +{inject*100:.0f}%p · classes short of noise {len(sh1)+len(sh2)}"
          f"\n         → {sdir}  (+ sm/list_*_noisy_stratified.csv)")
    if sh1[:3]: print("     (shortfall e.g.:", sh1[:3], ")")

BYTE = ["2dcnn","etbert","yatc","netmamba","trafficformer"]
def build_ds(tag, workers):
    six = CODE/"06_make_dataset.py"
    def run(model, extra):
        cmd=[sys.executable,str(six),"--dataset",tag,"--model",model,
             "--variant","strat","--workers",str(workers)]+extra
        print(f"  >> 06 {model} (strat)"); subprocess.run(cmd, check=False)
    run("xgboost",[]); run("rf",[])
    for m in BYTE: run(m,["--ip_mask","--port_mask"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=ALL)
    ap.add_argument("--inject", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--build", action="store_true", help="after the filelist, build 7 models with 06 --variant strat")
    ap.add_argument("--workers", type=int, default=8)
    a=ap.parse_args()
    print(f"[07b] inject=+{a.inject*100:.0f}%p · seed={SEED} · procs={a.procs}\n{'='*66}")
    for tag in a.dataset:
        subdir,nrfile = DSMAP[tag]
        build(tag, subdir, nrfile, a.inject, a.procs, a.dry_run)
        if a.build and not a.dry_run: build_ds(tag, a.workers)
    print(f"{'='*66}\n[done] strat filelist → 06 --variant strat build"
          + ("(done)" if a.build else " required"))

if __name__=="__main__":
    main()
