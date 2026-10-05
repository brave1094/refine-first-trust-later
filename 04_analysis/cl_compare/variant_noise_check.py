#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""variant_noise_check.py — checks the sampled noise rate of the feature files actually used for training
(full / sizectrl / strat) (cic18, iot23).

Background: the training data of the seven models for cic18 and iot23 were built from the 8/22 lists, while the sampled
noise rates of the article were computed from the lists rebuilt on 8/27–28 (sizectrl is checked in the keys stage of
cl_compare.py: cic18 0.31%, iot23 25.1%).
This script checks the full and strat variants with the same method (default_clean_rules − Aa_7, 04_session_noisy_labeled).

Output: 99_documents/results/analysis/09_cl_compare/variant_noise_check.csv
  dataset, variant, split, n_features, noise_n, noise_pct, n_list, overlap_with_list, classes_features, classes_list
Run: python3 variant_noise_check.py --procs 16
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, glob, os, csv
from multiprocessing import Pool
import pandas as pd
import cl_compare as C

VARIANTS = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}
SPLITS = ["train_noisy", "test_noisy", "train_denoised", "test_denoised"]


def fkeys(ds, variant, split):
    p = os.path.join(C.DSROOT, ds, variant, "xgboost", split, "features.csv")
    df = pd.read_csv(p, usecols=["Label", "filename", "Stream_num", "protocol"], low_memory=False)
    pr = pd.to_numeric(df["protocol"], errors="coerce").fillna(-1).astype(int)
    st = pd.to_numeric(df["Stream_num"], errors="coerce").fillna(-1).astype(int)
    keys = [(fn, f"{C.PROTO.get(q, str(q))}_{s}") for fn, q, s in zip(df["filename"].astype(str), pr, st)]
    return keys, df["Label"].astype(str).nunique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cic18", "iot23"])
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(C.OUT, exist_ok=True)
    outp = os.path.join(C.OUT, "variant_noise_check.csv")
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "split", "n_features", "noise_n", "noise_pct", "n_list", "overlap_with_list",
                    "classes_features", "classes_list"])
        for ds in a.datasets:
            R = C.rules_of(ds)
            fk = {}
            for v in VARIANTS:
                for sp in SPLITS:
                    p = os.path.join(C.DSROOT, ds, v, "xgboost", sp, "features.csv")
                    if os.path.exists(p):
                        fk[(v, sp)] = fkeys(ds, v, sp)
            need = set(k for keys, _ in fk.values() for k in keys)
            files = sorted(glob.glob(os.path.join(C.D8, C.MAP[ds][0], "04_session_noisy_labeled", "*.csv")))
            C.log(f"[variant] {ds}: sessions {len(need):,} · rules {len(R)} · 04 CSVs {len(files)}")
            hit = {}
            with Pool(min(a.procs, max(1, len(files))), initializer=C._kinit, initargs=(R, need)) as pool:
                for i, (h, s) in enumerate(pool.imap_unordered(C._kwork, files), 1):
                    hit.update(h)
                    if i % 50 == 0 or i == len(files):
                        C.log(f"   {i}/{len(files)}  noise {len(hit):,}")
            for (v, sp), (keys, ncls) in fk.items():
                lp = os.path.join(C.DSROOT, ds, VARIANTS[v], f"list_{sp}.csv")
                lk, lcls = set(), ""
                if os.path.exists(lp):
                    l = pd.read_csv(lp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                    lk = set(zip(l["filename"], l["session_id"]))
                    lab = [c for c in l.columns if c.lower() in ("label", "class", "task3", "label_name")]
                    lcls = l[lab[0]].nunique() if lab else ""
                nn = sum(k in hit for k in keys)
                ov = sum(k in lk for k in keys) / max(1, len(keys))
                row = [ds, v, sp, len(keys), nn, round(100 * nn / max(1, len(keys)), 2), len(lk), round(ov, 4), ncls, lcls]
                w.writerow(row)
                f.flush()
                C.log("   " + " ".join(map(str, row)))
    C.log(f"[variant] → {outp}")


if __name__ == "__main__":
    main()
