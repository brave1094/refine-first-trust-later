#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a6_audit.py — integrity audit of shap_dl outputs + automatic deletion of contaminated files.

Background: if a run is restarted while an earlier process still writes into the same output folder,
outputs drawn from different samples can get mixed. This script compares the pred.csv y_true column of each exp output
with 'the expected values recomputed from the deterministic stratified indices' and deletes mismatches.

Detection principle (deterministic reproduction):
  expected y = y_all[ridx] drawn with stratified_idx(y_all, 500)  (same function as a6)
  y_true of a head sample = y_all[:500] (class-sorted, so only the leading classes) → the vectors differ.

Usage:  python3 a6_audit.py [--variant sizectrl] [--n 500] [--dry]
  --dry : only print verdicts without deleting
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import csv
import json
import shutil
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
DATA = os.path.join(WORK, "01_dataset")
NPY = {"2dcnn", "yatc", "netmamba"}
TSV = {"etbert", "trafficformer"}


def stratified_idx(Y, n):
    """Same as a6_dlshap / a6_dlshap_uer (deterministic)."""
    Y = np.asarray(Y)
    n = min(n, len(Y))
    cls = {c: np.where(Y == c)[0] for c in np.unique(Y)}
    picked, i = [], 0
    while len(picked) < n:
        added = False
        for c in sorted(cls):
            if i < len(cls[c]):
                picked.append(int(cls[c][i]))
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        i += 1
    return np.array(sorted(picked))


def labels_of(model, ds, variant):
    d = os.path.join(DATA, ds, variant, model, "test_denoised")
    if model in NPY:
        return np.load(os.path.join(d, "y_data.npy"))
    ys = []
    with open(os.path.join(d, "data.tsv"), encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t", quotechar=None)
        header = next(rd)
        li = header.index("label")
        for row in rd:
            ys.append(int(row[li]))
    return np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="sizectrl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    root = os.path.join(str(RP.SHAP_DL), args.variant)
    if not os.path.isdir(root):
        print(f"[missing] {root}")
        return
    ok = bad = 0
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        for ds in sorted(os.listdir(mdir)):
            d = os.path.join(mdir, ds)
            y_all = labels_of(model, ds, args.variant)
            exp_y = y_all[stratified_idx(y_all, args.n)]
            # validate rows_idx itself
            rip = os.path.join(d, "rows_idx.npy")
            ridx_ok = (os.path.exists(rip)
                       and np.array_equal(np.load(rip),
                                          stratified_idx(y_all, args.n)))
            for e in (1, 3):
                ap_ = os.path.join(d, f"exp{e}_attr.npy")
                pp = os.path.join(d, f"exp{e}_pred.csv")
                if not os.path.exists(ap_):
                    continue
                good = ridx_ok and os.path.exists(pp)
                if good:
                    yt = []
                    with open(pp, encoding="utf-8") as f:
                        rd = csv.DictReader(f)
                        for r in rd:
                            yt.append(int(r["y_true"]))
                    good = np.array_equal(np.array(yt), exp_y[:len(yt)]) \
                        and len(yt) == len(exp_y)
                if good:
                    ok += 1
                else:
                    bad += 1
                    tag = "[deleted]" if not args.dry else "[contaminated]"
                    print(f"  {tag} {model}/{ds} exp{e} — "
                          f"{'rows_idx mismatch/missing' if not ridx_ok else 'pred y_true mismatch'}")
                    if not args.dry:
                        os.remove(ap_)
                        if os.path.exists(pp):
                            os.remove(pp)
    print(f"[audit] ok {ok} · contaminated {bad}"
          + ("  (dry — nothing deleted)" if args.dry else "  (contaminated files deleted — a rerun recomputes only those cells)"))


if __name__ == "__main__":
    main()
