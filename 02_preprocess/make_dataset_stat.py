#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset_stat.py
─────────────────────────────────────────────────────────────────────────────
Summarizes per-dataset class distribution statistics in 4 stages (origin→refined→sampling→train/test)
and saves them as CSVs under 99_documents/results/dataset_stats/{dataset}/.

  01_origin_{ds}.csv     : class | count            (all sessions from 04 labeling, task3)
  02_refined_{ds}.csv    : row 1 = list of applied refinement rules,
                           class | count             (sessions remaining after the rules)
  03_sampling_n_{ds}.csv : row 1 = sampling n,
                           class | noisy | denoised  (after sampling at most n per class)
  04_train_test_{ds}.csv : class | train_noisy | train_denoised
                                 | test_noisy | test_denoised

- Class slots (rows) are 'all kept' based on 01_origin. Classes that disappear in refinement/sampling
  are shown as 0 in their slot.
- 01/02 are aggregated from the 04_session_noisy_labeled CSVs (all), 03/04 from 00_filelist.

Usage:
  python3 make_dataset_stat.py                 # all 10
  python3 make_dataset_stat.py --dataset vpn16 tor16
  python3 make_dataset_stat.py --n 2000        # sampling n to display (default 2000)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent          # …/02_preprocess/
WORK_DIR = CODE_DIR.parent                           # repository root
OUT_BASE = RP.DATASET_STATS
LIST_BASE = WORK_DIR / "01_dataset"

sys.path.insert(0, str(CODE_DIR))
from lib import datasets as ds                       # noqa: E402
try:
    from noise_rule import get_default_clean_rules
except Exception:
    def get_default_clean_rules(dataset):
        return []
try:
    # Per-dataset excluded class list from 05_make_filelist (case-insensitive)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "_mk05", CODE_DIR / "05_make_filelist.py")
    _m05 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m05)
    EXCLUDE_LABELS = getattr(_m05, "EXCLUDE_LABELS", {})
except Exception:
    EXCLUDE_LABELS = {}

TASK_COL = "task3"
ALL_DATASETS = ["vpn16", "tor16", "tls1.3", "cispec", 
                "ustc16", "cic17", "cic18", "iot23"]
CHUNK = 200_000


def _labeled_csvs(dataset):
    d = ds.labeled_dir(dataset)
    return sorted(glob.glob(str(d / f"session_stat_{dataset}_*_labeled.csv")))


def origin_refined_counts(dataset, clean_rules):
    """Reads the 04 labeling CSVs (all) in chunks and
       returns origin(class→count), refined(class→count).
       refined = sessions whose applied rule columns are all 0."""
    origin, refined = Counter(), Counter()
    files = _labeled_csvs(dataset)
    if not files:
        return origin, refined, []
    # Use only rule columns that actually exist in the header of the first file
    hdr = pd.read_csv(files[0], nrows=0, encoding="utf-8-sig").columns.tolist()
    rules = [r for r in clean_rules if r in hdr]
    usecols = [TASK_COL] + rules
    for f in files:
        for ch in pd.read_csv(f, usecols=usecols, chunksize=CHUNK,
                              encoding="utf-8-sig", dtype=str, na_filter=False):
            lab = ch[TASK_COL].astype(str)
            origin.update(lab.value_counts().to_dict())
            if rules:
                keep = (ch[rules].astype(int).sum(axis=1) == 0)
                refined.update(lab[keep].value_counts().to_dict())
            else:
                refined.update(lab.value_counts().to_dict())
    return origin, refined, rules


def filelist_counts(dataset):
    """Counts per group_key (class) from the 4 lists in 00_filelist."""
    fl = LIST_BASE / dataset / "00_filelist"
    out = {}
    for split in ("train_noisy", "train_denoised", "test_noisy", "test_denoised"):
        p = fl / f"list_{split.split('_')[0]}_{split.split('_')[1]}.csv"
        c = Counter()
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                c[str(r["group_key"])] += 1
        out[split] = c
    return out


def _write(path, header, rows, preface=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if preface is not None:
            w.writerow(preface)
        w.writerow(header)
        w.writerows(rows)


def gen(dataset, n):
    print(f"[{dataset}] aggregating...")
    clean = get_default_clean_rules(dataset)
    origin, refined, applied = origin_refined_counts(dataset, clean)
    if not origin:
        print(f"  [Skipped] no 04 labeling CSV: {ds.labeled_dir(dataset)}")
        return
    fc = filelist_counts(dataset)
    # Class order = by origin count, descending (fixed slots)
    classes = [k for k, _ in sorted(origin.items(), key=lambda x: -x[1])]
    out = OUT_BASE / dataset

    # Classes excluded from experiments in this dataset (05 EXCLUDE_LABELS, case-insensitive)
    excl = {v.lower() for v in EXCLUDE_LABELS.get(dataset, set())}
    excl_here = [c for c in classes if c.lower() in excl]

    # 00 List excluded classes (txt) — including why they were dropped
    lines = [f"# {dataset} classes excluded from experiments (05_make_filelist EXCLUDE_LABELS)",
             f"# excluded {len(excl_here)} / total {len(classes)}\n"]
    for c in excl_here:
        lines.append(f"{c}\torigin={origin.get(c,0)}  refined={refined.get(c,0)}  "
                     f"(excluded: too few in origin / untrainable after refinement)")
    if not excl_here:
        lines.append("(no excluded classes)")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"00_excluded_{dataset}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    # 01 origin (excluded classes are marked as excluded)
    _write(out / f"01_origin_{dataset}.csv", ["class", "count", "excluded"],
           [[c, origin.get(c, 0), "Y" if c in excl_here else ""] for c in classes])

    def ex(c):
        return "Y" if c in excl_here else ""

    # 02 refined (row 1: applied rules)
    _write(out / f"02_refined_{dataset}.csv", ["class", "count", "excluded"],
           [[c, refined.get(c, 0), ex(c)] for c in classes],
           preface=["applied_rules"] + applied)

    # 03 sampling_n (noisy=train_noisy+test_noisy, denoised=train_den+test_den)
    n_noisy = {c: fc["train_noisy"].get(c, 0) + fc["test_noisy"].get(c, 0)
               for c in classes}
    n_den = {c: fc["train_denoised"].get(c, 0) + fc["test_denoised"].get(c, 0)
             for c in classes}
    _write(out / f"03_sampling_n_{dataset}.csv",
           ["class", "noisy", "denoised", "excluded"],
           [[c, n_noisy[c], n_den[c], ex(c)] for c in classes],
           preface=["sampling_n", n])

    # 04 train/test
    _write(out / f"04_train_test_{dataset}.csv",
           ["class", "train_noisy", "train_denoised", "test_noisy",
            "test_denoised", "excluded"],
           [[c, fc["train_noisy"].get(c, 0), fc["train_denoised"].get(c, 0),
             fc["test_noisy"].get(c, 0), fc["test_denoised"].get(c, 0), ex(c)]
            for c in classes])

    print(f"  saved → {out}/  (classes {len(classes)}, excluded {len(excl_here)}, "
          f"origin {sum(origin.values()):,} → refined {sum(refined.values()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--n", type=int, default=2000,
                    help="sampling n to show in 03 (make_all_datasets default 2000)")
    args = ap.parse_args()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for d in args.dataset:
        gen(d, args.n)
    print("\n[done] → 99_documents/results/dataset_stats/{dataset}/0X_*.csv")


if __name__ == "__main__":
    main()
