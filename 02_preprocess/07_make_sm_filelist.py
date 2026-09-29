#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_make_sm_filelist.py  —  builds the sizectrl (size-controlled) filelist (self-contained within this tree)
─────────────────────────────────────────────────────────────────────────────
Purpose: to remove the training-set 'size confound' and compare den sessions vs noisy sessions purely,
      the noisy list is subsampled (fixed seed) to match the 'per-class denoised count'.

★ This script reads 00_filelist, the 05 (full) output, and creates 00_filelist_sm.
  (inside the same tree, different subfolder → no overwriting. Removes the dependency on the ablation tree.)

Operation (per dataset):  01_dataset/{ds}/00_filelist/ →  01_dataset/{ds}/00_filelist_sm/
  - list_train_denoised.csv, list_test_denoised.csv, label_map.json → copied as is
  - list_train_noisy.csv → subsampled per class to the 'train_denoised count'
  - list_test_noisy.csv  → subsampled per class to the 'test_denoised count'
      (classes with 0 denoised are excluded = same coverage as den)
  ※ The test-matched copy may not be used in experiments (Illusion 4 uses the full noisy_test), but it is created anyway.
    → safe, since the full variant (00_filelist) keeps the original noisy_test intact.

Then build the datasets:  06_make_dataset.py --variant sizectrl  (or this script with --build)

Usage:
  python3 07_make_sm_filelist.py                         # all 10, seed 42 (filelist only)
  python3 07_make_sm_filelist.py --dataset ustc16 cic17
  python3 07_make_sm_filelist.py --dataset ustc16 --build --workers 8   # including build (7 models)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent          # …/02_preprocess/
WORK_DIR = CODE_DIR.parent                           # …/(this tree)/
DS_BASE = WORK_DIR / "01_dataset"

ALL = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]
BYTE = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
COPY_AS_IS = ["list_train_denoised.csv", "list_test_denoised.csv", "label_map.json"]


def size_match(noisy_csv: Path, den_csv: Path, seed: int) -> pd.DataFrame:
    """Subsample the noisy list per class to the denoised count (deterministic)."""
    den = pd.read_csv(den_csv, encoding="utf-8-sig")
    noi = pd.read_csv(noisy_csv, encoding="utf-8-sig")
    cap = den.groupby("group_key").size().to_dict()      # denoised count per class
    rng = np.random.default_rng(seed)
    keep = []
    for gk, grp in noi.groupby("group_key", sort=True):
        n = int(cap.get(gk, 0))                           # denoised 0 → excluded
        if n <= 0:
            continue
        idx = grp.sort_values(["filename", "session_id"],
                              kind="mergesort").index.to_numpy()
        idx = idx[rng.permutation(len(idx))][:n]          # if short, take what is available
        keep.extend(idx.tolist())
    return noi.loc[noi.index.isin(keep)].reset_index(drop=True), noi, cap


def make_sm(dsn: str, seed: int) -> bool:
    src = DS_BASE / dsn / "00_filelist"
    dst = DS_BASE / dsn / "00_filelist_sm"
    if not (src / "list_train_noisy.csv").exists():
        print(f"[SKIP] {dsn}: 00_filelist not found ({src}) — run 05 first"); return False
    dst.mkdir(parents=True, exist_ok=True)
    for f in COPY_AS_IS:
        if (src / f).exists():
            shutil.copyfile(src / f, dst / f)
    parts = []
    for split in ("train", "test"):
        sm, noi, cap = size_match(src / f"list_{split}_noisy.csv",
                                  src / f"list_{split}_denoised.csv", seed)
        sm.to_csv(dst / f"list_{split}_noisy.csv", index=False, encoding="utf-8-sig")
        parts.append(f"{split}: noisy {len(noi):,}→{len(sm):,} "
                     f"(den size {sum(cap.values()):,})")
    print(f"[{dsn}] {' | '.join(parts)}  → {dst}")
    return True


def build(dsn: str, workers: int):
    """Build 7 models with 06_make_dataset.py --variant sizectrl (ip/port masking for the 5 byte models)."""
    six = CODE_DIR / "06_make_dataset.py"
    def run(model, extra):
        cmd = [sys.executable, str(six), "--dataset", dsn, "--model", model,
               "--variant", "sizectrl", "--workers", str(workers)] + extra
        print(f"  >> 06 {model} (sizectrl)")
        subprocess.run(cmd, check=False)
    run("xgboost", [])
    run("rf", [])
    for m in BYTE:
        run(m, ["--ip_mask", "--port_mask"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=ALL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--build", action="store_true",
                    help="after the filelist, also build the 7-model datasets with 06 --variant sizectrl")
    ap.add_argument("--workers", type=int, default=8, help="06 workers for --build")
    args = ap.parse_args()

    print(f"[tree] {DS_BASE}\n[seed] {args.seed}\n{'='*66}")
    for dsn in args.dataset:
        if make_sm(dsn, args.seed) and args.build:
            build(dsn, args.workers)
    print(f"{'='*66}\n[done] size-matched filelist → 00_filelist_sm"
          + ("  (+dataset build)" if args.build else "  (build via 06 --variant sizectrl)"))


if __name__ == "__main__":
    main()
