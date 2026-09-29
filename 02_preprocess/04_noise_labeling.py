#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_noise_labeling.py
─────────────────────────────────
Reads session_stat CSVs and marks noise rules as 'one-hot columns per rule'.

  Common rules (noise_rule_all)   : 19 rules without class reference. Same marking for all datasets (all).
  Class rules (noise_rule_{ds})   : class-dependent (Eg_n: Cc_4/Cd_1 override, label errors, etc.) additions/replacements.

Marking = information only. Which sessions to drop (refinement) is decided downstream from this CSV.

Performance:
  - All marking is vectorized (pandas). Large CSVs are read by chunk streaming for memory safety.
  - tqdm progress bar (if installed). --workers N for 'per-file' parallelism.
    (large datasets are split into multiple files by the 1M split in 03, so file parallelism applies directly)

Output columns:
  [noise_label left columns] + [rule one-hot columns (table order, common+Eg merged)] + [noise_label right columns]

Input  : <dataset_dir>/03_session_stat/session_stat_{dataset}_*.csv
Output : <dataset_dir>/04_session_noisy_labeled/session_stat_{dataset}_*_labeled.csv

Usage:
  python3 04_noise_labeling.py --dataset vpn16
  python3 04_noise_labeling.py --dataset all --workers 8
  python3 04_noise_labeling.py --dataset cic17 cic18 --workers 6 --chunk-rows 500000
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                      # no-op wrapper if tqdm is missing
    def tqdm(it=None, **k):
        return it if it is not None else []

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
from noise_rule import get_common_rules, get_class_rules, has_class_rules, ALL_DATASETS
from noise_rule.base import merge_marks, marks_to_onehot
from lib import datasets as ds

INPUT_SUBDIR  = "03_session_stat"
OUTPUT_SUBDIR = "04_session_noisy_labeled"
NOISE_COL     = "noise_label"

# ── Per-dataset input subdirectory override ─────────────────────────────────
# cic17/cic18 do not use the raw 03_session_stat; instead the labeling step (01_cic{17,18}_labeling.py)
# fills task1/2/3 with Benign/Malware/... and the resulting 03_session_stat_labeled is used as input.
# (task in the raw 03_session_stat holds placeholders (file number/weekday/-) and must not be used as is)
INPUT_SUBDIR_OVERRIDE = {
    "cic17": "03_session_stat_labeled",
    "cic18": "03_session_stat_labeled",
}
DEFAULT_CHUNK = 500_000


# ── Process one file (chunk streaming + vectorized marking + incremental write) ──
def process_file(dataset: str, csv_in: str, csv_out: str,
                 chunk_rows: int, progress: bool = False) -> dict:
    common = get_common_rules()
    class_rules = get_class_rules(dataset)
    overrides = class_rules.overrides if class_rules else set()

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(csv_in, dtype=str, na_filter=False, chunksize=chunk_rows)
    if progress:
        reader = tqdm(reader, desc=Path(csv_in).name, unit="chunk", leave=False)

    first = True
    total = noisy = multi = 0
    counts: dict = {}
    order: list = []

    for chunk in reader:
        cols = list(chunk.columns)

        cm = list(common.mark(chunk))
        xm = list(class_rules.mark(chunk)) if class_rules else []
        marks = merge_marks(cm, xm, overrides)
        onehot = marks_to_onehot(marks, chunk.index)
        oh_df = pd.DataFrame(onehot, index=chunk.index)
        if not order:
            order = list(onehot.keys())

        if NOISE_COL in cols:
            i = cols.index(NOISE_COL)
            left, right = cols[:i], [c for c in cols[i + 1:] if c != "class_name"]
        else:
            anchor = "task3" if "task3" in cols else cols[min(4, len(cols) - 1)]
            i = cols.index(anchor) + 1
            left, right = cols[:i], [c for c in cols[i:] if c != "class_name"]

        out = pd.concat([chunk[left], oh_df, chunk[right]], axis=1)
        out.to_csv(csv_out, index=False, mode="w" if first else "a",
                   header=first, encoding="utf-8-sig" if first else "utf-8")

        n = len(chunk)
        total += n
        if order:
            ohi = oh_df.astype(int)
            noisy += int(ohi.any(axis=1).sum())
            multi += int((ohi.sum(axis=1) >= 2).sum())
            for code in onehot:
                counts[code] = counts.get(code, 0) + int(oh_df[code].sum())
        first = False

    return dict(dataset=dataset, csv_out=csv_out, total=total,
                noisy=noisy, multi=multi, counts=counts, order=order)


def _worker(job):
    dataset, ci, co, chunk_rows = job
    try:
        return process_file(dataset, ci, co, chunk_rows, progress=False)
    except Exception as e:                 # so that one failed file does not block the rest
        return dict(dataset=dataset, csv_out=co, error=f"{type(e).__name__}: {e}")


def format_report(r: dict) -> str:
    if "error" in r:
        return f"[ERROR] {Path(r['csv_out']).name}: {r['error']}"
    t = r["total"] or 1
    lines = [f"\n[Saved] {Path(r['csv_out']).name}",
             f"  Total sessions : {r['total']:,}",
             f"  Noisy sessions : {r['noisy']:,} ({r['noisy']/t*100:.1f}%)",
             f"  Clean sessions : {r['total']-r['noisy']:,} ({(r['total']-r['noisy'])/t*100:.1f}%)",
             f"  Multi-label    : {r['multi']:,}",
             f"  Rule columns   : {len(r['order'])}  [sessions per rule]"]
    for code in r["order"]:
        lines.append(f"    {code:8s} {r['counts'].get(code, 0):,}")
    return "\n".join(lines)


def collect_jobs(dataset: str, input_glob: str, chunk_rows: int,
                 input_subdir: str = None):
    dataset_root = Path(ds.root(dataset))
    # Input subdirectory: manual setting > dataset override > default (03_session_stat)
    subdir = (input_subdir or INPUT_SUBDIR_OVERRIDE.get(dataset, INPUT_SUBDIR))
    in_dir, out_dir = dataset_root / subdir, dataset_root / OUTPUT_SUBDIR
    if not in_dir.exists():
        print(f"[ERROR] {dataset}: input directory not found: {in_dir}  (skipped)")
        return []
    # Input file names are the same for the labeled and raw directories: session_stat_{dataset}_*.csv
    # (only the directory name differs, 03_session_stat_labeled; file names have no _labeled suffix)
    pattern = input_glob or f"session_stat_{dataset}_*.csv"
    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[ERROR] {dataset}: no matching CSV: {in_dir}/{pattern}  (skipped)")
        return []
    cls = (f"yes({get_class_rules(dataset).__class__.__name__})"
           if has_class_rules(dataset) else "none")
    print(f"  - {dataset:8s} files {len(files)} · input {subdir} · class rules {cls}")

    def _out_name(fname: str) -> str:
        # Output is always 04_session_noisy_labeled/session_stat_{dataset}_*_labeled.csv
        if fname.endswith("_labeled.csv"):
            return fname
        return fname.replace(".csv", "_labeled.csv")

    return [(dataset, str(f), str(out_dir / _out_name(f.name)), chunk_rows)
            for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="Datasets (multiple allowed, 'all'=all 10).")
    ap.add_argument("--input-glob", default=None,
                    help="Input CSV glob. Default: 'session_stat_{dataset}_*.csv'. "
                         "cic17/cic18 are read under the same name from 03_session_stat_labeled/.")
    ap.add_argument("--input-subdir", default=None,
                    help="Force the input subdirectory (e.g. 03_session_stat_labeled_cor_60). "
                         "Default: cic17/cic18=03_session_stat_labeled, others=03_session_stat")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of per-file parallel processes (default 1=sequential).")
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK,
                    help=f"Rows per streaming chunk (default {DEFAULT_CHUNK:,}).")
    args = ap.parse_args()

    requested = []
    for d in args.dataset:
        requested.extend(ALL_DATASETS if d == "all" else [d])
    seen = set()
    targets = [d for d in requested if not (d in seen or seen.add(d))]

    print(f"[Targets] {targets}  | workers={args.workers}  chunk={args.chunk_rows:,}")
    jobs = []
    for dataset in targets:
        if dataset == "all":
            continue
        jobs += collect_jobs(dataset, args.input_glob, args.chunk_rows,
                             args.input_subdir)

    if not jobs:
        print("No files to process.")
        return

    print(f"\nStarting to process {len(jobs)} files\n{'='*60}")

    if args.workers > 1 and len(jobs) > 1:
        with Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs),
                          desc="files", unit="file"):
                print(format_report(r))
    else:
        for job in jobs:
            r = process_file(job[0], job[1], job[2], job[3], progress=True)
            print(format_report(r))

    print(f"\n{'='*60}\n[Done] {len(jobs)} files\n")


if __name__ == "__main__":
    main()