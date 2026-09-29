#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_result_table.py — scans results/ and rebuilds result_table/{model}.csv.

train_common.update_result_table updates it incrementally during training, but
run this when you want to rebuild the whole table from the result CSVs.

  python3 02_result_table.py                 # all models
  python3 02_result_table.py --model xgboost

Table format: dataset | exp1 | exp2 | exp3 | exp4  (value = best test_accuracy)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import re
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
RESULTS = RP.MODEL_RESULTS
TABLE_DIR = RP.RESULT_TABLES

PAT = re.compile(r"^(?P<model>.+)_(?P<dataset>.+)_exp(?P<exp>[1-4])\.csv$")


def best_acc(csv_path: Path):
    # test-peeking removed: the test_accuracy of the 'last epoch' (max epoch), not max-over-epochs.
    #   host models (xgboost/rf) have only one row, so that row value is returned as is.
    try:
        df = pd.read_csv(csv_path)
        col = "test_accuracy"
        if col not in df.columns:
            return None
        d = df[["epoch", col]].copy()
        d["epoch"] = pd.to_numeric(d["epoch"], errors="coerce")
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=[col])
        if d.empty:
            return None
        return float(d.sort_values("epoch").iloc[-1][col])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="only a specific model (default all)")
    args = ap.parse_args()

    if not RESULTS.exists():
        print(f"results directory not found: {RESULTS}")
        return

    tables = {}          # model → {dataset → {expN: acc}}
    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        if args.model and model != args.model:
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for f in ds_dir.glob("*.csv"):
                m = PAT.match(f.name)
                if not m:
                    continue
                acc = best_acc(f)
                if acc is None:
                    continue
                tables.setdefault(model, {}).setdefault(
                    m["dataset"], {})[f"exp{m['exp']}"] = acc

    TABLE_DIR.mkdir(exist_ok=True)
    for model, per_ds in tables.items():
        rows = []
        for dataset in sorted(per_ds):
            row = {"dataset": dataset}
            for e in ("exp1", "exp2", "exp3", "exp4"):
                v = per_ds[dataset].get(e)
                row[e] = f"{v:.4f}" if v is not None else ""
            rows.append(row)
        out = TABLE_DIR / f"{model}.csv"
        pd.DataFrame(rows, columns=["dataset", "exp1", "exp2", "exp3", "exp4"]) \
            .to_csv(out, index=False, encoding="utf-8")
        print(f"[write] {out}  ({len(rows)} datasets)")

    if not tables:
        print("no results to aggregate")


if __name__ == "__main__":
    main()
