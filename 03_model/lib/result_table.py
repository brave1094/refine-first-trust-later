#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/lib/result_table.py
─────────────────────────────────────────────────────────────────────────────
Utility to update the unified XGBoost performance table (separate CSV for noisy / denoised).

Table format (2-row header):
  Dataset  Task |   Accuracy(tuned) |   F1-score(tuned)
  {mode}   best | Accuracy F1-score | Accuracy F1-score
  vpn16    2    |   ...      ...    |   ...     ...
  ...

- rows = (dataset, task) combinations
- left 2 columns  = (accuracy, f1) of the best model tuned on acc (--optuna_bas acc)
- right 2 columns = (accuracy, f1) of the best model tuned on f1 (--optuna_bas f1)
- files           = 01_xgboost_noisy.csv / 01_xgboost_denoised.csv (per mode)

update_cell(csv_path, mode, dataset, task, basis, acc, f1):
  basis="acc" → written to the left 2 columns, basis="f1" → the right 2 columns.
  If a cell already has a value, update 'only when the basis metric is higher'.
"""
import csv
from pathlib import Path

# (dataset, task) row order
ROWS = [
    ("vpn16", "2"), ("vpn16", "3"),
    ("tor16", "2"), ("tor16", "3"),
    ("tls1.3", "3"),
    ("cispec", "3"),
    
    ("ustc16", "1"), ("ustc16", "3"),
    ("cic17", "1"), ("cic17", "3"),
    ("cic18", "1"), ("cic18", "3"),
    ("iot23", "1"), ("iot23", "2"), ("iot23", "3"),
]

# 9 value columns: (tuning basis, metric)  basis/metric = acc / f1 / bal
BASES = ["acc", "f1", "bal"]
METRICS = ["acc", "f1", "bal"]
VALUE_COLS = [(b, m) for b in BASES for m in METRICS]
_BASIS_LABEL = {"acc": "Accuracy", "f1": "F1-score", "bal": "BalancedAcc"}
_METRIC_LABEL = {"acc": "Accuracy", "f1": "F1-score", "bal": "BalancedAcc"}


def _empty_grid():
    return {rc: {vc: "" for vc in VALUE_COLS} for rc in ROWS}


def _normalize_task(task: str) -> str:
    return str(task).strip().lower().replace("task", "")


def _read(csv_path: Path):
    grid = _empty_grid()
    if not csv_path.exists():
        return grid
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        return grid
    for r in rows[2:]:
        if len(r) < 2 + len(VALUE_COLS):
            continue
        ds = r[0].strip()
        task = _normalize_task(r[1])
        key = (ds, task)
        if key not in grid:
            continue
        for i, vc in enumerate(VALUE_COLS):
            grid[key][vc] = r[2 + i].strip()
    return grid


def _write(csv_path: Path, grid: dict, mode: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    # Row 1: tuning-basis group header (each basis spans 3 columns)
    row1 = ["Dataset", "Task"]
    for b in BASES:
        row1 += [_BASIS_LABEL[b] + "(tuned)", "", ""]
    out.append(row1)
    # Row 2: 3 metrics under each basis
    row2 = [mode, "best"]
    for b in BASES:
        row2 += [_METRIC_LABEL[m] for m in METRICS]
    out.append(row2)
    # Data rows
    for (ds, task) in ROWS:
        cell = grid[(ds, task)]
        out.append([ds, task] + [cell[vc] for vc in VALUE_COLS])
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(out)


def update_cell(csv_path, mode, dataset, task, basis, acc, f1, bal, decimals=4):
    """
    Write (acc, f1, bal) into the 3 columns of basis (acc/f1/bal) tuning in the (dataset, task) row.
    If a cell already has a value, update only when the basis metric is higher.
    Returns: 'written' | 'updated' | 'kept'
    """
    csv_path = Path(csv_path)
    task = _normalize_task(task)
    key = (dataset, task)
    if key not in _empty_grid():
        raise ValueError(f"combination not in table: {dataset}/task{task}")
    if basis not in BASES:
        raise ValueError(f"basis must be acc/f1/bal: {basis}")

    grid = _read(csv_path)
    vals = {"acc": round(float(acc), decimals),
            "f1":  round(float(f1),  decimals),
            "bal": round(float(bal), decimals)}
    new_val = vals[basis]                          # representative metric of this basis

    cur = grid[key][(basis, basis)]
    if cur == "":
        status = "written"
    else:
        try:
            old_val = float(cur)
        except ValueError:
            old_val = float("-inf")
        if new_val > old_val:
            status = "updated"
        else:
            return "kept"

    for m in METRICS:
        grid[key][(basis, m)] = f"{vals[m]:.{decimals}f}"
    _write(csv_path, grid, mode)
    return status