#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/own_models/02_xgboost/make_dataset.py
─────────────────────────────────────────────────────────────────────────────
Builds the session-statistics feature table for XGBoost.

No pcap re-parsing — directly reuses the statistics columns of the
04_session_noisy_labeled CSVs already extracted with tshark in stages 03/04.
(fastest, and the values are identical to the tshark output of 03 = most accurate)

Feature composition (edit only the constants below to swap the feature set):
  BASE_NUMERIC : CSV numeric columns as is
  DERIVED      : derived features built with vectorized ops (duration, rate, ratio ...)
  FLAG_COLS    : 0/1 flags
  Options      : --use-port (src/dst port), --use-l4 (tcp/udp one-hot)

Input :
  4 filelists    : 01_dataset/{dataset}/00_filelist{,_sm,_strat}/list_{train|test}_{noisy|denoised}.csv
  Feature source : <dataset_root>/04_session_noisy_labeled/session_stat_{dataset}_*_labeled.csv
Output : 01_dataset/{dataset}/{full|sizectrl|strat}/xgboost/
  x_{split}.npy          (N, F) float32
  y_{split}.npy          (N,)   int64
  feature_names.json     feature names (column order)
  stats.csv              sessions per split × class
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# Feature definitions — edit only here to change the feature set
# ═══════════════════════════════════════════════════════════════════════════
BASE_NUMERIC = [
    "pkt_count", "payload_size",
    "fwd_pkt", "fwd_byte", "bwd_pkt", "bwd_byte",
]

FLAG_COLS = [
    "has_SYN", "has_SYN_ACK", "has_FIN_cli", "has_FIN_serv",
    "has_TLS_CHLO", "has_TLS_SHLO",
]

# Derived features: name → (required columns, compute function)  ※ all pandas vectorized ops
_EPS = 1e-9

def _derived(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    dur = (df["ts_last"] - df["ts_first"]).clip(lower=0.0)
    out["duration"]       = dur
    out["pkt_rate"]       = df["pkt_count"]    / (dur + _EPS)
    out["byte_rate"]      = df["payload_size"] / (dur + _EPS)
    out["mean_pkt_size"]  = df["payload_size"] / (df["pkt_count"] + _EPS)
    tot_pkt  = df["fwd_pkt"]  + df["bwd_pkt"]
    tot_byte = df["fwd_byte"] + df["bwd_byte"]
    out["fwd_pkt_ratio"]  = df["fwd_pkt"]  / (tot_pkt + _EPS)
    out["fwd_byte_ratio"] = df["fwd_byte"] / (tot_byte + _EPS)
    out["fwd_bwd_pkt_ratio"]  = df["fwd_pkt"]  / (df["bwd_pkt"]  + 1.0)
    out["fwd_bwd_byte_ratio"] = df["fwd_byte"] / (df["bwd_byte"] + 1.0)
    return out

DERIVED_NAMES = ["duration", "pkt_rate", "byte_rate", "mean_pkt_size",
                 "fwd_pkt_ratio", "fwd_byte_ratio",
                 "fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio"]

PORT_COLS = ["src_port", "dst_port"]           # --use-port
L4_ONEHOT = ["is_tcp", "is_udp"]               # --use-l4

KEY_COLS  = ["filename", "session_id"]
TS_COLS   = ["ts_first", "ts_last"]
SPLIT_NAMES = ["train", "test"]


# ═══════════════════════════════════════════════════════════════════════════
# Internal utilities
# ═══════════════════════════════════════════════════════════════════════════
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _load_feature_table(labeled_dir: Path, dataset: str, use_port: bool,
                        use_l4: bool, chunk_rows: int = 500_000) -> pd.DataFrame:
    """04 CSV(s) → feature table indexed by (filename, session_id) (fully numeric)."""
    files = sorted(labeled_dir.glob(f"session_stat_{dataset}_*_labeled.csv"))
    if not files:
        raise FileNotFoundError(f"no labeled CSV: {labeled_dir}")

    need = KEY_COLS + BASE_NUMERIC + TS_COLS + FLAG_COLS
    if use_port:
        need += PORT_COLS
    if use_l4:
        need += ["L4"]
    header = list(pd.read_csv(files[0], nrows=0, encoding="utf-8-sig").columns)
    missing = [c for c in need if c not in header]
    if missing:
        raise KeyError(f"columns missing from labeled CSV: {missing}")

    parts = []
    for fp in files:
        for ch in pd.read_csv(fp, dtype=str, na_filter=False,
                              encoding="utf-8-sig", usecols=need,
                              chunksize=chunk_rows):
            t = pd.DataFrame(index=ch.index)
            t["filename"]   = ch["filename"]
            t["session_id"] = ch["session_id"]
            for c in BASE_NUMERIC + TS_COLS:
                t[c] = _to_num(ch[c])
            for c in FLAG_COLS:
                t[c] = (ch[c] == "1").astype(np.float32)
            if use_port:
                for c in PORT_COLS:
                    t[c] = _to_num(ch[c])
            if use_l4:
                l4 = ch["L4"].str.lower()
                t["is_tcp"] = (l4 == "tcp").astype(np.float32)
                t["is_udp"] = (l4 == "udp").astype(np.float32)
            parts.append(t)

    feat = pd.concat(parts, ignore_index=True)
    # Derived features (batch vectorized ops)
    der = _derived(feat)
    feat = pd.concat([feat, der], axis=1)
    feat = feat.drop_duplicates(subset=KEY_COLS, keep="first")
    feat = feat.set_index(KEY_COLS)
    return feat


def feature_names(use_port: bool, use_l4: bool) -> list:
    names = list(BASE_NUMERIC) + DERIVED_NAMES + list(FLAG_COLS)
    if use_port:
        names += PORT_COLS
    if use_l4:
        names += L4_ONEHOT
    return names


# ═══════════════════════════════════════════════════════════════════════════
# Router (06_make_dataset.py) entry point
# ═══════════════════════════════════════════════════════════════════════════
def build_all(*, dataset: str, task: str, mode: str, mode_dir: Path, out_dir: Path,
              label_map: dict, labeled_dir: Path, use_port: bool = False,
              use_l4: bool = False, **_):
    """
    Build train/test for one mode ({noisy|denoised}) (04 CSVs are read only once).
    Returns: summary dict of (n, miss) per split
    """
    mode_dir, out_dir = Path(mode_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = feature_names(use_port, use_l4)
    print(f"  [xgboost] features({len(names)}) = {names}")

    feat = _load_feature_table(Path(labeled_dir), dataset, use_port, use_l4)
    print(f"  [xgboost] feature table rows = {len(feat):,}")

    summary = {}
    stat_rows = []
    for split in SPLIT_NAMES:
        lst = pd.read_csv(mode_dir / f"list_{split}.csv", dtype=str,
                          na_filter=False, encoding="utf-8-sig")
        idx = pd.MultiIndex.from_frame(lst[KEY_COLS])
        found = idx.isin(feat.index)
        miss = int((~found).sum())
        sub = feat.loc[idx[found]]

        X = sub[names].to_numpy(dtype=np.float32)
        y = lst.loc[found, "group_key"].map(label_map).to_numpy(dtype=np.int64)

        np.save(out_dir / f"x_{split}.npy", X)
        np.save(out_dir / f"y_{split}.npy", y)
        summary[split] = (len(y), miss)
        print(f"  [xgboost] {split:15s} X={X.shape}  y={y.shape}  missing={miss}")

        for k, i in label_map.items():
            stat_rows.append((split, k, i, int((y == i).sum())))

    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    pd.DataFrame(stat_rows, columns=["split", "label_str", "label_id", "count"]
                 ).to_csv(out_dir / "stats.csv", index=False, encoding="utf-8-sig")
    return summary
