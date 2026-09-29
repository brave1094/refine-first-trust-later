# -*- coding: utf-8 -*-
"""Extract per-class value distribution summaries of Unconditional header features (for the 5.3 extension).

Input : 01_dataset/{ds}/sizectrl/xgboost/{train,test}_denoised/features.csv (E1 refined)
Target: ip_checksum, tcp_checksum, udp_checksum, ip_id, ip_flags_frag (hi*256+lo, 16bit)
       ip_ver_ihl (single byte)
Output: 99_documents/results/analysis/07_uncond_dist/{ds}.npz
       - hist_{field}: (n_class, 256) histogram (16bit: bins 256 values wide, 1B: raw values)
       - classes: array of class names, counts: sessions per class
Run   : cd <repository root> && python 04_analysis/05c_uncond_dist.py
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import json
import os

import numpy as np
import pandas as pd

BASE = os.environ.get("SCIE_WORK", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
OUT = os.path.join(str(RP.ANALYSIS_OUT), "07_uncond_dist")
os.makedirs(OUT, exist_ok=True)

DS = ["vpn16", "tor16", "tls1.3", "cispec", 
      "ustc16", "cic17", "cic18", "iot23"]
PAIRS = {"ip_checksum": ("ip_checksum_hi", "ip_checksum_lo"),
         "tcp_checksum": ("tcp_checksum_hi", "tcp_checksum_lo"),
         "udp_checksum": ("udp_checksum_hi", "udp_checksum_lo"),
         "ip_id": ("ip_id_hi", "ip_id_lo"),
         "ip_flags_frag": ("ip_flags_frag_hi", "ip_flags_frag_lo")}
SINGLE = ["ip_ver_ihl"]
USE = ["Label"] + [c for p in PAIRS.values() for c in p] + SINGLE

for ds in DS:
    if os.path.exists(os.path.join(OUT, f"{ds}.npz")):
        print(f"  [skip-done] {ds}")
        continue
    parts = []
    for sp in ("train_denoised", "test_denoised"):
        p = f"{BASE}/01_dataset/{ds}/sizectrl/xgboost/{sp}/features.csv"
        if not os.path.exists(p):
            print(f"  [skip] {p}")
            continue
        parts.append(pd.read_csv(p, usecols=USE, low_memory=False))
    df = pd.concat(parts, ignore_index=True)
    classes = sorted(df["Label"].astype(str).unique())
    cidx = {c: i for i, c in enumerate(classes)}
    ci = df["Label"].astype(str).map(cidx).to_numpy()
    out = {"classes": np.array(classes),
           "counts": np.bincount(ci, minlength=len(classes))}
    for name, (hi, lo) in PAIRS.items():
        v = (pd.to_numeric(df[hi], errors="coerce").to_numpy(dtype=float) * 256
             + pd.to_numeric(df[lo], errors="coerce").to_numpy(dtype=float))
        ok = np.isfinite(v)
        h = np.zeros((len(classes), 256), dtype=np.int64)
        b = np.clip((v[ok] / 256).astype(int), 0, 255)
        np.add.at(h, (ci[ok], b), 1)
        out[f"hist_{name}"] = h
    for name in SINGLE:
        v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(v)
        h = np.zeros((len(classes), 256), dtype=np.int64)
        b = np.clip(v[ok].astype(int), 0, 255)
        np.add.at(h, (ci[ok], b), 1)
        out[f"hist_{name}"] = h
    np.savez_compressed(os.path.join(OUT, f"{ds}.npz"), **out)
    print(f"  [ok] {ds}: {len(df):,} sessions · {len(classes)} classes")

print("[done] ->", OUT)
