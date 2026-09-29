#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_2dcnn.py — Wang et al. 2017 (2D-CNN) input conversion.

Concatenates all packets of the session (from L3) → first 784 bytes → 28×28 image.
  - normalization: /256.0 (keeps the KNOM convention)
  - masking/padding value: 257 (257/256 after normalization) — same as KNOM 06_make_dataset_cnn.py
Output: x_data.npy (N,28,28) float32 / y_data.npy (N,) int64 / files.csv
"""
import csv
from pathlib import Path

import numpy as np

from ..parser import dpkt_parser as dp

N_BYTES = 784
MASK_VAL = 257
NORM_DIV = 256.0


def build_from_views(views, meta, opt):
    """Assemble 784B→28×28 from parsed views (shared multi-model path)."""
    if not views:
        return None
    buf = []
    for v in views:
        ints = dp.masked_ints(v, ip_mask=opt.get("ip_mask", False),
                              port_mask=opt.get("port_mask", False),
                              l3_mask=opt.get("l3_mask", False),
                              l4_mask=opt.get("l4_mask", False))
        buf.extend(MASK_VAL if b == dp.MASKED else b for b in ints)
        if len(buf) >= N_BYTES:
            break
    buf = buf[:N_BYTES]
    if len(buf) < N_BYTES:
        buf += [MASK_VAL] * (N_BYTES - len(buf))
    arr = np.asarray(buf, dtype=np.float32) / NORM_DIV
    return arr.reshape(28, 28)


def make_sample(packets, meta, opt):
    views = dp.parse_session(packets, opt.get("parser", "dpkt"),
                             max_bytes=N_BYTES)
    return build_from_views(views, meta, opt)


class Writer:
    def __init__(self, out_dir, opt):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.xs, self.ys, self.files = [], [], []

    def add(self, meta, sample, label_idx):
        self.xs.append(sample)
        self.ys.append(label_idx)
        self.files.append((meta["filename"], meta.get("session_id", "-"),
                           meta["group_key"]))

    def close(self):
        n = len(self.xs)
        x = np.stack(self.xs) if n else np.zeros((0, 28, 28), dtype=np.float32)
        y = np.asarray(self.ys, dtype=np.int64)
        np.save(self.out_dir / "x_data.npy", x)
        np.save(self.out_dir / "y_data.npy", y)
        with open(self.out_dir / "files.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "session_id", "group_key"])
            w.writerows(self.files)
        return n
