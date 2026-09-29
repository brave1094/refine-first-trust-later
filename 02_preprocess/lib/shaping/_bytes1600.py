#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_bytes1600.py — shared 1600-byte (MFR-style) builder for YaTC / NetMamba.

Both models: first 5 packets × (header 80B + payload 240B) = 1600B → (40,40) uint8.
  header  = IP header(+options) + L4 header(+options)
  payload = L4 payload
  shortfall padded with 0x00; masked positions are also 0x00.
The difference is on the training side (model architecture/pre-training); the input representation is identical (see each INTEGRATION_SPEC.md).
"""
import csv
from pathlib import Path

import numpy as np

from ..parser import dpkt_parser as dp

N_PKT = 5
HDR_LEN = 80
PAY_LEN = 240
PKT_LEN = HDR_LEN + PAY_LEN     # 320
TOTAL = N_PKT * PKT_LEN         # 1600


def _crop_pad(ints, n):
    """int list(-1=masked) → uint8 list of length n (masked/padding=0)."""
    out = [0 if b == dp.MASKED else b for b in ints[:n]]
    if len(out) < n:
        out += [0] * (n - len(out))
    return out


def build_mfr_from_views(views, meta, opt):
    """Assemble MFR from already-parsed views (multi-model path: parse_session shared once)."""
    if not views:
        return None
    buf = []
    for v in views[:N_PKT]:
        hdr, pay = dp.masked_header_payload(
            v, ip_mask=opt.get("ip_mask", False),
            port_mask=opt.get("port_mask", False),
            l3_mask=opt.get("l3_mask", False),
            l4_mask=opt.get("l4_mask", False))
        buf += _crop_pad(hdr, HDR_LEN)
        buf += _crop_pad(pay, PAY_LEN)
    if len(buf) < TOTAL:
        buf += [0] * (TOTAL - len(buf))
    return np.asarray(buf, dtype=np.uint8).reshape(40, 40)


def make_mfr(packets, meta, opt):
    views = dp.parse_session(packets, opt.get("parser", "dpkt"),
                             max_packets=N_PKT)
    return build_mfr_from_views(views, meta, opt)


class MfrWriter:
    """x_data.npy (N,40,40) uint8 / y_data.npy int64 / files.csv"""

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
        x = np.stack(self.xs) if n else np.zeros((0, 40, 40), dtype=np.uint8)
        y = np.asarray(self.ys, dtype=np.int64)
        np.save(self.out_dir / "x_data.npy", x)
        np.save(self.out_dir / "y_data.npy", y)
        with open(self.out_dir / "files.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "session_id", "group_key"])
            w.writerows(self.files)
        return n
