#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_etbert.py — ET-BERT (WWW 2022) input conversion.

Concatenates all packets of the session (L3→L4→payload order = raw bytes from L3) and
generates an overlapping byte-bigram (4-hex-digit) token sequence.
  - MAX_TOKENS=128, PAD_TOKEN="0000"
  - masked bytes are replaced with "00"
Output: data.tsv (label \t text_a, with header) / files.csv
"""
import csv
from pathlib import Path

from ..parser import dpkt_parser as dp

MAX_TOKENS = 128
PAD_TOKEN = "0000"


def build_from_views(views, meta, opt):
    """Assemble 128 bigram tokens from parsed views (shared multi-model path)."""
    if not views:
        return None
    byte_list = []
    for v in views:
        ints = dp.masked_ints(v, ip_mask=opt.get("ip_mask", False),
                              port_mask=opt.get("port_mask", False),
                              l3_mask=opt.get("l3_mask", False),
                              l4_mask=opt.get("l4_mask", False))
        byte_list.extend("00" if b == dp.MASKED else f"{b:02x}" for b in ints)
        if len(byte_list) > MAX_TOKENS + 1:
            break

    if len(byte_list) < 2:
        bigrams = [PAD_TOKEN] * MAX_TOKENS
    else:
        bigrams = [byte_list[i] + byte_list[i + 1]
                   for i in range(len(byte_list) - 1)]
    if len(bigrams) >= MAX_TOKENS:
        bigrams = bigrams[:MAX_TOKENS]
    else:
        bigrams += [PAD_TOKEN] * (MAX_TOKENS - len(bigrams))
    return " ".join(bigrams)


def make_sample(packets, meta, opt):
    views = dp.parse_session(packets, opt.get("parser", "dpkt"),
                             max_bytes=2 * MAX_TOKENS)
    return build_from_views(views, meta, opt)


class Writer:
    def __init__(self, out_dir, opt):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.files = [], []

    def add(self, meta, sample, label_idx):
        self.rows.append((label_idx, sample))
        self.files.append((meta["filename"], meta.get("session_id", "-"),
                           meta["group_key"]))

    def close(self):
        with open(self.out_dir / "data.tsv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["label", "text_a"])
            w.writerows(self.rows)
        with open(self.out_dir / "files.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "session_id", "group_key"])
            w.writerows(self.files)
        return len(self.rows)
