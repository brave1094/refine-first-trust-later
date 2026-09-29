#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_mm4flow.py — MM4Flow (CCS 2025) input conversion.

2 modalities (INTEGRATION_SPEC.md):
  bytes : first 256B of the per-direction transport payload stream (fwd/bwd), lowercase hex
  ps    : direction-signed length sequence of packets with payload>0 (fwd +, bwd −), RLE encoded
Headers are excluded from the input, so the ip/port/l3 masks have no effect. l4_mask is also irrelevant to payload.
Direction  : sender of the first IP packet = fwd (originator convention)
ps lengths are clipped to the vocab limit (±1500).

※ To keep the same sample set across models, the original 'exclude sessions with <5 payload packets' (min_pkts)
  filter has been removed. Sessions without payload (control/handshake only) are still turned into samples
  (in that case ps='' , fwd_raw/bwd_raw='(empty)'). The drop condition is the same as for the other byte models:
  only 'when there is not a single parseable IP packet'.
Output: data.csv.gz (uid, ps, fwd_raw, bwd_raw, label) / files.csv
"""
import csv
import gzip
from pathlib import Path

from ..parser import dpkt_parser as dp

MAX_STREAM = 256      # per-direction payload byte cap
MAX_PS = 256          # ps token cap
PS_CLIP = 1500        # vocab coverage range ±1500


def _rle(vals):
    if not vals:
        return ""
    out, cur, cnt = [], vals[0], 1
    for v in vals[1:]:
        if v == cur:
            cnt += 1
        else:
            out.append(f"{cur}:{cnt}")
            cur, cnt = v, 1
    out.append(f"{cur}:{cnt}")
    return ",".join(out)


def make_sample(packets, meta, opt):
    if opt.get("parser", "dpkt") != "dpkt":
        raise NotImplementedError("mm4flow shaping currently supports dpkt only")

    c_key = None                               # originator = fwd (based on the first IP packet)
    ps_vals = []
    fwd_buf, bwd_buf = b"", b""
    any_ip = False
    for ts, buf, lt in packets:                # incremental parsing + early stop
        v = dp.parse_packet(ts, buf, lt)
        if v is None:
            continue
        any_ip = True
        if c_key is None:
            c_key = (v.src, v.sport)
        payload = v.raw[v.hdr_len:]
        plen = len(payload)
        if plen == 0:                          # packets without payload (ACK/handshake etc.)
            continue                           #   are not added to bytes/ps (session is kept)
        fwd = (v.src, v.sport) == c_key
        if len(ps_vals) < MAX_PS:
            val = min(plen, PS_CLIP)
            ps_vals.append(val if fwd else -val)
        if fwd and len(fwd_buf) < MAX_STREAM:
            fwd_buf += payload[:MAX_STREAM - len(fwd_buf)]
        elif not fwd and len(bwd_buf) < MAX_STREAM:
            bwd_buf += payload[:MAX_STREAM - len(bwd_buf)]
        # once all three buffers are full, later packets do not affect the result → stop
        if (len(ps_vals) >= MAX_PS and len(fwd_buf) >= MAX_STREAM
                and len(bwd_buf) >= MAX_STREAM):
            break

    if not any_ip:                             # no parseable IP packet (= same drop as other models)
        return None

    return {
        "ps": _rle(ps_vals),
        "fwd_raw": fwd_buf.hex() if fwd_buf else "(empty)",
        "bwd_raw": bwd_buf.hex() if bwd_buf else "(empty)",
    }


class Writer:
    def __init__(self, out_dir, opt):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.files = [], []

    def add(self, meta, sample, label_idx):
        uid = f"{meta['filename']}|{meta.get('session_id', '-')}"
        self.rows.append({"uid": uid, "ps": sample["ps"],
                          "fwd_raw": sample["fwd_raw"],
                          "bwd_raw": sample["bwd_raw"],
                          "label": meta["group_key"],
                          "label_idx": label_idx})
        self.files.append((meta["filename"], meta.get("session_id", "-"),
                           meta["group_key"]))

    def close(self):
        cols = ["uid", "ps", "fwd_raw", "bwd_raw", "label", "label_idx"]
        with gzip.open(self.out_dir / "data.csv.gz", "wt", newline="",
                       encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)
        with open(self.out_dir / "files.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "session_id", "group_key"])
            w.writerows(self.files)
        return len(self.rows)
