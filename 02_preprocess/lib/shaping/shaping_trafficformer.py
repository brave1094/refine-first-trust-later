#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shaping_trafficformer.py — TrafficFormer (S&P 2025) input conversion.

Based on the original finetuning_data_gen.py (INTEGRATION_SPEC.md):
  - first 5 packets, 64B per packet starting from the first IP header byte
  - overlapping byte-bigram(stride 1B) → 63 tokens, "[SEP] " prefix before each packet
  - anonymization (default on): random IP/port (direction-consistent), random TCP Timestamp (delta preserved),
    random TLS1.2 Hello gmt_unix_time. Seed based on session (filename) → deterministic.
  - RIFA augmentation: if opt["tf_enhance"]>1 and split_role=="train", for each copy
    re-randomize identifying fields to create factor samples (same delta-preserving principle).

※ To keep the sample set identical across models, the original filters were removed:
  - exclude <3-packet sessions  → removed (short sessions are sampled as-is, missing packets are padded)
  - exclude IPv6 sessions    → removed (IPv6 is parsed and sampled too. To drop IPv6, use noise_rule refinement
                        in step 05 for all at once — so every model sees the same sessions)
  Drop condition is the same as other byte models: only 'when there is no parseable IP packet'.
Output: data.tsv (label \t text_a, with header) / files.csv
"""
import csv
import random
import zlib
from pathlib import Path

from ..parser import dpkt_parser as dp

N_PKT = 5
SEL_LEN = 64


def _rand_ip(rng):
    return bytes(rng.randrange(256) for _ in range(4))


def _rewrite_identifiers(views, rng):
    """IP/port randomization (direction-consistent) + delta-preserving TCP TS randomization + TLS1.2 time randomization.
    Returns: list of modified bytearrays per packet (based on views[i].raw)."""
    if not views:
        return []
    # direction reference: first packet (src, sport) = client
    c_key = (views[0].src, views[0].sport)
    ip_a, ip_b = _rand_ip(rng), _rand_ip(rng)
    pa, pb = rng.randrange(1, 65536), rng.randrange(1, 65536)

    # TCP TS: per-direction offset (original first value → random base value)
    ts_base = {}

    out = []
    for v in views:
        raw = bytearray(v.raw)
        fwd = (v.src, v.sport) == c_key
        if v.ipver == 4:
            raw[12:16] = ip_a if fwd else ip_b
            raw[16:20] = ip_b if fwd else ip_a
        if v.port_span:
            off = v.port_span[0]
            sp = pa if fwd else pb
            dpo = pb if fwd else pa
            raw[off:off + 2] = sp.to_bytes(2, "big")
            raw[off + 2:off + 4] = dpo.to_bytes(2, "big")
        # walk TCP Timestamp option (kind=8, len=10)
        if v.proto == "tcp" and v.l4_hdr_len > 20:
            o = v.ip_hdr_len + 20
            end = v.ip_hdr_len + v.l4_hdr_len
            while o < end and o < len(raw):
                kind = raw[o]
                if kind == 0:
                    break
                if kind == 1:
                    o += 1
                    continue
                if o + 1 >= len(raw):
                    break
                length = raw[o + 1]
                if length < 2:
                    break
                if kind == 8 and o + 10 <= len(raw):
                    tsval = int.from_bytes(raw[o + 2:o + 6], "big")
                    dkey = ("f" if fwd else "b")
                    if dkey not in ts_base:
                        ts_base[dkey] = (tsval, rng.getrandbits(32))
                    orig0, rand0 = ts_base[dkey]
                    new = (rand0 + (tsval - orig0)) & 0xFFFFFFFF
                    raw[o + 2:o + 6] = new.to_bytes(4, "big")
                o += length
        # TLS1.2 ClientHello/ServerHello gmt_unix_time
        h = v.hdr_len
        pl = raw[h:]
        if (len(pl) >= 15 and pl[0] == 22 and pl[1] == 3
                and pl[5] in (1, 2)):
            raw[h + 11:h + 15] = rng.getrandbits(32).to_bytes(4, "big")
        out.append(raw)
    return out


def _to_text(pkt_bytes_list, mask_spans_list):
    parts = []
    for raw, spans in zip(pkt_bytes_list[:N_PKT], mask_spans_list[:N_PKT]):
        sel = list(raw[:SEL_LEN])
        for (s, e) in spans:
            for i in range(s, min(e, SEL_LEN)):
                sel[i] = 0
        h = "".join(f"{b:02x}" for b in sel)
        toks = [h[i * 2:i * 2 + 4] for i in range(len(h) // 2 - 1)]
        parts.append("[SEP] " + " ".join(toks))
    return " ".join(parts) + " "


def _mask_spans(v, opt):
    spans = []
    if opt.get("l3_mask", False):
        spans.append((0, v.ip_hdr_len))
    elif opt.get("ip_mask", False):
        spans.append(v.addr_span)
    if opt.get("l4_mask", False) and v.l4_hdr_len:
        spans.append((v.ip_hdr_len, v.ip_hdr_len + v.l4_hdr_len))
    elif opt.get("port_mask", False) and v.port_span:
        spans.append(v.port_span)
    return spans


def build_from_views(views, meta, opt):
    """Assemble TF text from parsed views (shared multi-model path). views needs at least 8 packets."""
    if not views:                                # no parseable IP packet (= same drop as other models)
        return None

    seed = zlib.crc32(f"{meta['filename']}|{meta.get('session_id','-')}"
                      .encode()) & 0xFFFFFFFF
    factor = int(opt.get("tf_enhance", 1) or 1)
    if opt.get("split_role") != "train":
        factor = 1
    randomize = opt.get("tf_randomize", True)

    spans = [_mask_spans(v, opt) for v in views]
    samples = []
    for k in range(factor):
        rng = random.Random(seed + k * 1_000_003)
        if randomize:
            pkt_bytes = _rewrite_identifiers(views, rng)
        else:
            pkt_bytes = [bytearray(v.raw) for v in views]
        samples.append(_to_text(pkt_bytes, spans))
    return samples if factor > 1 else samples[0]


def make_sample(packets, meta, opt):
    views = dp.parse_session(packets, opt.get("parser", "dpkt"), max_packets=8)
    return build_from_views(views, meta, opt)


class Writer:
    def __init__(self, out_dir, opt):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.files = [], []

    def add(self, meta, sample, label_idx):
        items = sample if isinstance(sample, list) else [sample]
        for s in items:
            self.rows.append((label_idx, s))
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
