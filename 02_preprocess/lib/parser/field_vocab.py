#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lib/parser/field_vocab.py — field name↔id vocab (0=<pad>). tshark field names as-is."""
import csv

PAD_FIELD = "<pad>"
SEP_FIELD = "<sep>"                 # position of the [SEP] token in trafficformer/etbert
MASK_SUFFIX = "#masked"            # masked fields are marked after the name (ip.src#masked)
SII_FIELDS = {
    "ip.src", "ip.dst", "ipv6.src", "ipv6.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport",
}


class FieldVocab:
    def __init__(self):
        self.name2id = {PAD_FIELD: 0}
        self.id2name = [PAD_FIELD]

    def add(self, name):
        if name not in self.name2id:
            self.name2id[name] = len(self.id2name)
            self.id2name.append(name)
        return self.name2id[name]

    def __len__(self):
        return len(self.id2name)

    def save(self, path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["field_id", "field"])
            for i, n in enumerate(self.id2name):
                w.writerow([i, n])

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, encoding="utf-8-sig") as f:
            rd = csv.reader(f); next(rd, None)
            for row in rd:
                if len(row) >= 2 and row[1] != PAD_FIELD:
                    v.add(row[1])
        return v


# ── 98 (SII fields.py) scheme: k-th byte (1-based) within 'a run of the same field in the same packet' ──
#    field-k labels (sni-1, sni-2 …) are derived at analysis time from field_id.npy + pkt_id.npy.
def byte_index(fid_row, pkt_row, pos):
    q = pos
    while q > 0 and fid_row[q - 1] == fid_row[pos] and pkt_row[q - 1] == pkt_row[pos]:
        q -= 1
    return pos - q + 1


def byte_run_matrix(fid_chunk, pkt_chunk):
    """(n,L) field_id/pkt_id → per-position byte_idx (1-based) matrix (vectorized)."""
    import numpy as np
    n, L = fid_chunk.shape
    idx = np.arange(L)
    new = np.ones((n, L), dtype=bool)
    new[:, 1:] = ((fid_chunk[:, 1:] != fid_chunk[:, :-1]) |
                  (pkt_chunk[:, 1:] != pkt_chunk[:, :-1]))
    start = np.where(new, idx[None, :], 0)
    start = np.maximum.accumulate(start, axis=1)
    return idx[None, :] - start + 1
