#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/lib/xgboost_featset.py
─────────────────────────────────────────────────────────────────────────────
XGBoost category combination (feat set) definitions — for category ablation.

Preprocessing (06 --feat a) saves all 1205 features to CSV.
Training (train_xgboost.py --featset NN) uses this module to pick 'which category columns to use'
and trains on only those CSV columns. (no need to rerun preprocessing)

5 categories:
  Header(30) Flag(9) Scalar1(945) Scalar2(21) Vector(200)  = 1205

31 combinations = 2^5 - 1 (minus the empty one). featset code = "01".."31".
  feat_a(original)  = all categories = same composition as featset "31".
"""
from itertools import product

CATEGORIES = ["Header", "Flag", "Scalar1", "Scalar2", "Vector"]
CATEGORY_SIZES = {"Header": 30, "Flag": 9, "Scalar1": 945, "Scalar2": 21, "Vector": 200}

# ── fixed name sets used for category classification ──────────────────────────────────────────
_HEADER_COLS = {
    "ip_ver_ihl", "ip_dsfield", "ip_total_len_hi", "ip_total_len_lo",
    "ip_id_hi", "ip_id_lo", "ip_flags_frag_hi", "ip_flags_frag_lo",
    "ip_ttl", "ip_checksum_hi", "ip_checksum_lo",
    "tcp_seq_b3", "tcp_seq_b2", "tcp_seq_b1", "tcp_seq_b0",
    "tcp_ack_b3", "tcp_ack_b2", "tcp_ack_b1", "tcp_ack_b0",
    "tcp_header_len", "tcp_window_hi", "tcp_window_lo",
    "tcp_checksum_hi", "tcp_checksum_lo", "tcp_urgptr_hi", "tcp_urgptr_lo",
    "udp_len_hi", "udp_len_lo", "udp_checksum_hi", "udp_checksum_lo",
}
_SCALAR2_COLS = {
    "Bps_Both", "Bps_Forward", "Bps_Backward",
    "Pps_Both", "Pps_Forward", "Pps_Backward",
    "Ratio_Packets_Forward", "Ratio_Packets_Backward", "Ratio_Packets_Bidir",
    "Ratio_Bytes_Forward", "Ratio_Bytes_Backward", "Ratio_Bytes_Bidir",
    "Packet_Count_Both", "Packet_Count_Forward", "Packet_Count_Backward",
    "Duration_Both", "Duration_Forward", "Duration_Backward",
    "Mean_Arrived_Both", "Mean_Arrived_Forward", "Mean_Arrived_Backward",
}


def category_of(col: str) -> str:
    """feature column name → category. (based on feat_a feature_columns rules)"""
    if col in _HEADER_COLS:
        return "Header"
    if col.startswith("flag_"):
        return "Flag"
    if col in _SCALAR2_COLS:
        return "Scalar2"
    if col.startswith("PSD_") or col.startswith("IAT_"):
        return "Vector"
    if col.startswith(("packet_sizes_", "iat_", "payload_")):
        return "Scalar1"
    return "UNKNOWN"


# ── build 31 combinations (all inclusion patterns of categories, except the empty one) ───────────────
def _build_featsets():
    """
    featset code → set of included categories.
    Codes are "01".."31". Sort order: number of included categories ascending, then CATEGORIES order.
    (5 with 1 → 10 with 2 → 10 with 3 → 5 with 4 → 1 with 5)
    """
    combos = []
    for bits in product([False, True], repeat=len(CATEGORIES)):
        included = [c for c, b in zip(CATEGORIES, bits) if b]
        if included:                      # skip the all-excluded (empty) combination
            combos.append(included)
    # sort by inclusion count → CATEGORIES index
    def sort_key(inc):
        idxs = tuple(CATEGORIES.index(c) for c in inc)
        return (len(inc), idxs)
    combos.sort(key=sort_key)
    return {f"{i+1:02d}": inc for i, inc in enumerate(combos)}


FEATSETS = _build_featsets()          # {"01": ["Header"], ..., "31": [all]}


def get_included_categories(featset: str) -> list:
    if featset not in FEATSETS:
        raise ValueError(f"unknown featset: {featset} "
                         f"(one of 01~{len(FEATSETS):02d})")
    return list(FEATSETS[featset])


def select_columns(featset: str, all_feat_cols: list) -> list:
    """Return only the columns in the categories included by featset (original order kept)."""
    inc = set(get_included_categories(featset))
    return [c for c in all_feat_cols if category_of(c) in inc]


def featset_size(featset: str) -> int:
    inc = get_included_categories(featset)
    return sum(CATEGORY_SIZES[c] for c in inc)


def all_featsets() -> list:
    return list(FEATSETS.keys())
