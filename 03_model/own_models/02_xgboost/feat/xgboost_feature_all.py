#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/own_models/02_xgboost/feat/xgboost_feature_a.py
─────────────────────────────────────────────────────────────────────────────
feat "a" — feature definition following the ParkEunHyeok feature.py (1211 features) approach.

extractor.py reads the session pcap with dpkt as (ts,buf,linktype) and calls extract_features(packets) of this module
per session. One session pcap = one flow; a single feature dict is returned.

Feature composition (1,211 total):
  Header    30 : IP 11 + TCP 15 + UDP 4 byte (based on the first packet)
  Flag       9 : occurrence count per TCP flag type
  Scalar1  945 : feature(3) × direction(3) × packet_num(7) × stat(15)
  Scalar2   21 : Bps/Pps/Ratio/Count/Duration/Mean × direction
  Vector   200 : PSD 100 + IAT 100
  ─────────────
  Total    1,205  (+ meta is attached separately by the extractor as the common 8 columns)

Note: meta (Label/ip/port/protocol) is not added here. The extractor
prepends the common 8 columns (Label|filename|Stream_num|protocol|srcip|srcport|dstip|dstport),
so the feature dict does not duplicate ip/port/protocol.
"""
from __future__ import annotations

import warnings

import numpy as np
import dpkt
from scipy.stats import skew, kurtosis

# suppress scipy skew/kurtosis precision-loss warnings (occur on near-constant sessions, harmless)
warnings.filterwarnings("ignore", message="Precision loss occurred")

FEAT_NAME = "a"
MISSING = "-"

# Scalar1 settings
PACKET_NUMS = ["All", 5, 10, 15, 20, 25, 30]
VECTOR_LEN  = 100
FEATURES = [
    ("packet_sizes", "packet_sizes"),
    ("iat",          "both_start_interval_time"),
    ("payload",      "payload"),
]
STAT_NAMES = [
    "Sum", "Max", "Min", "Mean", "Variance",
    "Pop_Std", "Sample_Std", "Geo_Mean",
    "Q25", "Q50", "Q75", "IQR", "Range", "Skewness", "Kurtosis",
]
DIR_NAMES = {0: "Forward", 1: "Backward", 2: "Both"}
FLAG_TYPES = ["S", "SA", "A", "FA", "F", "R", "PA", "FPA", "RA"]


# ═══════════════ STEP 1: packets → flow dictionary (dpkt parsing) ═══════════════
def _decap_gre(ip):
    """If the IP proto is GRE(47), strip one more layer and return the inner IP (v4/v6).
    Targets the inner packets of GRE-encapsulated sessions such as Mirai GRE-Flood (GREIP/GREETH)
    for feature extraction. Returned unchanged if not GRE."""
    try:
        proto = getattr(ip, "p", None)
        if proto is None:
            proto = getattr(ip, "nxt", None)
        if proto != 47:                          # 47 = GRE
            return ip
        gre = ip.data
        inner = getattr(gre, "data", None)
        if isinstance(inner, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return inner
        if isinstance(inner, (bytes, bytearray)) and inner:
            etype = getattr(gre, "p", 0x0800)
            if etype == 0x86dd:
                return dpkt.ip6.IP6(inner)
            if etype == 0x6558:                  # Transparent Ethernet Bridging (GREETH)
                d = dpkt.ethernet.Ethernet(inner).data
                return d if isinstance(d, (dpkt.ip.IP, dpkt.ip6.IP6)) else None
            ver = (inner[0] >> 4) & 0xF
            return dpkt.ip6.IP6(inner) if ver == 6 else dpkt.ip.IP(inner)
    except Exception:
        return None
    return ip


def _unpack_ip(ts, buf, linktype):
    """(ts, buf, linktype) → dpkt IP(v4) or IP6 object or None.
    Strips the link layer (Ethernet / Raw IP / Linux SLL), distinguished by datalink, to get IP.
    Handles both IPv4/IPv6 (including VPN/tunnel raw IP). GRE (proto 47) is stripped to the inner."""
    def _raw_ip(b):
        # raw IP: upper 4 bits of the first byte are the version (4 or 6)
        if not b:
            return None
        ver = (b[0] >> 4) & 0xF
        if ver == 6:
            return dpkt.ip6.IP6(b)
        return dpkt.ip.IP(b)
    try:
        if linktype == dpkt.pcap.DLT_EN10MB:            # Ethernet
            ip = dpkt.ethernet.Ethernet(buf).data
        elif linktype == dpkt.pcap.DLT_LINUX_SLL:       # Linux cooked
            ip = dpkt.sll.SLL(buf).data
        elif linktype in (dpkt.pcap.DLT_RAW, 12, 14):   # Raw IP
            ip = _raw_ip(buf)
        elif linktype == dpkt.pcap.DLT_NULL:            # BSD loopback
            ip = dpkt.loopback.Loopback(buf).data
        else:
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    ip = _raw_ip(buf)
            except Exception:
                ip = _raw_ip(buf)
        if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return None
        ip = _decap_gre(ip)                             # if GRE, strip to inner
        return ip if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)) else None
    except Exception:
        return None


def _ip_str(addr) -> str:
    import socket
    try:
        if len(addr) == 16:                             # IPv6
            return socket.inet_ntop(socket.AF_INET6, addr)
        return socket.inet_ntoa(addr)                   # IPv4
    except Exception:
        return str(addr)


# TCP flag bits → scapy notation string mapping (F,S,R,P,A,U,E,C)
_FLAG_BITS = [
    (dpkt.tcp.TH_FIN, "F"), (dpkt.tcp.TH_SYN, "S"), (dpkt.tcp.TH_RST, "R"),
    (dpkt.tcp.TH_PUSH, "P"), (dpkt.tcp.TH_ACK, "A"), (dpkt.tcp.TH_URG, "U"),
    (dpkt.tcp.TH_ECE, "E"), (dpkt.tcp.TH_CWR, "C"),
]


def _flag_str(tcp) -> str:
    """dpkt TCP → scapy-style flag string (e.g. 'SA','PA','FPA').
    str(flags) in scapy combines in F,S,R,P,A,U,E,C order. Generated in the same order."""
    fl = tcp.flags
    return "".join(ch for bit, ch in _FLAG_BITS if fl & bit)


def process_packet(packets: list) -> dict:
    """list of (ts, buf, linktype) tuples → 5-tuple flow dictionary (Forward +, Backward -).
    Parses IP/TCP/UDP with dpkt. (produces the same flow structure as the former scapy version)"""
    flows = {}
    for pkt in packets:
        if len(pkt) == 3:
            ts, buf, linktype = pkt
        else:                                    # (ts, buf) → assume Ethernet
            ts, buf = pkt
            linktype = dpkt.pcap.DLT_EN10MB
        ip = _unpack_ip(ts, buf, linktype)
        if ip is None:
            continue
        is_v6 = isinstance(ip, dpkt.ip6.IP6)
        l4 = ip.data
        is_tcp = isinstance(l4, dpkt.tcp.TCP)
        is_udp = isinstance(l4, dpkt.udp.UDP)
        if not (is_tcp or is_udp):
            continue

        # proto: IPv4=ip.p, IPv6=ip.nxt (last next-header). dpkt parses extension headers
        # automatically, so proto is determined by the l4 type (6=TCP,17=UDP).
        proto = 6 if is_tcp else 17
        src_ip, dst_ip = _ip_str(ip.src), _ip_str(ip.dst)
        src_port, dst_port = l4.sport, l4.dport

        # payload length: scapy uses len of packet.payload.payload.payload (=L4 payload).
        payload_len = len(l4.data) if l4.data else 0
        pkt_len  = len(buf)                       # full frame length (same as scapy len(packet))
        pkt_time = float(ts)

        flow_key    = (src_ip, src_port, dst_ip, dst_port, proto)
        reverse_key = (dst_ip, dst_port, src_ip, src_port, proto)
        flags = _flag_str(l4) if is_tcp else "n"

        if flow_key in flows:
            f = flows[flow_key]
            f["forward_packets"] += 1
            f["packet_sizes"].append(pkt_len)
            f["payload"].append(payload_len)
            f["packet_num"].append(f["forward_packets"])
            f["end_time"] = pkt_time
            f["tcp_flags"].append(flags)
            f["packet_start_time_f"].append(pkt_time)
            f["packet_start_time"].append(pkt_time)
            fwd_times = f["packet_start_time_f"]
            f["start_interval_time"].append(
                pkt_time - fwd_times[-2] if len(fwd_times) >= 2 else 0.0)
            both_times = f["packet_start_time"]
            f["both_start_interval_time"].append(
                pkt_time - both_times[-2] if len(both_times) >= 2 else 0.0)

        elif reverse_key in flows:
            f = flows[reverse_key]
            f["backward_packets"] += 1
            f["packet_sizes"].append(-pkt_len)
            f["payload"].append(-payload_len)
            f["packet_num"].append(-f["backward_packets"])
            f["end_time"] = pkt_time
            f["tcp_flags"].append(flags)
            bwd_times = f["packet_start_time_r"]
            if len(bwd_times) >= 1:
                f["start_interval_time"].append(-(pkt_time - bwd_times[-1]))
            f["packet_start_time_r"].append(pkt_time)
            both_times = f["packet_start_time"]
            f["packet_start_time"].append(pkt_time)
            f["both_start_interval_time"].append(
                -(pkt_time - both_times[-2]) if len(both_times) >= 2 else 0.0)

        else:
            tcp_layer = l4 if is_tcp else None
            udp_layer = l4 if is_udp else None
            if is_v6:
                # ── map IPv6 header onto IPv4 feature slots ──
                #   ip_total_len : IPv6 payload_len + 40 (fixed header) = IPv4 total_len meaning
                #   ip_dsfield   : IPv6 traffic class (= IPv4 tos)
                #   ip_ttl       : IPv6 hop limit    (= IPv4 ttl)
                #   ip_id / frag / checksum : absent in IPv6 → -1 (IPv6 marker)
                ip_total_len = int(getattr(ip, "plen", 0)) + 40
                ip_id, ip_frag, ip_checksum = -1, -1, -1
                ip_ver = 6
                ip_ihl = 0                       # IPv6 has no IHL concept (fixed 40B)
                ip_tos = int(getattr(ip, "fc", 0))
                ip_ttl = int(getattr(ip, "hlim", 0))
            else:
                # ── IPv4 ──
                ip_total_len = ip.len
                # scapy ip.frag = fragment offset (lower 13 bits) only. dpkt _flags_offset is
                # flags+offset 16 bits → take only the offset with & 0x1FFF to match scapy.
                _fo = getattr(ip, "_flags_offset", 0)
                ip_id, ip_frag, ip_checksum = ip.id, _fo & 0x1FFF, ip.sum
                ip_ver = (ip.v_hl >> 4) & 0xF if hasattr(ip, "v_hl") else ip.v
                ip_ihl = ip.v_hl & 0xF if hasattr(ip, "v_hl") else ip.hl
                ip_tos = ip.tos
                ip_ttl = ip.ttl
            tcp_seq    = tcp_layer.seq    if tcp_layer else 0
            tcp_ack    = tcp_layer.ack    if tcp_layer else 0
            tcp_window = tcp_layer.win    if tcp_layer else 0
            tcp_chksum = tcp_layer.sum    if tcp_layer else 0
            tcp_urgptr = tcp_layer.urp    if tcp_layer else 0
            udp_len    = udp_layer.ulen   if udp_layer else 0
            udp_chksum = udp_layer.sum    if udp_layer else 0
            # tcp dataofs (header len in 32-bit words)
            tcp_dataofs = (tcp_layer.off if tcp_layer else 0)

            # -1 (IPv6-absent marker) sets both hi/lo to -1.
            def _hi(v):
                return -1 if v < 0 else (v >> 8) & 0xFF
            def _lo(v):
                return -1 if v < 0 else v & 0xFF

            header = {
                "ip_ver_ihl":       (ip_ver << 4) | ip_ihl,
                "ip_dsfield":        ip_tos,
                "ip_total_len_hi":   _hi(ip_total_len),
                "ip_total_len_lo":   _lo(ip_total_len),
                "ip_id_hi":          _hi(ip_id),
                "ip_id_lo":          _lo(ip_id),
                "ip_flags_frag_hi":  _hi(ip_frag),
                "ip_flags_frag_lo":  _lo(ip_frag),
                "ip_ttl":            ip_ttl,
                "ip_checksum_hi":    _hi(ip_checksum),
                "ip_checksum_lo":    _lo(ip_checksum),
                "tcp_seq_b3":       (tcp_seq >> 24) & 0xFF,
                "tcp_seq_b2":       (tcp_seq >> 16) & 0xFF,
                "tcp_seq_b1":       (tcp_seq >> 8)  & 0xFF,
                "tcp_seq_b0":        tcp_seq & 0xFF,
                "tcp_ack_b3":       (tcp_ack >> 24) & 0xFF,
                "tcp_ack_b2":       (tcp_ack >> 16) & 0xFF,
                "tcp_ack_b1":       (tcp_ack >> 8)  & 0xFF,
                "tcp_ack_b0":        tcp_ack & 0xFF,
                "tcp_header_len":    tcp_dataofs,
                "tcp_window_hi":    (tcp_window >> 8) & 0xFF,
                "tcp_window_lo":     tcp_window & 0xFF,
                "tcp_checksum_hi":  (tcp_chksum >> 8) & 0xFF,
                "tcp_checksum_lo":   tcp_chksum & 0xFF,
                "tcp_urgptr_hi":    (tcp_urgptr >> 8) & 0xFF,
                "tcp_urgptr_lo":     tcp_urgptr & 0xFF,
                "udp_len_hi":       (udp_len >> 8) & 0xFF,
                "udp_len_lo":        udp_len & 0xFF,
                "udp_checksum_hi":  (udp_chksum >> 8) & 0xFF,
                "udp_checksum_lo":   udp_chksum & 0xFF,
            }
            flows[flow_key] = {
                "src_ip": src_ip, "src_port": src_port,
                "dst_ip": dst_ip, "dst_port": dst_port, "protocol": proto,
                "start_time": pkt_time, "end_time": pkt_time,
                "forward_packets": 1, "backward_packets": 0,
                "packet_sizes": [pkt_len], "payload": [payload_len],
                "packet_num": [1],
                "packet_start_time_f": [pkt_time], "packet_start_time_r": [],
                "packet_start_time": [pkt_time],
                "start_interval_time": [], "both_start_interval_time": [],
                "tcp_flags": [flags], "header": header,
            }
    return flows


# ═══════════════ statistics (feature.py compute_stats) ═══════════════
def compute_stats(arr: np.ndarray) -> dict:
    if len(arr) == 0 or (len(arr) == 1 and arr[0] == 0):
        return {name: MISSING for name in STAT_NAMES}
    safe = np.where(arr > 0, arr, 1.0)
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    stats = {
        "Sum":        float(arr.sum()),
        "Max":        float(arr.max()),
        "Min":        float(arr.min()),
        "Mean":       float(arr.mean()),
        "Variance":   float(np.var(arr)),
        "Pop_Std":    float(np.std(arr)),
        "Sample_Std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "Geo_Mean":   float(np.exp(np.mean(np.log(safe)))),
        "Q25": float(q25), "Q50": float(q50), "Q75": float(q75),
        "IQR": float(q75 - q25), "Range": float(arr.max() - arr.min()),
        "Skewness": float(skew(arr)), "Kurtosis": float(kurtosis(arr, fisher=True)),
    }
    return {k: (MISSING if (np.isnan(v) or np.isinf(v)) else v)
            for k, v in stats.items()}


def safe_div(a, b):
    return float(a / b) if b != 0 else MISSING


# ═══════════════ Scalar1 (945) ═══════════════
def _scalar1(flow: dict) -> dict:
    row = {}
    for col_name, dict_key in FEATURES:
        raw = np.array(flow[dict_key], dtype=float)
        dir_arrays = {2: np.abs(raw), 0: raw[raw > 0], 1: np.abs(raw[raw < 0])}
        for direction in (2, 0, 1):
            dir_name = DIR_NAMES[direction]
            full = dir_arrays[direction]
            if len(full) == 0:
                full = np.array([0.0])
            n = len(full)
            stats_cache = {}
            for pnum in PACKET_NUMS:
                eff = n if (pnum == "All" or pnum >= n) else pnum
                stats = stats_cache.get(eff)
                if stats is None:
                    arr = full if eff == n else full[:eff]
                    stats = compute_stats(arr)
                    stats_cache[eff] = stats
                prefix = f"{col_name}_{dir_name}_{pnum}"
                for stat_name, val in stats.items():
                    row[f"{prefix}_{stat_name}"] = val
    return row


# ═══════════════ Scalar2 (21) ═══════════════
def _scalar2(flow: dict) -> dict:
    row = {}
    times_all = np.array(flow["packet_start_time"], dtype=float)
    times_fwd = np.array(flow["packet_start_time_f"], dtype=float)
    times_bwd = np.array(flow["packet_start_time_r"], dtype=float)
    sizes_all = np.abs(np.array(flow["packet_sizes"], dtype=float))
    sizes_fwd = np.array([s for s in flow["packet_sizes"] if s > 0], dtype=float)
    sizes_bwd = np.abs(np.array([s for s in flow["packet_sizes"] if s < 0], dtype=float))
    cnt_all, cnt_fwd, cnt_bwd = len(sizes_all), len(sizes_fwd), len(sizes_bwd)
    dur_all = float(times_all[-1] - times_all[0]) if len(times_all) > 1 else 0.0
    dur_fwd = float(times_fwd[-1] - times_fwd[0]) if len(times_fwd) > 1 else 0.0
    dur_bwd = float(times_bwd[-1] - times_bwd[0]) if len(times_bwd) > 1 else 0.0

    row["Bps_Both"]     = safe_div(sizes_all.sum(), dur_all)
    row["Bps_Forward"]  = safe_div(sizes_fwd.sum(), dur_fwd)
    row["Bps_Backward"] = safe_div(sizes_bwd.sum(), dur_bwd)
    row["Pps_Both"]     = safe_div(cnt_all, dur_all)
    row["Pps_Forward"]  = safe_div(cnt_fwd, dur_fwd)
    row["Pps_Backward"] = safe_div(cnt_bwd, dur_bwd)
    row["Ratio_Packets_Forward"]  = safe_div(cnt_fwd, cnt_all)
    row["Ratio_Packets_Backward"] = safe_div(cnt_bwd, cnt_all)
    row["Ratio_Packets_Bidir"]    = safe_div(cnt_fwd, cnt_bwd)
    sum_all, sum_fwd, sum_bwd = sizes_all.sum(), sizes_fwd.sum(), sizes_bwd.sum()
    row["Ratio_Bytes_Forward"]  = safe_div(sum_fwd, sum_all)
    row["Ratio_Bytes_Backward"] = safe_div(sum_bwd, sum_all)
    row["Ratio_Bytes_Bidir"]    = safe_div(sum_fwd, sum_bwd)
    row["Packet_Count_Both"]     = cnt_all
    row["Packet_Count_Forward"]  = cnt_fwd
    row["Packet_Count_Backward"] = cnt_bwd
    row["Duration_Both"]     = dur_all
    row["Duration_Forward"]  = dur_fwd
    row["Duration_Backward"] = dur_bwd
    row["Mean_Arrived_Both"]     = safe_div(dur_all, cnt_all)
    row["Mean_Arrived_Forward"]  = safe_div(dur_fwd, cnt_fwd)
    row["Mean_Arrived_Backward"] = safe_div(dur_bwd, cnt_bwd)
    return row


# ═══════════════ Vector (200) ═══════════════
def _vector(flow: dict) -> dict:
    psd = np.abs(np.array(flow["packet_sizes"], dtype=float))[:VECTOR_LEN]
    psd_padded = np.zeros(VECTOR_LEN); psd_padded[:len(psd)] = psd
    iat = np.abs(np.array(flow["both_start_interval_time"], dtype=float))[:VECTOR_LEN]
    iat_padded = np.zeros(VECTOR_LEN); iat_padded[:len(iat)] = iat
    row = {f"PSD_{i}": psd_padded[i] for i in range(VECTOR_LEN)}
    row.update({f"IAT_{i}": iat_padded[i] for i in range(VECTOR_LEN)})
    return row


# ═══════════════ Flag (9) ═══════════════
def _flag(flow: dict) -> dict:
    flags = flow.get("tcp_flags", [])
    return {f"flag_{f}": flags.count(f) for f in FLAG_TYPES}


# ═══════════════ column order (deterministic, used by the extractor as the sort key) ═══════════════
def feature_columns() -> list:
    cols = []
    # Header 30
    header_keys = [
        "ip_ver_ihl", "ip_dsfield", "ip_total_len_hi", "ip_total_len_lo",
        "ip_id_hi", "ip_id_lo", "ip_flags_frag_hi", "ip_flags_frag_lo",
        "ip_ttl", "ip_checksum_hi", "ip_checksum_lo",
        "tcp_seq_b3", "tcp_seq_b2", "tcp_seq_b1", "tcp_seq_b0",
        "tcp_ack_b3", "tcp_ack_b2", "tcp_ack_b1", "tcp_ack_b0",
        "tcp_header_len", "tcp_window_hi", "tcp_window_lo",
        "tcp_checksum_hi", "tcp_checksum_lo", "tcp_urgptr_hi", "tcp_urgptr_lo",
        "udp_len_hi", "udp_len_lo", "udp_checksum_hi", "udp_checksum_lo",
    ]
    cols += header_keys
    # Flag 9
    cols += [f"flag_{f}" for f in FLAG_TYPES]
    # Scalar1 945
    for col_name, _ in FEATURES:
        for direction in (2, 0, 1):
            dir_name = DIR_NAMES[direction]
            for pnum in PACKET_NUMS:
                for stat_name in STAT_NAMES:
                    cols.append(f"{col_name}_{dir_name}_{pnum}_{stat_name}")
    # Scalar2 21
    cols += [
        "Bps_Both", "Bps_Forward", "Bps_Backward",
        "Pps_Both", "Pps_Forward", "Pps_Backward",
        "Ratio_Packets_Forward", "Ratio_Packets_Backward", "Ratio_Packets_Bidir",
        "Ratio_Bytes_Forward", "Ratio_Bytes_Backward", "Ratio_Bytes_Bidir",
        "Packet_Count_Both", "Packet_Count_Forward", "Packet_Count_Backward",
        "Duration_Both", "Duration_Forward", "Duration_Backward",
        "Mean_Arrived_Both", "Mean_Arrived_Forward", "Mean_Arrived_Backward",
    ]
    # Vector 200
    cols += [f"PSD_{i}" for i in range(VECTOR_LEN)]
    cols += [f"IAT_{i}" for i in range(VECTOR_LEN)]
    return cols


# ═══════════════ feature categories (8) ═══════════════
# only_{cat} / no_{cat} files filter columns using this classification.
CATEGORIES = ["header", "tcpflags", "pktsize", "payload",
              "iat_stat", "psd", "iat_vec", "flowscalar"]


def category_of(col: str) -> str:
    """feature column name → one of the 8 categories."""
    if col.startswith("PSD_"):
        return "psd"
    if col.startswith("IAT_"):
        return "iat_vec"
    if col.startswith("flag_"):
        return "tcpflags"
    if col.startswith("ip_") or col.startswith("tcp_") or col.startswith("udp_"):
        return "header"
    if col.startswith("packet_sizes"):
        return "pktsize"
    if col.startswith("iat_"):
        return "iat_stat"
    if col.startswith("payload"):
        return "payload"
    return "flowscalar"    # Bps/Pps/Ratio/Count/Duration/Mean_Arrived


def select_columns(only=None, exclude=None) -> list:
    """Category filter over the all columns.
      only=[cat,...]   → only those categories
      exclude=[cat,...]→ exclude those categories
    """
    cols = feature_columns()
    if only is not None:
        only = set(only)
        return [c for c in cols if category_of(c) in only]
    if exclude is not None:
        exclude = set(exclude)
        return [c for c in cols if category_of(c) not in exclude]
    return cols


# ═══════════════ extractor entry point ═══════════════
def extract_features(packets: list) -> dict | None:
    """
    packet list of a session pcap → 1 feature dict (+ extra info).
    A session is treated as one flow. If multiple flows are found, the flow with the most packets is used.
    Returns: {"features": {...}, "meta": {src_ip, src_port, dst_ip, dst_port, protocol}}
          None if the flow is empty.
    """
    flows = process_packet(packets)
    if not flows:
        return None
    # a session pcap is in principle one flow. If there are several, pick the flow with the most packets.
    def _npkt(fl):
        return fl["forward_packets"] + fl["backward_packets"]
    flow = max(flows.values(), key=_npkt)

    feats = {}
    feats.update(flow["header"])
    feats.update(_flag(flow))
    feats.update(_scalar1(flow))
    feats.update(_scalar2(flow))
    feats.update(_vector(flow))

    meta = {
        "src_ip":   flow["src_ip"],   "src_port": flow["src_port"],
        "dst_ip":   flow["dst_ip"],   "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
    }
    return {"features": feats, "meta": meta}