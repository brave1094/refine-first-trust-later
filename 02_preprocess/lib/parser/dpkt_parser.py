#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/parser/dpkt_parser.py
─────────────────────────────────────────────────────────────────────────────
dpkt-based single-packet parser.

  parse_packet(ts, buf, linktype) → PktView | None
    - Strips L2 (Ethernet/SLL/loopback/RawIP) and builds a byte view starting at L3 (IP)
    - Supports both IPv4 / IPv6 (including IPv6 extension-header chain walk)
    - Non-IP packets (ARP, etc.) -> None

  PktView fields:
    raw        : original bytes from L3 (no mask applied)
    ts         : timestamp
    ipver      : 4 | 6
    proto      : "tcp" | "udp" | "other"
    ip_hdr_len : IP header length (IPv6 includes extension headers)
    l4_hdr_len : L4 header length (TCP=data offset*4, UDP=8, other=0)
    addr_span  : (start, end) — contiguous src+dst address span
    port_span  : (start, end) — src+dst port 4 bytes (tcp/udp only, else None)
    src, dst, sport, dport : metadata (values before masking)

  masked_ints(view, ip_mask, port_mask, l3_mask, l4_mask) → list[int]
    - Masked positions are -1 (sentinel). Each shaping module replaces them with its own MASK value.
    - Relations: l3_mask ⊃ ip_mask (addresses), l4_mask ⊃ port_mask.

  header_payload_split(view) → (header_bytes, payload_bytes)
    - header = IP header + L4 header, payload = L4 payload (shared by the YaTC/NetMamba specs)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import dpkt

# linktype constants
DLT_EN10MB = 1
DLT_RAW_A = 12
DLT_RAW_B = 101
DLT_NULL = 0
DLT_LOOP = 108
DLT_LINUX_SLL = 113

_IPV6_EXT = {0, 43, 44, 50, 51, 60, 135}   # hop-by-hop, routing, fragment, ESP, AH, dst-opts, mobility


@dataclass
class PktView:
    raw: bytes
    ts: float
    ipver: int
    proto: str
    ip_hdr_len: int
    l4_hdr_len: int
    addr_span: tuple
    port_span: Optional[tuple]
    src: str
    dst: str
    sport: int
    dport: int

    @property
    def hdr_len(self) -> int:
        """Total length of IP header + L4 header (payload start offset)"""
        return self.ip_hdr_len + self.l4_hdr_len


def _strip_l2(buf: bytes, linktype: int) -> Optional[bytes]:
    """Strip L2 header → bytes from L3 (IP). None for non-IP."""
    try:
        if linktype == DLT_EN10MB:
            eth = dpkt.ethernet.Ethernet(buf)
            # dpkt strips VLAN etc. IP payload position = len(buf) - len(ip_bytes)
            if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return bytes(eth.data)
            return None
        if linktype == DLT_LINUX_SLL:
            sll = dpkt.sll.SLL(buf)
            if isinstance(sll.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
                return bytes(sll.data)
            return None
        if linktype in (DLT_NULL, DLT_LOOP):
            return buf[4:] if len(buf) > 4 else None
        if linktype in (DLT_RAW_A, DLT_RAW_B):
            return buf
        # Unknown linktype → guess IP from the first nibble
        if buf and (buf[0] >> 4) in (4, 6):
            return buf
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
            return bytes(eth.data)
        return None
    except Exception:
        return None


def _ipv6_walk(raw: bytes):
    """Walk the IPv6 extension-header chain → (l4_offset, l4_proto). On failure (40, nxt)."""
    if len(raw) < 40:
        return len(raw), -1
    nxt = raw[6]
    off = 40
    while nxt in _IPV6_EXT:
        if nxt in (50,):                       # ESP: cannot parse further
            return off, nxt
        if off + 2 > len(raw):
            return off, nxt
        if nxt == 44:                          # fragment: fixed 8 bytes
            ext_len = 8
        elif nxt == 51:                        # AH: (len+2)*4
            ext_len = (raw[off + 1] + 2) * 4
        else:
            ext_len = (raw[off + 1] + 1) * 8
        nxt = raw[off]
        off += ext_len
        if off > len(raw):
            return len(raw), nxt
    return off, nxt


def _ip_str(b: bytes) -> str:
    import socket
    try:
        if len(b) == 4:
            return socket.inet_ntop(socket.AF_INET, b)
        return socket.inet_ntop(socket.AF_INET6, b)
    except Exception:
        return b.hex()


def parse_packet(ts: float, buf: bytes, linktype: int) -> Optional[PktView]:
    raw = _strip_l2(buf, linktype)
    if raw is None or len(raw) < 20:
        return None

    ver = raw[0] >> 4
    try:
        if ver == 4:
            ihl = (raw[0] & 0x0F) * 4
            if ihl < 20 or len(raw) < ihl:
                return None
            l4p = raw[9]
            ip_hdr_len = ihl
            addr_span = (12, 20)
            src, dst = _ip_str(raw[12:16]), _ip_str(raw[16:20])
        elif ver == 6:
            if len(raw) < 40:
                return None
            l4_off, l4p = _ipv6_walk(raw)
            ip_hdr_len = l4_off
            addr_span = (8, 40)
            src, dst = _ip_str(raw[8:24]), _ip_str(raw[24:40])
        else:
            return None
    except Exception:
        return None

    sport = dport = 0
    port_span = None
    if l4p == 6 and len(raw) >= ip_hdr_len + 20:
        proto = "tcp"
        off = ip_hdr_len
        l4_hdr_len = ((raw[off + 12] >> 4) & 0x0F) * 4
        l4_hdr_len = max(20, min(l4_hdr_len, len(raw) - off))
        sport = int.from_bytes(raw[off:off + 2], "big")
        dport = int.from_bytes(raw[off + 2:off + 4], "big")
        port_span = (off, off + 4)
    elif l4p == 17 and len(raw) >= ip_hdr_len + 8:
        proto = "udp"
        off = ip_hdr_len
        l4_hdr_len = 8
        sport = int.from_bytes(raw[off:off + 2], "big")
        dport = int.from_bytes(raw[off + 2:off + 4], "big")
        port_span = (off, off + 4)
    else:
        proto = "other"
        l4_hdr_len = 0

    return PktView(raw=raw, ts=ts, ipver=ver, proto=proto,
                   ip_hdr_len=ip_hdr_len, l4_hdr_len=l4_hdr_len,
                   addr_span=addr_span, port_span=port_span,
                   src=src, dst=dst, sport=sport, dport=dport)


MASKED = -1


def masked_ints(view: PktView, ip_mask=False, port_mask=False,
                l3_mask=False, l4_mask=False) -> list:
    """Bytes from L3 as an int list. Masked positions are MASKED(-1)."""
    ints = list(view.raw)
    n = len(ints)

    def _fill(s, e):
        for i in range(max(0, s), min(e, n)):
            ints[i] = MASKED

    if l3_mask:
        _fill(0, view.ip_hdr_len)
    elif ip_mask:
        _fill(*view.addr_span)
    if l4_mask and view.l4_hdr_len:
        _fill(view.ip_hdr_len, view.ip_hdr_len + view.l4_hdr_len)
    elif port_mask and view.port_span:
        _fill(*view.port_span)
    return ints


def header_payload_split(view: PktView):
    """(IP header+L4 header bytes, L4 payload bytes) — split shared by YaTC/NetMamba."""
    h = view.hdr_len
    return view.raw[:h], view.raw[h:]


def masked_header_payload(view: PktView, **mask_kw):
    """After masking: (header ints, payload ints). Masked positions are -1."""
    ints = masked_ints(view, **mask_kw)
    h = view.hdr_len
    return ints[:h], ints[h:]


def mask_wrap_to_ethernet(in_path: str, out_path: str, ip_mask=False,
                          port_mask=False, l3_mask=False, l4_mask=False) -> int:
    """Session pcap → (same masked_ints masking as the byte models) + Ethernet-wrapped pcap.
    For NetFound input: extract data with the 'same parser and same masking' as other models, then set only L2 to Ethernet.
    Returns: number of (IP) packets written. Non-IP packets are excluded by parse_session (same criterion as the byte models)."""
    import struct as _st
    with open(in_path, "rb") as f:
        rdr = dpkt.pcap.Reader(f)
        try:
            lt = rdr.datalink()
        except Exception:
            lt = DLT_EN10MB
        pkts = [(ts, bytes(buf), lt) for ts, buf in rdr]
    views = parse_session(pkts)                       # strip L2 + IP-only (same as the byte models)
    n = 0
    with open(out_path, "wb") as f:
        w = dpkt.pcap.Writer(f, linktype=DLT_EN10MB)
        for v in views:
            ints = masked_ints(v, ip_mask=ip_mask, port_mask=port_mask,
                               l3_mask=l3_mask, l4_mask=l4_mask)
            l3 = bytes((0 if b == MASKED else (b & 0xFF)) for b in ints)
            etype = 0x86DD if v.ipver == 6 else 0x0800
            frame = b"\x00" * 12 + _st.pack(">H", etype) + l3   # dummy MAC + ethertype + L3
            w.writepkt(frame, ts=v.ts)
            n += 1
    return n


def parse_session(packets, parser_name: str = "dpkt",
                  max_packets: int = 0, max_bytes: int = 0):
    """(ts, buf, linktype) list → PktView list (non-IP packets excluded).

    parser_name : "dpkt" (implemented) / "tshark" (reserved — not implemented)
    max_packets : if >0, stop after parsing this many valid packets (e.g. 5 packets for yatc-like models)
    max_bytes   : if >0, stop once cumulative L3 bytes exceed this value (e.g. 784B for 2dcnn)
    Early-stop hints to avoid fully parsing long sessions (thousands of packets).
    """
    if parser_name != "dpkt":
        raise NotImplementedError(
            f"--parser {parser_name} is not implemented yet (only dpkt is currently supported)")
    views = []
    acc = 0
    for ts, buf, lt in packets:
        v = parse_packet(ts, buf, lt)
        if v is None:
            continue
        views.append(v)
        acc += len(v.raw)
        if max_packets and len(views) >= max_packets:
            break
        if max_bytes and acc >= max_bytes:
            break
    return views
