#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/parser/tshark_pdml.py — session pcap → ParsedPacket (tshark PDML, per-byte field names)
─────────────────────────────────────────────────────────────────────────────
Parser dedicated to Illusion 5-B (byte→field). Based on SII_debias/traffic_explainer/lib/pcap_io.py.
  · Assigns each byte the 'most specific' field name decoded by tshark
    (ip.src, ip.id, tcp.window_size, tls.handshake.extensions_server_name, dns.qry.name …)
  · Accurate decoding up to L7 + port→protocol Decode As support (not possible with dpkt)

ParsedPacket
  raw    : bytes       # raw frame (as in the pcap)
  labels : list[str]   # same length as raw; most specific field name for each byte
  ip_pos : int         # L3 start (= L2 strip point). Consistent with the L3 bytes of dpkt shaping
  pay_pos: int         # L7 start (len(raw) if none)
  is_ack_only : bool

Decode As / heuristics:
  run_tshark_pdml(path, decode_as=[("tcp.port==8443","tls"), ...], tshark_opts=["-o","tls.desegment_ssl_records:TRUE"])
  → tshark -d tcp.port==8443,tls ...  (forced decoding of non-standard ports)
"""
from collections import namedtuple
from pathlib import Path
import shutil, subprocess

import dpkt

ParsedPacket = namedtuple("ParsedPacket",
                          ["raw", "labels", "ip_pos", "pay_pos", "is_ack_only"])
READ_CAP = 200
_L2_PROTOS = {"geninfo", "frame", "eth", "ethertype", "sll", "null", "raw",
              "loop", "linux-cooked", "fake-field-wrapper"}
_PAYLOAD_FIELDS = ("tcp.payload", "udp.payload", "data.data", "data")


def read_raw_frames(path, cap=READ_CAP):
    frames = []
    try:
        with open(path, "rb") as f:
            try:
                rdr = dpkt.pcap.Reader(f)
            except ValueError:
                # pcapng (e.g. tshark -w default format) fallback
                f.seek(0)
                try:
                    rdr = dpkt.pcapng.Reader(f)
                except (ValueError, AttributeError):
                    return None, frames
            dl = rdr.datalink()
            for _, buf in rdr:
                frames.append(bytes(buf))
                if len(frames) >= cap:
                    break
            return dl, frames
    except Exception:
        return None, frames


def tshark_available():
    return shutil.which("tshark") is not None


def run_tshark_pdml(path, cap=READ_CAP, decode_as=None, tshark_opts=None):
    cmd = ["tshark", "-r", str(path), "-c", str(cap), "-T", "pdml"]
    for expr, proto in (decode_as or []):
        cmd += ["-d", f"{expr},{proto}"]          # e.g. tcp.port==8443,tls
    cmd += list(tshark_opts or [])
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=300)
    except Exception:
        return None


def parse_pdml(xml_bytes, frames, cap=READ_CAP):
    import xml.etree.ElementTree as ET
    out = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return out
    for pi, pkt in enumerate(root.findall("packet")):
        if pi >= len(frames) or len(out) >= cap:
            break
        raw = frames[pi]; n = len(raw)
        labels = ["payload"] * n
        cover = [1 << 30] * n
        ip_pos, pay_pos = 0, n
        ack_flag, has_tcp, tcp_end = False, False, None
        for proto in pkt.findall("proto"):
            pname = proto.get("name", "")
            ppos = int(proto.get("pos", 0) or 0)
            psize = int(proto.get("size", 0) or 0)
            if pname in _L2_PROTOS:
                continue
            if pname in ("ip", "ipv6") and ip_pos == 0:
                ip_pos = ppos
            if pname in ("tcp", "udp"):
                has_tcp = pname == "tcp"; tcp_end = ppos + psize
            for b in range(max(ppos, 0), min(ppos + psize, n)):
                if psize < cover[b]:
                    labels[b], cover[b] = pname, psize
            for fld in proto.iter("field"):
                fname = fld.get("name", "")
                if not fname:
                    continue
                try:
                    fpos = int(fld.get("pos")); fsize = int(fld.get("size"))
                except (TypeError, ValueError):
                    continue
                if fsize <= 0:
                    continue
                if fname in _PAYLOAD_FIELDS:
                    pay_pos = min(pay_pos, fpos)
                if fname == "tcp.flags":
                    try:
                        ack_flag = (int(fld.get("value", "0"), 16) & 0x3F) == 0x10
                    except ValueError:
                        pass
                for b in range(max(fpos, 0), min(fpos + fsize, n)):
                    if fsize < cover[b]:
                        labels[b], cover[b] = fname, fsize
        if pay_pos == n and tcp_end is not None:
            pay_pos = min(tcp_end, n)
        has_payload = pay_pos < n
        out.append(ParsedPacket(raw, labels, ip_pos, pay_pos,
                                ack_flag and has_tcp and not has_payload))
    return out


def read_session_packets(path, cap=READ_CAP, decode_as=None, tshark_opts=None):
    """Session pcap → ParsedPacket list (requires tshark; None if not installed)."""
    if not tshark_available():
        return None
    xml = run_tshark_pdml(path, cap, decode_as, tshark_opts)
    if xml is None:
        return None
    _, frames = read_raw_frames(path, cap)
    if not frames:
        return None
    return parse_pdml(xml, frames, cap)


def parse_frames_pdml(frames_pcap_path, cap=READ_CAP, decode_as=None, tshark_opts=None):
    """Parse tshark PDML from an existing (session) pcap path. Alias of read_session_packets."""
    return read_session_packets(frames_pcap_path, cap, decode_as, tshark_opts)


def build_pcap_index(session_root):
    idx = {}
    session_root = Path(session_root)
    if not session_root.exists():
        return idx
    for pat in ("*.pcap", "*.pcapng"):
        for p in session_root.rglob(pat):
            idx.setdefault(p.name, p)
    return idx
