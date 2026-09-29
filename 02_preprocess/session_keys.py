# -*- coding: utf-8 -*-
"""Content key of a session, independent of file names and of the tshark version.

    key = sha1( l4 | endpoint_a | endpoint_b | ts_first_us )[:16]

l4 is the transport protocol in lower case, endpoint_a/endpoint_b are the two "ip:port" strings in sorted order, and
ts_first_us is the first-packet time in integer microseconds (tshark prints 6 or 9 decimals depending on its version).
pkt_count is kept next to the key as a consistency check. Used by align_session_lists.py (and by the release tools that
wrote 01_dataset/<dataset>/session_keys.csv)."""
import hashlib
from decimal import Decimal, InvalidOperation


def ts_us(ts):
    try:
        return str(int((Decimal(str(ts).strip()) * 1_000_000).to_integral_value()))
    except (InvalidOperation, ValueError):
        return ""


def endpoints(src_ip, src_port, dst_ip, dst_port):
    return tuple(sorted([f"{src_ip}:{int(float(src_port))}" if str(src_port).strip() not in ("", "-") else f"{src_ip}:-",
                         f"{dst_ip}:{int(float(dst_port))}" if str(dst_port).strip() not in ("", "-") else f"{dst_ip}:-"]))


def key_from_parts(l4, a, b, ts):
    return hashlib.sha1(f"{str(l4).lower()}|{a}|{b}|{ts_us(ts)}".encode()).hexdigest()[:16]


def key_from_stat_row(r):
    """r: a row of 03_session_stat (dict with L4, src_ip, src_port, dst_ip, dst_port, ts_first)."""
    a, b = endpoints(r["src_ip"], r["src_port"], r["dst_ip"], r["dst_port"])
    return key_from_parts(r["L4"], a, b, r["ts_first"])
