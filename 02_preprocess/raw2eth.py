#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""raw2eth.py — Ethernet wrapping of session pcaps + IP/port masking (in-place, parallel).
   · DLT_RAW(IP) → Ethernet: required because NetFound 1_filter assumes Ethernet.
   · --ip_mask   : IP src/dst → 0   · --port_mask : TCP/UDP ports → 0 (same as the byte models)
   · --workers N : per-file parallelism (masking is heavy, so parallelism is required)
   Usage: python3 raw2eth.py <raw_dir> [--ip_mask] [--port_mask] [--workers N]
"""
import sys, os, glob, struct, argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from scapy.all import rdpcap, wrpcap, Ether, IP, IPv6, TCP, UDP   # inherited by workers via fork

RAW_DLTS = {12, 101, 228, 229}
IP_MASK = PORT_MASK = False          # set in main → inherited by workers via fork


def datalink(path):
    with open(path, "rb") as f:
        h = f.read(24)
    if len(h) < 24:
        return None
    end = "<" if h[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    return struct.unpack(end + "I", h[20:24])[0]


def process_one(fp):
    dl = datalink(fp)
    need_wrap = dl in RAW_DLTS
    if not need_wrap and dl != 1:
        return ("unk", dl)
    if not need_wrap and not (IP_MASK or PORT_MASK):
        return ("eth", dl)
    try:
        out = []
        for p in rdpcap(fp):
            if IP_MASK:
                if IP in p:
                    p[IP].src = "0.0.0.0"; p[IP].dst = "0.0.0.0"
                    if hasattr(p[IP], "chksum"): del p[IP].chksum
                if IPv6 in p:
                    p[IPv6].src = "::"; p[IPv6].dst = "::"
            if PORT_MASK:
                if TCP in p:
                    p[TCP].sport = 0; p[TCP].dport = 0
                    if hasattr(p[TCP], "chksum"): del p[TCP].chksum
                if UDP in p:
                    p[UDP].sport = 0; p[UDP].dport = 0
                    if hasattr(p[UDP], "chksum"): del p[UDP].chksum
            out.append(Ether() / p if need_wrap else p)
        wrpcap(fp, out)
        return ("conv", dl)
    except Exception as e:
        return ("err", str(e))


def main():
    global IP_MASK, PORT_MASK
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("--ip_mask", action="store_true")
    ap.add_argument("--port_mask", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    IP_MASK, PORT_MASK = a.ip_mask, a.port_mask
    files = glob.glob(os.path.join(a.raw_dir, "**", "*.pcap"), recursive=True)
    tally = Counter(); dist = Counter(); errs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for kind, info in ex.map(process_one, files, chunksize=8):
            tally[kind] += 1
            if kind == "err" and len(errs) < 3:
                errs.append(info)
            else:
                dist[info] += 1
    for e in errs:
        print("  [raw2eth] err:", e)
    print(f"[raw2eth] converted {tally['conv']} / already-Eth-unmasked {tally['eth']} / unsupported {tally['unk']} / errors {tally['err']}"
          f"  (total {len(files)}, ip_mask={a.ip_mask} port_mask={a.port_mask}, workers={a.workers})")
    print(f"[raw2eth] datalink distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
