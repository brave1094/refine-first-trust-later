#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_session_stat.py
──────────────────────────────────────────────────────────────
Build per-session statistics / field feature CSV.
( The feature CSV is extracted directly from pcap, not JSON — tshark -T fields,
  desegment ON, _ws.col.info included. It becomes the base table for noise refinement (04). )

Two modes:
  • session (default) : 02_session/<class>/*.pcap — 1 file = 1 session.
  • whole          : 01_pcap/<class>/*.pcap   — from one monolithic pcap,
                     split sessions by tcp.stream / udp.stream and aggregate in one pass.

  --mode not given -> per-dataset default: session.  But for iot23, session splitting
  explodes inodes, so whole is the default (DEFAULT_MODE). Override with --mode if needed.

Output columns (common to all datasets):
  filename, session_id, task1, task2, task3, noise_label,
  L2, L3, L4, L7, src_ip, src_port, dst_ip, dst_port,
  pkt_count, payload_size, ws_col_info,
  has_SYN, has_SYN_ACK, has_FIN_cli, has_FIN_serv,
  has_TLS_CHLO, has_TLS_SHLO, tls_sni, http_uri, dns_qry, http_ua,
  has_RST, tls_heartbeat, http_method, http_host, http_resp_code,
  http_req_count, http_susp_uri
    └ extra columns (at the end) for class labeling in 03_1_labeling_cic17.py. task1/2/3 are
      based on weekday/folder name, not the actual attack class → relabeled in 03_1 via timestamp+features.

  - task1/2/3 = first 3 parts of class_name (directory name) split by '_'. The rest (task4+) is ignored.
    04_noise_labeling joins task1_task2_task3 into class_name and passes it to the rules.
  - session unique key = (filename, session_id)
  - corrupted pcaps are salvaged with editcap and retried (enabled by default, disable with --no-salvage).
  - whole   mode: filename=original pcap name, session_id="tcp_N" / "udp_N"
                  (stream restarts at 0 per file, so it is globally unique only together with filename)
  - session mode: filename=session file name,   session_id="-"
  - noise_label is empty (placeholder). Filled by 04_noise_labeling.py.

Output: <dataset_dir>/03_session_stat/session_stat_{dataset}_1.csv  (UTF-8 BOM)
           (dataset_dir = ds.root(name) in lib/datasets.py, i.e. each dataset directory)
Failure log: <dataset_dir>/03_session_stat/error_step3_{dataset}_{time}.txt

Usage:
  python3 03_session_stat.py --dataset vpn16                       # session(default)
  python3 03_session_stat.py --dataset iot23                       # whole(auto)
  python3 03_session_stat.py --dataset vpn16 --mode whole          # forced whole
  python3 03_session_stat.py --dataset cispec tls1.3 tor16 --workers 16
  python3 03_session_stat.py --dataset all
"""

import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import csv
import datetime as _dt
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import unquote as _unquote

try:
    from tqdm import tqdm
except Exception:                                   # works without tqdm
    def tqdm(it, **kw):
        return it

sys.path.insert(0, str(RP.PREPROCESS))
from lib import datasets as ds
from lib.salvage import salvage_pcap

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
N_WORKERS = 16
_TIMEOUT  = 0          # tshark timeout per pcap (s). 0 = unlimited. (for large cic17 day-pcaps)

# Per-dataset default when --mode is not given. (iot23 requires whole: session splitting explodes inodes)
DEFAULT_MODE = {"ustc16": "whole", "cic17": "whole", "cic18": "whole",
                "iot23": "whole"}   # all 5 use whole (re-extraction v1)

# Force desegment off per class (huge flood/frag/Mirai — no app-layer SNI/URI + OOM risk).
#   Even with global --desegment on, the classes below use off. Stats/flags/streams are identical.
#   iot23 rules (Bb_24/25/26) use only IP/L4/L7/port, so SNI/URI loss does not affect refinement.
CLASS_DESEG_OFF = {
    "iot23": {
        "attack_DDoS_ACK-Fragmentation", "attack_DDoS_ICMP-Flood",
        "attack_DDoS_ICMP-Fragmentation", "attack_DDoS_PSHACK-Flood",
        "attack_DDoS_RSTFIN-Flood", "attack_DDoS_SYN-Flood",
        "attack_DDoS_SynonymousIP-Flood", "attack_DDoS_TCP-Flood",
        "attack_DDoS_UDP-Flood", "attack_DDoS_UDP-Fragmentation",
        "attack_DoS_SYN-Flood", "attack_DoS_TCP-Flood", "attack_DoS_UDP-Flood",
        "attack_DoS_HTTP-Flood",
        "attack_Mirai_Greeth-Flood", "attack_Mirai_Greip-Flood",
        "attack_Mirai_UDPPlain",
    },
}

# fields extracted by tshark -T fields (order = parse index)
FIELDS = [
    "frame.number",                              # 0  pkt count
    "frame.protocols",                           # 1  L2~L7 detection
    "tcp.stream",                                # 2  whole session key
    "udp.stream",                                # 3  whole session key
    "ip.src",                                    # 4
    "ipv6.src",                                  # 5
    "ip.dst",                                    # 6
    "ipv6.dst",                                  # 7
    "tcp.srcport",                               # 8
    "udp.srcport",                               # 9
    "tcp.dstport",                               # 10
    "udp.dstport",                               # 11
    "tcp.len",                                   # 12 TCP payload length
    "udp.length",                                # 13 UDP length (incl. 8-byte header)
    "tcp.flags.syn",                             # 14
    "tcp.flags.ack",                             # 15
    "tcp.flags.fin",                             # 16
    "tls.handshake.type",                        # 17 1=CHLO 2=SHLO
    "tls.handshake.extensions_server_name",      # 18 SNI
    "http.request.uri",                          # 19
    "dns.qry.name",                              # 20
    "http.user_agent",                           # 21
    "_ws.col.info",                              # 22 noise rules (Aa_3 etc.)
    "frame.time_epoch",                          # 23 ts_first / ts_last
    # ── Below: extra fields for class labeling in 03_1_labeling_cic17.py. Appended only at the end, so
    #    existing indices (0~23) stay unchanged. run_tshark options are unchanged.
    "tcp.flags.reset",                           # 24 RST  → PortScan/Bot identification
    "tls.record.content_type",                   # 25 24=heartbeat → Heartbleed
    "http.request.method",                       # 26 GET/POST  → Web attack
    "http.host",                                 # 27 Host (target)
    "http.response.code",                        # 28 response code
    # ── For botnet (ARES) C2 identification: check markers such as botid in POST body/content type ──
    "http.content_type",                         # 29 multipart / urlencoded / text-html
    "urlencoded-form.key",                       # 30 POST /api/report body keys (botid,output)
    "mime_multipart.header.content-disposition", # 31 POST /api/upload form (name="botid", filename=..)
    # ── For brute-force signatures (FTP-Patator / SSH-Patator) ──
    "ftp.request.arg",                           # 32 USER/PASS argument values
    "ssh.protocol",                              # 33 SSH version string
    "udp.payload",                               # 34 nmap probe detection (raw payload not stored)
    "sip.User-Agent",                            # 35 SIP UA (friendly-scanner etc.)
]

OUT_COLUMNS = [
    "filename", "session_id", "task1", "task2", "task3", "noise_label",
    "L2", "L3", "L4", "L7", "src_ip", "src_port", "dst_ip", "dst_port",
    "pkt_count", "payload_size",
    "fwd_pkt", "fwd_byte", "bwd_pkt", "bwd_byte",
    "ts_first", "ts_last", "ws_col_info",
    "has_SYN", "has_SYN_ACK", "has_FIN_cli", "has_FIN_serv",
    "has_TLS_CHLO", "has_TLS_SHLO", "tls_sni", "http_uri", "dns_qry", "http_ua",
    # ── Extra columns for labeling (at the end → existing column order/names unchanged) ──
    "has_RST", "tls_heartbeat",
    "http_method", "http_host", "http_resp_code", "http_req_count", "http_susp_uri",
    # ── Extra columns for botnet (ARES) C2 identification ──
    "http_content_type", "http_post_keys", "upload_filename", "has_botid",
    # ── For brute-force signatures ──
    "ftp_arg", "ssh_proto", "is_nmap_probe", "sip_ua",
]

# frame.protocols token → layer classification
_L2 = {"eth", "ethernet", "sll", "sll2", "linux_sll", "raw", "null",
       "loop", "ppp", "nflog", "ipnet", "ieee802", "wlan"}
_L3 = {"ip", "ipv6", "arp", "rarp"}
_L4 = {"tcp", "udp", "icmp", "icmpv6", "igmp", "sctp", "gre"}
_SKIP = {"ethertype", "data", "frame", "vlan", "llc", "fc",
         "_ws.malformed", "tcp.segments", "tls.segments"}

# SMB exception: nbss (NetBIOS Session Service) is only a wrapper, so if smb/smb2 is inside
# it is promoted to L7. (otherwise nbss stays)
_SMB_WRAP  = {"nbss"}
_SMB_INNER = {"smb", "smb2"}

INFO_CAP = 1000
FTP_ARG_CAP = 512      # ftp_arg accumulation cap (chars)        # ws_col_info accumulation cap (chars)

# ── nmap-payloads 7.90 UDP probe signatures (is_nmap_probe detection) ──────────────
#   Per-port matching. Identifies the fixed probes sent by the scanner byte by byte.
#   Used for Bb_27 (nmap UDP probes under a benign label).
#   Source: https://svn.nmap.org/nmap-releases/nmap-7.90/nmap-payloads
#   Verified: matches manual review (835 version.bind probes).
#   mode: "p"=prefix, "s"=substring
NMAP_PROBE_SIGS = {
    "53":    [("76657273696f6e0462696e64", "s"),  # version.bind CHAOS TXT
              ("000010000000000000000000", "p")],  # DNSStatusRequest
    "5353":  [("76657273696f6e0462696e64", "s"),
              ("000010", "p"), ("0000000000010000", "p")],
    "111":   [("72fe1d13", "p"), ("3eece3ca", "p")],           # RPCCheck
    "137":   [("80f00010", "p"), ("01910000", "p"),
              ("01910010", "p"), ("434b414141", "s")],          # NBTStat CKAAAA
    "161":   [("303a020103", "p"), ("301f0201000406", "p"),
              ("7075626c6963", "s")],                            # SNMPv3 / public
    "1900":  [("4d2d534541524348", "p")],                       # SSDP M-SEARCH
    "177":   [("00010002000100", "p")],                         # XDMCP
    "623":   [("0600ff06", "p"), ("0600ff07", "p")],            # IPMI RMCP
    "123":   [("e30004fa", "p"), ("d9000afa", "p")],            # NTP
    "47808": [("810a", "p")],                                   # BACnet/IP Who-Is
    "5060":  [("4f5054494f4e53207369703a", "p")],               # SIP OPTIONS
}


def _is_nmap_probe(dport: str, payhex: str) -> bool:
    sigs = NMAP_PROBE_SIGS.get(dport)
    if not sigs or not payhex:
        return False
    h = payhex.lower()
    for sig, mode in sigs:
        if (h.startswith(sig) if mode == "p" else sig in h):
            return True
    return False



# ws_col_info is used only by noise rule Aa_3 (warn/error). Only info containing the signals below
# is kept; info of benign traffic (e.g. "443 → 1234 Len=..") is dropped and set to "-".
WARN_SUBSTRS = (
    "tcp dup ack", "previous segment not captured",
    "packet size limited during capture", "malformed", "truncated",
    "retransmission", "zerowindow", "zero window",
    "out-of-order", "out of order", "acked unseen segment",
    "spurious", "reassembly error",
)

# http_susp_uri: kept if the request URI (raw + URL-decoded) matches an attack marker below.
#   One column indicating whether the session contains an actual web-attack request (XSS/SQLi/BruteForce).
#   Note: short substring matches such as 'xss'/'sqli' give false positives on random cache-buster queries (/?XSSOMOJ=..),
#         so they are not used. Only attack-module 'paths' + concrete 'injection tokens' are used.
ATTACK_URI_MARKERS = (
    "/vulnerabilities/xss", "/vulnerabilities/sqli", "/vulnerabilities/brute",
    "/vulnerabilities/exec", "/vulnerabilities/fi", "/vulnerabilities/upload",
    "<script", "</script", "alert(", "onerror=", "onmouseover=", "javascript:",
    "union select", "union all select", "' or ", "or 1=1", "1=1--", "1=1#",
    "/etc/passwd", "../../",
)


def classify_layers(protocols: str):
    """frame.protocols('eth:ethertype:ip:tcp:tls') → (L2,L3,L4,L7)."""
    toks = [t for t in protocols.split(":") if t]
    l2 = l3 = l4 = "-"
    for t in toks:
        if l2 == "-" and t in _L2:
            l2 = t
        if l3 == "-" and t in _L3:
            l3 = t
        if l4 == "-" and t in _L4:
            l4 = t
    # L7 = "first" token that is not L2/L3/L4/SKIP (child dissectors roll up to the parent).
    #   Content dissectors merge into their parent, e.g. x509ce→tls, data-text-lines→http.
    #   Exception: the nbss wrapper — if smb/smb2 is inside, that is used as L7.
    l7 = "-"
    for t in toks:
        if t in _L2 or t in _L3 or t in _L4 or t in _SKIP:
            continue
        if t in _SMB_WRAP:
            inner = next((x for x in toks if x in _SMB_INNER), None)
            l7 = inner if inner else t
        else:
            l7 = t
        break
    return l2, l3, l4, l7


def split_tasks(class_name: str):
    """Split the directory name (class_name) by '_'; first 3 parts = task1/2/3.
       The rest (task4+) is ignored. Missing parts become '-'."""
    parts = class_name.split("_")
    t1 = parts[0] if len(parts) > 0 and parts[0] else "-"
    t2 = parts[1] if len(parts) > 1 and parts[1] else "-"
    t3 = parts[2] if len(parts) > 2 and parts[2] else "-"
    return t1, t2, t3


def _clean_err(stderr_bytes: bytes) -> str:
    """Remove 'Running as user root'-style warning lines from tshark stderr,
       keep only real errors, joined into one line."""
    s = stderr_bytes.decode("utf-8", "replace")
    lines = []
    for ln in s.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "Running as user" in ln or "This could be dangerous" in ln:
            continue
        lines.append(ln)
    msg = " ".join(lines)
    return msg[:200] if msg else "(no stderr)"


# ═══════════════════════════════════════════════════════════════════
# session accumulator (shared by session / whole)
# ═══════════════════════════════════════════════════════════════════
def _flag_set(v: str) -> bool:
    """Whether a tshark boolean field is set. Supports both '1'/'0' and 'True'/'False'."""
    for t in v.split(","):
        if t.strip().lower() in ("1", "true"):
            return True
    return False


class SessionAcc:
    __slots__ = (
        "pkt_count", "payload_size", "proto_best", "info_set",
        "client", "server", "first_src", "first_dst", "sid",
        "a_pkt", "a_byte", "b_pkt", "b_byte", "fin_a", "fin_b",
        "ts_first", "ts_last",
        "has_SYN", "has_SYN_ACK",
        "has_CHLO", "has_SHLO", "sni", "uri", "qry", "ua",
        "has_RST", "tls_heartbeat",
        "http_method", "http_host", "http_resp_code", "http_req_count", "http_susp_uri",
        "http_content_type", "http_post_keys", "upload_filename", "has_botid",
        "ftp_args", "ssh_proto", "is_nmap_probe", "sip_ua",
    )

    def __init__(self):
        self.pkt_count = 0
        self.payload_size = 0
        self.proto_best = ""          # frame.protocols with the most tokens
        self.info_set = []            # accumulated ws_col_info (warn/error)
        self.client = None            # (ip, port) — side that sent SYN
        self.server = None
        self.first_src = None
        self.first_dst = None
        self.sid = None               # tcp_N / udp_N (fixed by the first stream)
        self.a_pkt = self.a_byte = 0  # direction A = from first_src
        self.b_pkt = self.b_byte = 0  # direction B = reverse
        self.fin_a = self.fin_b = 0   # FIN presence per direction
        self.ts_first = None
        self.ts_last = None
        self.has_SYN = self.has_SYN_ACK = 0
        self.has_CHLO = self.has_SHLO = 0
        self.sni = self.uri = self.qry = self.ua = ""
        # extra fields for labeling
        self.has_RST = 0
        self.tls_heartbeat = 0
        self.http_method = self.http_host = self.http_resp_code = ""
        self.http_susp_uri = ""
        self.http_req_count = 0
        # botnet C2 (ARES)
        self.http_content_type = ""
        self.http_post_keys = ""
        self.upload_filename = ""
        self.has_botid = 0
        # For brute-force signatures: FTP args need the full repetition, so accumulate (capped)
        self.ftp_args = []
        self.ssh_proto = []
        self.is_nmap_probe = 0
        self.sip_ua = ""

    def add(self, f):
        self.pkt_count += 1

        proto = f[1]
        if proto.count(":") > self.proto_best.count(":"):
            self.proto_best = proto

        # session_id: fixed by the first tcp/udp.stream (tcp_N / udp_N)
        if self.sid is None:
            if f[2]:
                self.sid = f"tcp_{f[2].split(',')[0]}"
            elif f[3]:
                self.sid = f"udp_{f[3].split(',')[0]}"

        src = f[4] or f[5]            # ip.src or ipv6.src
        dst = f[6] or f[7]
        sport = f[8] or f[9]          # tcp/udp srcport
        dport = f[10] or f[11]
        ep_s = (src, sport)
        ep_d = (dst, dport)
        if self.first_src is None:
            self.first_src, self.first_dst = ep_s, ep_d

        # payload length of this packet
        plen = 0
        if f[12]:                                   # tcp.len
            try:
                plen = int(f[12].split(",")[0])
            except ValueError:
                plen = 0
        elif f[13]:                                 # udp.length - 8
            try:
                plen = max(0, int(f[13].split(",")[0]) - 8)
            except ValueError:
                plen = 0
        self.payload_size += plen

        # per-direction accumulation (A = same source as first_src)
        if ep_s == self.first_src:
            self.a_pkt += 1
            self.a_byte += plen
        else:
            self.b_pkt += 1
            self.b_byte += plen

        # timestamp (frame.time_epoch)
        if f[23]:
            try:
                te = float(f[23].split(",")[0])
                if self.ts_first is None or te < self.ts_first:
                    self.ts_first = te
                if self.ts_last is None or te > self.ts_last:
                    self.ts_last = te
            except ValueError:
                pass

        # ── Flags: always evaluated per 'this packet' ──
        # Depending on tshark version, booleans are "1"/"0" or "True"/"False" → accept both
        syn = _flag_set(f[14])
        ack = _flag_set(f[15])
        fin = _flag_set(f[16])
        # has_SYN: only if a pure SYN packet (SYN=1, ACK=0) actually exists
        if syn and not ack:
            self.has_SYN = 1
            if self.client is None:
                self.client, self.server = ep_s, ep_d
        # has_SYN_ACK: only if a SYN+ACK packet (SYN=1, ACK=1) actually exists
        if syn and ack:
            self.has_SYN_ACK = 1
            if self.server is None:
                self.server, self.client = ep_s, ep_d
        # FIN: counted whenever the FIN bit is set, even with PSH/ACK etc. Stored per direction.
        if fin:
            if ep_s == self.first_src:
                self.fin_a = 1
            else:
                self.fin_b = 1

        # TLS handshake type
        tt = f[17].split(",")
        if "1" in tt:
            self.has_CHLO = 1
        if "2" in tt:
            self.has_SHLO = 1

        # take the first non-empty L7 value
        if not self.sni and f[18]:
            self.sni = f[18].split(",")[0]
        if not self.uri and f[19]:
            self.uri = f[19].split(",")[0]
        if not self.qry and f[20]:
            self.qry = f[20].split(",")[0]
        if not self.ua and f[21]:
            self.ua = f[21].split(",")[0]

        # ── Extra fields for labeling (indices 24~28) ──
        if _flag_set(f[24]):                         # tcp.flags.reset
            self.has_RST = 1
        if f[25]:                                    # tls.record.content_type
            self.tls_heartbeat += sum(1 for t in f[25].split(",") if t == "24")
        if f[26]:                                    # http.request.method → request
            self.http_req_count += 1
            if not self.http_method:
                self.http_method = f[26].split(",")[0]
        if not self.http_host and f[27]:
            self.http_host = f[27].split(",")[0]
        if not self.http_resp_code and f[28]:
            self.http_resp_code = f[28].split(",")[0]
        # Suspicious URI: match this packet's request URI(s) against attack markers, keep the first match
        if not self.http_susp_uri and f[19]:
            for u in f[19].split(","):
                if not u:
                    continue
                ul = u.lower()
                ud = _unquote(ul)
                if any(m in ul or m in ud for m in ATTACK_URI_MARKERS):
                    self.http_susp_uri = u
                    break

        # ── Botnet (ARES) C2: check markers such as botid in POST body/content type (indices 29~31) ──
        if not self.http_content_type and f[29]:
            self.http_content_type = f[29].split(",")[0]
        cd = f[31]                                   # mime_multipart content-disposition (upload)
        uk = f[30]                                   # urlencoded-form keys (report)
        if uk or cd:
            keys = []
            if uk:
                keys += [k for k in uk.split(",") if k]
            if cd:
                keys += re.findall(r'\bname="([^"]+)"', cd)
                if not self.upload_filename:
                    m = re.search(r'filename="([^"]+)"', cd)
                    if m:
                        self.upload_filename = m.group(1)
            if keys and not self.http_post_keys:
                # deduplicate while preserving order
                seen = []
                for k in keys:
                    if k not in seen:
                        seen.append(k)
                self.http_post_keys = "|".join(seen)
        # has_botid: 1 if botid appears in the request URI (GET) or the POST body keys (report/upload)
        if not self.has_botid:
            blob = (f[19] + " " + uk + " " + cd).lower()
            if "botid" in blob:
                self.has_botid = 1

        # ws_col_info: keep only warn/error (Aa_3) signals, drop other benign info
        # accumulate FTP args (USER/PASS values) — for FTP-Patator signature (literal 	) detection
        if f[32] and sum(len(x) for x in self.ftp_args) < FTP_ARG_CAP:
            for a in f[32].split(","):
                a = a.strip()
                if a and a not in self.ftp_args:
                    self.ftp_args.append(a[:64])
        # SSH version string — for tool fingerprinting (paramiko/JSCH/OpenSSH).
        #   [Note] In SSH the server sends its banner first (RFC 4253). Keeping only the 'first value'
        #   captures the server (OpenSSH) and misses the client (paramiko = attack tool).
        #   → accumulate all banners in the session. The rule checks whether paramiko is among them.
        if f[33]:
            for _b in f[33].split(","):
                _b = _b.strip()[:64]
                if _b and _b not in self.ssh_proto:
                    self.ssh_proto.append(_b)
        # nmap probe detection: match udp.payload (f[34]) against per-port signatures (raw payload not stored)
        if not self.is_nmap_probe and f[34]:
            _ph = f[34].split(",")[0].replace(":", "")
            if _is_nmap_probe(dport.split(",")[0], _ph):
                self.is_nmap_probe = 1
        # SIP User-Agent (f[35]) — friendly-scanner etc.
        if not self.sip_ua and f[35]:
            self.sip_ua = f[35].split(",")[0].strip()[:64]

        info = f[22].strip()
        if info:
            low = info.lower()
            if any(kw in low for kw in WARN_SUBSTRS) and info not in self.info_set:
                if sum(len(x) for x in self.info_set) < INFO_CAP:
                    self.info_set.append(info)

    def finalize(self):
        client = self.client or self.first_src or ("", "")
        server = self.server or self.first_dst or ("", "")
        # Direction mapping: A=first_src. If client is first_src then fwd=A, else fwd=B.
        if self.first_src is not None and client == self.first_src:
            fwd_pkt, fwd_byte = self.a_pkt, self.a_byte
            bwd_pkt, bwd_byte = self.b_pkt, self.b_byte
            fin_cli, fin_serv = self.fin_a, self.fin_b
        else:
            fwd_pkt, fwd_byte = self.b_pkt, self.b_byte
            bwd_pkt, bwd_byte = self.a_pkt, self.a_byte
            fin_cli, fin_serv = self.fin_b, self.fin_a
        l2, l3, l4, l7 = classify_layers(self.proto_best)
        ts_first = f"{self.ts_first:.6f}" if self.ts_first is not None else "-"
        ts_last = f"{self.ts_last:.6f}" if self.ts_last is not None else "-"
        ws = " || ".join(self.info_set) if self.info_set else "-"
        return {
            "L2": l2, "L3": l3, "L4": l4, "L7": l7,
            "src_ip": client[0], "src_port": client[1],
            "dst_ip": server[0], "dst_port": server[1],
            "pkt_count": self.pkt_count,
            "payload_size": self.payload_size,
            "fwd_pkt": fwd_pkt, "fwd_byte": fwd_byte,
            "bwd_pkt": bwd_pkt, "bwd_byte": bwd_byte,
            "ts_first": ts_first, "ts_last": ts_last,
            "ws_col_info": ws,
            "has_SYN": self.has_SYN, "has_SYN_ACK": self.has_SYN_ACK,
            "has_FIN_cli": fin_cli, "has_FIN_serv": fin_serv,
            "has_TLS_CHLO": self.has_CHLO, "has_TLS_SHLO": self.has_SHLO,
            "tls_sni": self.sni, "http_uri": self.uri,
            "dns_qry": self.qry, "http_ua": self.ua,
            "has_RST": self.has_RST, "tls_heartbeat": self.tls_heartbeat,
            "http_method": self.http_method, "http_host": self.http_host,
            "http_resp_code": self.http_resp_code,
            "http_req_count": self.http_req_count,
            "http_susp_uri": self.http_susp_uri,
            "http_content_type": self.http_content_type,
            "http_post_keys": self.http_post_keys,
            "upload_filename": self.upload_filename,
            "has_botid": self.has_botid,
            "ftp_arg": " | ".join(self.ftp_args) if self.ftp_args else "",
            "ssh_proto": " | ".join(self.ssh_proto) if self.ssh_proto else "",
            "is_nmap_probe": self.is_nmap_probe,
            "sip_ua": self.sip_ua,
        }


# ═══════════════════════════════════════════════════════════════════
# run tshark
# ═══════════════════════════════════════════════════════════════════
def run_tshark(pcap: Path, timeout, desegment=True):
    # desegment=False disables TCP/TLS reassembly.
    #   On huge single-session pcaps reassembly blows up CPU/memory → disabling it is fast with no OOM.
    #   Per-packet fields such as stats·flags·payload·stream are unaffected (identical).
    #   However, SNI/URI of large TLS/HTTP spanning segments are read only from the first segment, so some may be lost.
    flag = "TRUE" if desegment else "FALSE"
    cmd = [
        "tshark", "-r", str(pcap), "-n", "-T", "fields",
        "--disable-protocol", "prp",
        "--disable-protocol", "vssmonitoring",
        "-o", f"tcp.desegment_tcp_streams:{flag}",
        "-o", f"tls.desegment_ssl_records:{flag}",
        "-o", f"tls.desegment_ssl_application_data:{flag}",
        "-E", "separator=/t", "-E", "occurrence=a", "-E", "aggregator=,",
    ]
    for fld in FIELDS:
        cmd += ["-e", fld]
    to = None if (timeout in (0, None)) else timeout
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=to,
    )
    return proc


# Cap on the session dict in whole mode. When exceeded, the oldest sessions are flushed to a part file to free memory.
#   Even flood pcaps (tens of millions of sessions) use only this cap × number of workers in memory.
FLUSH_CAP = int(os.environ.get("STREAM_CAP", "500000"))


def _tshark_popen(pcap, desegment):
    """Streaming tshark. Same options as run_tshark. stderr is discarded (avoids pipe deadlock)."""
    flag = "TRUE" if desegment else "FALSE"
    cmd = [
        "tshark", "-r", str(pcap), "-n", "-T", "fields",
        "--disable-protocol", "prp",
        "--disable-protocol", "vssmonitoring",
        "-o", f"tcp.desegment_tcp_streams:{flag}",
        "-o", f"tls.desegment_ssl_records:{flag}",
        "-o", f"tls.desegment_ssl_application_data:{flag}",
        "-E", "separator=/t", "-E", "occurrence=a", "-E", "aggregator=,",
    ]
    for fld in FIELDS:
        cmd += ["-e", fld]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=1 << 20)


def _process_whole(pcap, fname, t1, t2, t3, desegment, do_salvage, part_dir):
    """whole mode: tshark streaming + periodic session-dict flush → part CSV.
       Memory is bounded by FLUSH_CAP (independent of pcap size·session count)."""
    n = len(FIELDS)
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in fname)
    part_path = Path(part_dir) / f"{safe}.csv"

    def _tag(row, proto, stream):
        row["filename"] = fname
        row["session_id"] = f"{proto}_{stream}"
        row["task1"], row["task2"], row["task3"] = t1, t2, t3
        row["noise_label"] = ""
        return row

    def run_once(target):
        """One streaming pass → (rc, nrows). Writes part_path fresh.
           ★ No intermediate flush. All sessions are collected and written once after EOF →
             exactly 1 row per session, fully matching the session count read directly from the pcap."""
        p = _tshark_popen(target, desegment)
        sessions = {}
        for raw in p.stdout:
            f = raw.decode("utf-8", "replace").rstrip("\r\n").split("\t")
            if len(f) < n:
                f += [""] * (n - len(f))
            tcp_s, udp_s = f[2], f[3]
            if tcp_s:
                key = ("tcp", tcp_s.split(",")[0])
            elif udp_s:
                key = ("udp", udp_s.split(",")[0])
            else:
                continue                       # drop non-TCP/UDP (ARP/ICMP)
            acc = sessions.get(key)
            if acc is None:
                acc = sessions[key] = SessionAcc()
            acc.add(f)
        p.wait()
        nrows = 0
        with open(part_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=OUT_COLUMNS)
            w.writeheader()
            for (proto, stream), acc in sessions.items():
                w.writerow(_tag(acc.finalize(), proto, stream))
                nrows += 1
        return p.returncode, nrows

    # first attempt
    try:
        rc, nrows = run_once(pcap)
    except Exception as e:                                      # noqa
        return ("ERR", f"exec:{e}\t{pcap}")
    # failure → retry with the salvaged copy (overwrites part)
    if rc != 0 and do_salvage:
        sp, snote = salvage_pcap(pcap)
        if sp is not None:
            try:
                rc, nrows = run_once(sp)
            except Exception as e:                              # noqa
                try: Path(sp).unlink()
                except OSError: pass
                return ("ERR", f"exec(salvaged):{e}\t{pcap}")
            try: Path(sp).unlink()
            except OSError: pass
    if nrows == 0:
        try: part_path.unlink()
        except OSError: pass
        return ("ERR", (f"tshark_rc{rc}" if rc != 0 else "0sessions") + f"\t{pcap}")
    return ("OK_PART", str(part_path), fname)


def process_pcap(args):
    """worker: process one pcap.
       whole mode → write part CSV via streaming+flush, ('OK_PART', path, fname).
       session mode → one session, ('OK', [row])."""
    pcap_str, class_name, mode, timeout, do_salvage, desegment, part_dir = args
    pcap = Path(pcap_str)
    fname = pcap.name                       # filename always keeps the original name
    t1, t2, t3 = split_tasks(class_name)

    if mode == "whole":
        return _process_whole(pcap, fname, t1, t2, t3, desegment, do_salvage, part_dir)

    # ── session mode (original kept) ──
    # first tshark pass
    try:
        proc = run_tshark(pcap, timeout, desegment)
    except subprocess.TimeoutExpired:
        return ("ERR", f"timeout\t{pcap}")
    except Exception as e:                                  # noqa
        return ("ERR", f"exec:{e}\t{pcap}")

    # failure → salvage and retry (cut short / oversized etc.)
    #   The salvaged copy is a temp file and is always deleted afterwards (try/finally). filename keeps the original name.
    salv_path = None
    try:
        salv_note = ""
        if proc.returncode != 0 and do_salvage:
            sp, snote = salvage_pcap(pcap)
            if sp is not None:
                salv_path = sp
                try:
                    proc2 = run_tshark(sp, timeout, desegment)
                    if proc2.returncode == 0:
                        proc = proc2
                        salv_note = f" [salvaged:{snote}]"
                except subprocess.TimeoutExpired:
                    return ("ERR", f"timeout(salvaged)\t{pcap}")
                except Exception as e:                          # noqa
                    return ("ERR", f"exec(salvaged):{e}\t{pcap}")

        if proc.returncode != 0:
            return ("ERR", f"tshark_rc{proc.returncode}:{_clean_err(proc.stderr)}\t{pcap}")

        text = proc.stdout.decode("utf-8", "replace")
        if not text.strip():
            return ("ERR", f"0packets\t{pcap}")

        def _tag(row, sid):
            row["filename"] = fname          # ★ always the original name (not the salvaged copy's)
            row["session_id"] = sid
            row["task1"], row["task2"], row["task3"] = t1, t2, t3
            row["noise_label"] = ""
            return row

        n = len(FIELDS)
        # session mode: whole file = one session
        acc = SessionAcc()
        cnt = 0
        for line in text.splitlines():
            f = line.split("\t")
            if len(f) < n:
                f += [""] * (n - len(f))
            acc.add(f)
            cnt += 1
        if cnt == 0:
            return ("ERR", f"0packets\t{pcap}")
        return ("OK", [_tag(acc.finalize(), acc.sid or "-")])
    finally:
        if salv_path is not None:                # clean up the temporary salvaged copy
            try:
                Path(salv_path).unlink()
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════
# dataset processing
# ═══════════════════════════════════════════════════════════════════
def collect_tasks(name, mode, timeout, do_salvage, desegment):
    base = ds.pcap_dir(name) if mode == "whole" else ds.session_dir(name)
    base = Path(base)
    if not base.is_dir():
        print(f"  [skip] {name}: input directory not found {base}")
        return []
    tasks = []
    _off = CLASS_DESEG_OFF.get(name, set())
    for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if class_dir.name == ".salvaged":
            continue
        # per-class desegment: forced off if in the OFF set, otherwise the global value
        _dg = False if class_dir.name in _off else desegment
        for pcap in sorted(class_dir.glob("*.pcap")) + sorted(class_dir.glob("*.pcapng")):
            tasks.append((str(pcap), class_dir.name, mode, timeout, do_salvage, _dg))
    return tasks


def _chunk_path(out_dir, name, k):
    return out_dir / f"session_stat_{name}_{k}.csv"


def _done_path(out_dir, name):
    return out_dir / f"session_stat_{name}_done.txt"


def _read_done(out_dir, name):
    """Set of 'fully written' files from the completion manifest (one line = one pcap filename)."""
    p = _done_path(out_dir, name)
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _resume_prepare(out_dir, name, keep, split_rows):
    """Fast resume preparation.
       - If the CSV has rows whose filename is not in the completion record (keep) (= partially written due to interruption),
         only then purge+rechunk (full rewrite). Usually there are none, so the rewrite is skipped.
       - If no cleanup is needed, only count the rows of the last chunk and return (lastk, last_rows) (no full rewrite).
       Returns: (lastk, last_rows, purged?)."""
    chunks = sorted(out_dir.glob(f"session_stat_{name}_*.csv"),
                    key=lambda x: int(x.stem.rsplit("_", 1)[1]))
    if not chunks:
        return 0, 0, False

    # 1) Fast scan: look only at the filename column (first column); stop as soon as one not in keep is found.
    #    Read only, no csv parsing/rewriting — break on first hit.
    need = False
    for p in chunks:
        with open(p, encoding="utf-8-sig") as f:
            next(f, None)                       # header
            for line in f:
                i = line.find(",")
                fn = line[:i] if i >= 0 else line.rstrip("\n")
                if fn and fn not in keep:
                    need = True
                    break
        if need:
            break

    if need:
        # incomplete/partially written rows exist → full cleanup·rechunk (slow, only when needed)
        lastk, last_rows = _purge_and_rechunk(out_dir, name, keep, split_rows)
        return lastk, last_rows, True

    # 2) No cleanup needed → quickly count the rows of the last chunk only (no rewrite)
    lastk = int(chunks[-1].stem.rsplit("_", 1)[1])
    with open(chunks[-1], encoding="utf-8-sig") as f:
        last_rows = sum(1 for _ in f) - 1       # excluding header
    if last_rows < 0:
        last_rows = 0
    return lastk, last_rows, False


def _purge_and_rechunk(out_dir, name, keep, split_rows):
    """Keep only rows with filename ∈ keep (= has a completion record) from existing chunks and rewrite as _1.._k.
       (files without a completion record = rows partially written due to interruption/OOM → all removed)
       Returns: (last chunk number, row count of the last chunk)."""
    chunks = sorted(out_dir.glob(f"session_stat_{name}_*.csv"),
                    key=lambda x: int(x.stem.rsplit("_", 1)[1]))
    if not chunks:
        return 0, 0
    split = split_rows if (split_rows and split_rows > 0) else None
    tmp_paths = []
    state = {"k": 0, "count": 0, "fh": None, "w": None}

    def _open():
        if state["fh"]:
            state["fh"].close()
        state["k"] += 1
        state["count"] = 0
        tp = out_dir / f".rechunk_{name}_{state['k']}.csv"
        tmp_paths.append(tp)
        state["fh"] = open(tp, "w", encoding="utf-8-sig", newline="")
        state["w"] = csv.writer(state["fh"])
        state["w"].writerow(OUT_COLUMNS)

    _open()
    for p in chunks:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rd = csv.reader(f)
            next(rd, None)                       # skip header
            for parts in rd:
                if not parts:
                    continue
                if parts[0] in keep:             # keep only rows of completed files
                    if split and state["count"] >= split:
                        _open()
                    state["w"].writerow(parts)
                    state["count"] += 1
    if state["fh"]:
        state["fh"].close()
    last_rows = state["count"]
    for p in chunks:                             # remove existing chunks, then temp → final name
        p.unlink()
    for i, tp in enumerate(tmp_paths, 1):
        tp.rename(out_dir / f"session_stat_{name}_{i}.csv")
    return len(tmp_paths), last_rows


class DoneLog:
    """Append completed pcaps one per line. Recorded only 'after' the rows are fully written to CSV and flushed → atomicity."""
    def __init__(self, path, resume):
        self.fh = open(path, "a" if resume else "w", encoding="utf-8")

    def mark(self, fname):
        self.fh.write(fname + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


class ChunkWriter:
    """Save the OUT_COLUMNS CSV split into _1 _2 _3... every split_rows rows.
       If resume_state=(lastk, last_rows), append to the last chunk (next chunk if full)."""
    def __init__(self, out_dir, name, split_rows, resume_state=None):
        self.out_dir = out_dir
        self.name = name
        self.split = split_rows if (split_rows and split_rows > 0) else None
        self.fh = None
        self.writer = None
        self.k = 1
        self.count = 0
        if resume_state is None:
            # fresh: remove all existing chunks, start from _1
            for old in out_dir.glob(f"session_stat_{name}_*.csv"):
                old.unlink()
            self._open(1, append=False)
        else:
            lastk, last_rows = resume_state
            if lastk < 1:
                self._open(1, append=False)
            elif self.split and last_rows >= self.split:
                self._open(lastk + 1, append=False)      # last chunk full → new chunk
            else:
                self._open(lastk, append=True, prefill=last_rows)  # append

    def _open(self, k, append, prefill=0):
        if self.fh:
            self.fh.close()
        self.k = k
        p = _chunk_path(self.out_dir, self.name, k)
        nonempty = p.exists() and p.stat().st_size > 0
        self.fh = open(p, "a" if append else "w", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=OUT_COLUMNS)
        if not (append and nonempty):
            self.writer.writeheader()
        self.count = prefill if append else 0

    def writerow(self, row):
        if self.split and self.count >= self.split:
            self._open(self.k + 1, append=False)
        self.writer.writerow(row)
        self.count += 1

    def writeraw(self, line):
        """Write the data rows (raw) of a part CSV to the chunk as-is (excluding header)."""
        if self.split and self.count >= self.split:
            self._open(self.k + 1, append=False)
        self.fh.write(line if line.endswith("\n") else line + "\n")
        self.count += 1

    def flush(self):
        if self.fh:
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


def process_dataset(name, mode, workers, timeout, do_salvage,
                    split_rows=1_000_000, resume=False, desegment=True):
    out_dir = Path(ds.root(name)) / "03_session_stat"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    err_log = out_dir / f"error_step3_{name}_{ts}.txt"
    done_path = _done_path(out_dir, name)

    tasks = collect_tasks(name, mode, timeout, do_salvage, desegment)
    if not tasks:
        return
    # part directory for whole-mode workers (streaming flush output)
    part_dir = out_dir / "_part_v1"
    part_dir.mkdir(parents=True, exist_ok=True)
    tasks = [tuple(t) + (str(part_dir),) for t in tasks]

    resume_state = None
    if resume:
        done = _read_done(out_dir, name)                 # completion manifest = source of truth
        # cleanup·rechunk only when incomplete (partially written) rows exist. Usually just quickly count the last chunk's rows.
        lastk, last_rows, purged = _resume_prepare(out_dir, name, done, split_rows)
        resume_state = (lastk, last_rows)
        before = len(tasks)
        tasks = [t for t in tasks if Path(t[0]).name not in done]
        if done:
            tail = ("cleaned incomplete rows·rechunked" if purged
                    else "no cleanup needed (rewrite skipped)")
            print(f"  [{name}] resume: done {len(done)} / to process {len(tasks)} (total {before})  "
                  f"{tail}→_{max(lastk,1)}({last_rows} rows)  "
                  f"timeout={timeout or 'inf'} salvage={do_salvage}")
        else:
            print(f"  [{name}] mode={mode}  files={len(tasks)}  "
                  f"split_rows={split_rows or 'off'}  -> session_stat_{name}_*.csv  (no done.txt → from scratch)")
        if not tasks:
            print("    No files to process — all complete.")
            return
    else:
        # --renew: discard existing chunks + completion record (ChunkWriter deletes chunks, DoneLog('w') truncates the manifest)
        print(f"  [{name}] RENEW  mode={mode}  files={len(tasks)}  "
              f"split_rows={split_rows or 'off'}  -> session_stat_{name}_*.csv (from scratch)")

    writer = ChunkWriter(out_dir, name, split_rows, resume_state)
    donelog = DoneLog(done_path, resume)
    n_ok = n_sess = n_err = 0
    errors = []
    try:
        with Pool(processes=workers) as pool:
            for status, *rest in tqdm(
                pool.imap_unordered(process_pcap, tasks),
                total=len(tasks), desc=name, unit="pcap",
            ):
                payload = rest if status == "OK_PART" else rest[0]
                if status == "OK_PART":
                    # whole mode: read the part CSV written by the worker, write it to the chunk, then delete it
                    n_ok += 1
                    part_path, fname = payload
                    with open(part_path, encoding="utf-8-sig") as pf:
                        next(pf, None)                 # skip header
                        for line in pf:
                            writer.writeraw(line)
                            n_sess += 1
                    writer.flush()
                    donelog.mark(fname)
                    try:
                        os.remove(part_path)
                    except OSError:
                        pass
                elif status == "OK":
                    # session mode: list of rows
                    n_ok += 1
                    for row in payload:
                        writer.writerow(row)
                        n_sess += 1
                    writer.flush()
                    donelog.mark(payload[0]["filename"])
                else:
                    n_err += 1
                    errors.append(payload)
    finally:
        writer.close()
        donelog.close()

    if errors:
        with open(err_log, "w", encoding="utf-8") as fh:
            fh.write("reason\tpath\n")
            fh.write("\n".join(errors) + "\n")
    print(f"    done: files_ok={n_ok} sessions={n_sess} files_err={n_err}"
          f"  chunks≤_{writer.k}"
          + (f"  (errlog: {err_log.name})" if errors else ""))


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Extract per-session feature CSV (session by default, whole only for iot23)")
    ap.add_argument("--dataset", nargs="+", required=True,
                    help="dataset names or all")
    ap.add_argument("--mode", choices=["whole", "session"], default=None,
                    help="If not given, per-dataset default (session; whole only for iot23). "
                         "whole=monolithic 01_pcap / session=02_session files")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--timeout", type=int, default=_TIMEOUT,
                    help="tshark timeout per pcap (s). 0=unlimited")
    ap.add_argument("--no-salvage", dest="salvage", action="store_false",
                    help="disable editcap salvage-and-retry for corrupted pcaps (enabled by default)")
    ap.set_defaults(salvage=True)
    ap.add_argument("--split-rows", type=int, default=1_000_000,
                    help="split the CSV into _1 _2 _3... every N rows (default 1,000,000, 0=no split)")
    ap.add_argument("--renew", action="store_true",
                    help="delete existing CSV chunks + completion record (done.txt) and restart from scratch. "
                         "(default is resume: read done.txt, skip completed pcaps, "
                         "process only incomplete/failed ones, clean up partially written rows)")
    ap.add_argument("--desegment", choices=["on", "off"], default="on",
                    help="TCP/TLS reassembly (default on). off makes huge single-session pcaps fast and "
                         "memory-safe (stats·flags·payload identical; only SNI/URI of large TLS/HTTP may be partly lost)")
    args = ap.parse_args()

    if len(args.dataset) == 1 and args.dataset[0] == "all":
        names = ds.names()
    else:
        names = args.dataset
        unknown = [x for x in names if x not in ds.names()]
        if unknown:
            sys.exit(f"[error] unknown dataset: {unknown}  (available: {ds.names()})")

    for name in names:
        # use --mode if given, otherwise the per-dataset default (session, iot23=whole)
        mode = args.mode or DEFAULT_MODE.get(name, "session")
        resume = not args.renew              # resume by default, from scratch with --renew
        desegment = (args.desegment == "on")
        print(f"[{name}] mode={mode}  workers={args.workers}  "
              f"timeout={args.timeout or 'inf'}  salvage={args.salvage}  "
              f"split_rows={args.split_rows or 'off'}  desegment={args.desegment}  "
              f"{'RENEW(from scratch)' if args.renew else 'resume(continue)'}")
        process_dataset(name, mode, args.workers, args.timeout, args.salvage,
                        args.split_rows, resume, desegment)


if __name__ == "__main__":
    main()