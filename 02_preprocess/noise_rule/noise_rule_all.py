"""
02_preprocess/noise_rule/noise_rule_all.py
──────────────────────────────────────────────
Common noise rule engine (19 rules, no class reference).

  Dataset quality : Aa_1..5
  Task dynamics   : Cc_1..5 / Cd_1..4 / De_1..4 / Df_1

Applied identically to all datasets: 'mark everything' (no gate). Marking = information only.
  Cc_4 / Cd_1 are the 'pure' versions without class exceptions. Datasets needing VoIP/P2P exceptions
  override these two in noise_rule_{ds}.py (Eg).

[Change history]
  - Removed old Aa_2 (payload 0)
  - Merged old Aa_1 (No-SYN) + Aa_5 (No-3way) → Aa_1 (fires unless both SYN and SYN-ACK are present)
  - Old Aa_7 (No-SHLO) → Aa_5; marks if ClientHello 'or' ServerHello is missing
  - Removed old Cc_6 (private/local dst)
  - Others renumbered in table order

[Guard] Aa_5 fires only for actual TLS sessions (L7==tls / 443 / handshake traces).

The List_* / regexes in this module are imported and reused by the class rules (noise_rule_{ds}.py).
"""

import re

import pandas as pd

from .base import BaseNoiseRules

List_L3Control   = {"arp", "rarp", "icmp", "icmpv6", "teredo"}
List_NameResolve = {"dhcp", "dhcpv6", "bootp", "dns", "mdns", "llmnr", "nbns"}
List_Windows     = {"nbdgm", "nbss", "smb", "smb2", "srvloc", "ssdp"}
List_InfraMgmt   = {"snmp", "ntp", "rtcp", "lsd", "chargen", "stun"}
# For the Bb_3 (protocol mismatch) exception: network-control protocols are not treated as mismatches
_List_NetControl = {
    "arp", "rarp", "icmp", "icmpv6", "teredo",
    "dhcp", "dhcpv6", "bootp", "dns", "mdns", "llmnr", "nbns",
    "nbdgm", "nbss", "smb", "smb2", "srvloc", "ssdp",
    "snmp", "ntp", "rtcp", "lsd", "chargen", "stun",
}

# Aa_2 (corrupted/truncated) indicator terms — 'genuine capture corruption' only.
#   [Fix 260820] Previously "tcp dup ack" / "tcp previous segment not captured" were also
#   included, but these are normal by-products when reassembly cannot keep up with high-speed floods, so
#   they wrongly deleted flood attack sessions en masse (95.9% of iot23 PSHACK triggered previous-segment).
#   → Exclude normal TCP phenomena; keep only signals that the capture itself is broken.
List_WARN = [
    "packet size limited during capture", "malformed packet", "truncated",
]
_WARN_PAT = "|".join(re.escape(k) for k in List_WARN)
List_WinBackground = {
    "watson.telemetry.microsoft.com", "javadl-esd-secure.oracle.com",
}
_MSDELIVERY_PAT = "|".join([
    r"geo.*\.do\.dsp\.mp\.microsoft\.com",
    r"geover.*\.do\.dsp\.mp\.microsoft\.com",
    r"kv.*\.prod\.do\.dsp\.mp\.microsoft\.com",
    r"cp.*\.prod\.do\.dsp\.mp\.microsoft\.com",
    r"disc.*\.prod\.do\.dsp\.mp\.microsoft\.com",
    r"array.*\.prod\.do\.dsp\.mp\.microsoft\.com",
    r"dl\.delivery\.mp\.microsoft\.com",
    r"download\.windowsupdate\.com",
])
_ADTRACKING_PAT = "|".join([
    r"doubleclick\.net", r"googlesyndication\.com", r"googleadservices\.com",
    r"googleads\.", r"google-analytics\.com", r"adnxs\.com",
    r"scorecardresearch\.com", r"adsystem\.",
])
_BROWSERINFRA_PAT = "|".join([
    r"safebrowsing\.google\.com", r"clients\d+\.google\.com",
    r"\.gstatic\.com", r"\.fbcdn\.net",
])
# For Bb_5 (attack signatures inside benign labels):
#   Injection patterns (XSS/SQLi/traversal) — ad/tracking false positives are excluded via _TRACK.
_INJECT_PAT = "|".join([
    r"<script", r"%3cscript", r"onerror\s*=", r"onerror%3d",
    r"javascript:", r"javascript%3a",
    r"union\s+select", r"union%20select", r"select.{1,20}from",
    r"\.\./", r"%2e%2e%2f", r"/etc/passwd", r"1=1", r"1%3d1",
    r"' or '", r"%27.{0,6}or",
])
_TRACK_PAT = r"\.gif\?|/pd\?|/i/1\.gif"

# ══════════════════════════════════════════════════════════════════════════
#  Bb_7 ~ Bb_26 : dataset-specific additional refinement rules
#  Numbering follows dataset order (ustc16 → cic17 → cic18 → iot23); Bb_27 ~ Bb_29 are rules for scans mixed into benign captures.
#    Bb_7        ustc16
#    Bb_9 ~ 11   cic17
#    (cic18 rules to be added in this block once finalized)
#    Bb_24 ~ 14  iot23
# ══════════════════════════════════════════════════════════════════════════

# ── Bb_7 (ustc16): automatic OS traffic of the infected VM ──────────────────────
#   Evidence : label_clean/ustc16/type1_malware_OS_traffic.md (294 sessions)
#   USTC-TFC2016 malware was captured on a Windows XP VM, and the automatic OS
#   traffic of that VM (name resolution/discovery/L2 control) also received the malware class label.
List_OSAuto_L7   = {"llmnr", "nbns", "nbdgm", "dhcpv6", "dhcp", "bootp",
                    "mdns", "ssdp", "browser"}
List_OSAuto_Port = {5355, 137, 138, 547, 546, 1900, 3702, 5353, 67, 68}
# dns_qry (hostname) and http_uri (path) have different forms, so the patterns are separated.
#   Applying the original "^wpad$" from the report directly to http_uri means the WPAD request
#   URI (/wpad.dat) never matches, missing 7 sessions (measured 283 → 290).
_OSAUTO_DNS_PAT = r"msftncsi|^wpad$|teredo|isatap|time\.windows\.com"
_OSAUTO_URI_PAT = r"msftncsi|wpad\.dat|teredo|isatap|time\.windows\.com"

# ── Bb_8 (cic17): remove the entire Infiltration class ──────────────────────────
#   Evidence : label_clean/cic17/type1_Infiltration.md
#     meterpreter remote-shell traces confirmed in 5 attack-labeled samples
#     (TLV_META_TYPE_*, core_channel_write, "Microsoft Windows [Version 6.0.6002]",
#      "C:\\Users\\cic2\\Downloads>"). One benign-labeled session also contains the same string.
#     However, this judgment depends on the raw payload, and session_stat has no payload
#     field, so it cannot be reproduced per session. The whole class is only ~20 sessions
#     and contributes nothing to training, so it is excluded entirely.
_INFILTRATION_PAT = r"infiltration"

# ── Bb_9 (cic17): FTP-Patator dictionary-attack argument signature ──────────────
#   Evidence : label_clean/cic17/type2_FTP-Patator.md (46 sessions)
#     PASS argument has the form "digits + literal \t (two chars: backslash+t) + dictionary word".
#     11,181 of 11,181 attack-labeled PASS (100%) have this form,
#     while 499 normal STOR-type benign PASS have 0% (all "1234") — fully separated.
#   Required field : ftp_arg  (accumulated ftp.request.arg in 03_session_stat*.py)
_FTP_PATATOR_PAT = r"\\t"

# ── Bb_10 (cic17): SSH-Patator automation-tool fingerprint ──────────────────────
#   Evidence : label_clean/cic17/type3_SSH-Patator.md (27 sessions)
#     SSH-Patator uses paramiko (2,949/2,951 = 99.9%).
#     Normal users use JSCH (1,044) / OpenSSH, so they are excluded from matching.
#   Required field : ssh_proto  (ssh.protocol in 03_session_stat*.py)
_SSH_TOOL_PAT = r"paramiko|libssh2|ganymed|dropbear"

# ── Bb_11 (cic17): HULK DoS URI signature ───────────────────────────────────────
#   Evidence : label_clean/cic17/type4_Hulk.md (1,916 sessions)
#     HULK source buildblock(random.randint(3,10)) builds 3~10 chars from uppercase A-Z only
#     and assembles the URI as "/?<blk>=<blk>". Both key and value are uppercase only.
#   Measured : Hulk label 156,545/156,546 (100.0%) matched, GoldenEye label 1 (0.0%),
#          benign 1,916 = exactly matches the label_clean confirmed count.
_HULK_URI_PAT = r"\?[A-Z]{3,10}=[A-Z]{3,10}(?:&|$)"

# ── Bb_12 (cic17): GoldenEye DoS URI signature ──────────────────────────────────
#   Evidence : label_clean/cic17/type5_GoldenEye.md (190 sessions)
#     Form: only 1~5 random parameters attached to the root ("/").
#     Distinction from normal queries: normal parameter names are pure lowercase words (output/site/page/
#     provider/partner/mapped), whereas GoldenEye keys are random strings containing
#     uppercase letters or digits. Without this condition 144 benign sessions are also caught (measured).
#   Measured : GoldenEye label 7,031/7,379 (95.3%) matched, benign 181.
#          All 181 fall inside the official attack window (Wed 11:10~11:19) → 0 false positives.
#          9 of the confirmed 190 are missed because their keys happen to be pure lowercase (conservative under-detection).
_GE_KEY = r"[A-Za-z0-9]{0,10}[A-Z0-9][A-Za-z0-9]{0,10}"
_GE_VAL = r"[A-Za-z0-9]{2,20}"
_GOLDENEYE_URI_PAT = (r"^/\?" + _GE_KEY + "=" + _GE_VAL
                      + r"(?:&" + _GE_KEY + "=" + _GE_VAL + r"){0,4}$")

# ── Bb_24 ~ Bb_26 (iot23 = CICIoT2023) ──────────────────────────────────────
#   Evidence : Neto et al., "CICIoT2023", Sensors 23(13), 5941, 2023 + official CIC page
#     (a) "In all scenarios, the attacks are performed by malicious IoT devices
#          targeting vulnerable IoT devices."      → all attacks are internal→internal
#     (b) "an ASUS router connects the network to the Internet"
#                                                  → devices are Internet-connected during experiments
#     (c) "A Gigamon Network Tap" / "passive way of accessing network traffic"
#                                                  → passive tap of the whole network
#     (d) "for each attack executed, the entire traffic captured is labeled as
#          belonging to that particular attack."   → the whole capture gets that attack label
#   ⇒ (a)+(b)+(c)+(d) make it inevitable that always-on device background is mixed into attack captures.

# For Bb_25 : protocols that 'none of the 33 CICIoT2023 attack classes use as an attack vector' —
#   only pure infrastructure/discovery protocols are listed.
#   ARP is the vector of Spoofing_MITM-ArpSpoofing,
#   DNS is the vector of Spoofing_DNS-Spoofing and UDP-Flood (:53), so both are intentionally excluded.
#   (including these two deletes the attack traffic itself — confirmed by measurement)
List_SafeInfra = {
    "stp", "cdp", "lldp", "igmp", "icmpv6", "rarp",
    "dhcp", "dhcpv6", "bootp",
    "mdns", "llmnr", "nbns", "nbdgm", "ssdp", "wsdd", "coap",
    "ntp", "lsd", "browser",
}
# Bb_24 exception : CIC/UNB own public range. It is not a device-vendor cloud, so it cannot be treated as an 'external cloud'
#   and its role is unknown (not mentioned in the CICIoT2023 paper) → conservatively excluded from marking.
#   (Evidence: the official CIC-IDS2017 page lists 205.174.165.69/.70/.71/.73/.80 as CIC equipment)
CIC_LAB_NET = (205, 174, 165)

# ── Bb_13/14 (cic18 = CSE-CIC-IDS2018) ──────────────────────────────────────
#   Evidence: https://www.unb.ca/cic/datasets/ids-2018.html (official attack schedule, Table 2)
#   cic18 was labeled by attackerIP→victimIP + time window, but the official windows are narrower than the actual
#   attack durations, so attacks falling outside the windows were labeled benign.
#   Measured (all 51.2M sessions):
#     Bot(ARES) C2   47,029  python-requests→18.219.211.138:8080, 10 bot victims, bot attack day
#     BruteForce/DoS   8,847  attackerIP→victim, attack port (21/22/80) match, attack day
#   These attacker IPs are dedicated AWS Kali attack machines, so benign traffic cannot exist.
#   → benign sessions involving these IPs = attacks missed due to time-window errors (0 false positives).
CIC18_ATTACKER_IPS = {
    "18.221.219.4",   "172.31.70.4",    # FTP-Patator
    "13.58.98.64",    "172.31.70.6",    # SSH-Patator
    "18.219.211.138", "172.31.70.46",   # GoldenEye + Bot(ARES)
    "18.217.165.70",  "172.31.70.8",    # Slowloris
    "13.59.126.31",   "172.31.70.23",   # SlowHTTPTest
    "18.219.193.20",  "172.31.70.16",   # Hulk
    "18.218.115.60",  "18.219.9.1", "18.219.32.43", "18.218.55.126",
    "52.14.136.135",  "18.216.200.189", "18.216.24.42", "18.218.11.51",
    "18.218.229.235", "18.219.5.43",    # HOIC/LOIC 10 hosts + Web-Attack
    "13.58.225.34",                     # Infiltration
}

_IPV4_RE = r"^\s*(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\s*$"


def _octets(s: pd.Series):
    """IP string Series -> (valid, o1, o2, o3). If comma-joined, use the first value (outer header)."""
    first = s.str.split(",").str[0]
    ext = first.str.extract(_IPV4_RE)
    o1 = pd.to_numeric(ext[0], errors="coerce")
    o2 = pd.to_numeric(ext[1], errors="coerce")
    o3 = pd.to_numeric(ext[2], errors="coerce")
    return o1.notna(), o1, o2, o3


def _is_public(valid, o1, o2):
    """Is it a public (Internet) IPv4? Excludes RFC1918/loopback/link-local/multicast/reserved."""
    priv = ((o1 == 10) | ((o1 == 172) & (o2 >= 16) & (o2 <= 31))
            | ((o1 == 192) & (o2 == 168)) | (o1 == 127) | (o1 == 0)
            | ((o1 == 169) & (o2 == 254)) | (o1 >= 224))
    return valid & ~priv.fillna(True)


COMMON_CODES = [
    "Aa_1", "Aa_2", "Aa_3", "Aa_4", "Aa_5", "Aa_6", "Aa_7",
    "Bb_1", "Bb_2", "Bb_3", "Bb_4", "Bb_5", "Bb_6",
    "Bb_7", "Bb_8", "Bb_9", "Bb_10", "Bb_11", "Bb_12",
    "Bb_13", "Bb_14", "Bb_15", "Bb_16", "Bb_17", "Bb_18",
    "Bb_19", "Bb_20", "Bb_21", "Bb_22", "Bb_23",
    "Bb_24", "Bb_25", "Bb_26", "Bb_27", "Bb_28", "Bb_29",
    "Cc_1", "Cc_2", "Cc_3", "Cc_4", "Cc_5",
    "Cd_1", "Cd_2", "Cd_3", "Cd_4",
    "De_1", "De_2", "De_3", "De_4", "Df_1",
]


# Class-scoped rules Bb_28 and Bb_29 remove single-packet SYN-only scans only inside the benign classes listed here
# (lower-case substrings of the class column). The lists are empty for the eight public datasets, so the two rules
# mark nothing there; fill them in for a dataset whose benign captures contain such scans.
BB_CLASS_SCOPE = {"Bb_28": [], "Bb_29": []}


def _in_scope(class_nm, code):
    names = BB_CLASS_SCOPE.get(code) or []
    if not names:
        return class_nm.str.len() < 0                     # all False, same index
    return class_nm.str.contains("|".join(re.escape(n) for n in names))


class AllNoiseRules(BaseNoiseRules):
    dataset_name = "all"

    def mark(self, df: pd.DataFrame) -> list:
        L3       = self._scol(df, "L3")
        L4       = self._scol(df, "L4")
        L7       = self._scol(df, "L7")
        dst_ip   = self._scol(df, "dst_ip")
        src_port = self._num(df, "src_port", -1).astype(int)
        dst_port = self._num(df, "dst_port", -1).astype(int)
        ws_info  = self._scol(df, "ws_col_info")
        has_SYN      = self._num(df, "has_SYN", 0).astype(int)
        has_SYN_ACK  = self._num(df, "has_SYN_ACK", 0).astype(int)
        has_FIN_cli  = self._num(df, "has_FIN_cli", 0).astype(int)
        has_FIN_serv = self._num(df, "has_FIN_serv", 0).astype(int)
        has_SHLO     = self._num(df, "has_TLS_SHLO", 0).astype(int)
        has_CHLO     = self._num(df, "has_TLS_CHLO", 0).astype(int)
        tls_sni  = self._scol(df, "tls_sni")
        http_uri = self._scol(df, "http_uri")
        http_ua  = self._scol(df, "http_ua")
        dns_qry  = self._scol(df, "dns_qry")
        # For Bb_5 (attack signatures inside benign labels) — datasets lacking these get defaults, so it never fires
        has_botid  = self._num(df, "has_botid", 0).astype(int)
        upload_fn  = self._scol(df, "upload_filename")
        susp_uri   = self._scol(df, "http_susp_uri")
        # Label (class) — for Aa_6 (unlabeled/double) and Bb (class exceptions)
        task1 = self._scol(df, "task1")
        task2 = self._scol(df, "task2")
        task3 = self._scol(df, "task3")
        class_nm = task1 + "_" + task2 + "_" + task3

        M = {}
        # ── Aa (quality) ─────────────────────────────────────────────────────
        # Aa_1: a normal 3-way needs both SYN and SYN-ACK → mark if either is missing
        M["Aa_1"] = (L4 == "tcp") & ~((has_SYN == 1) & (has_SYN_ACK == 1))
        # Aa_2: sessions with corruption/missing/truncation warnings (all WARN)
        M["Aa_2"] = ws_info.str.contains(_WARN_PAT, regex=True, na=False)
        # Aa_3: TCP sessions with no FIN in either direction
        M["Aa_3"] = (L4 == "tcp") & (has_FIN_cli == 0) & (has_FIN_serv == 0)
        # Aa_4: Malformed packets
        M["Aa_4"] = ws_info.str.contains("malformed", na=False)
        # Aa_5: TLS but ClientHello 'or' ServerHello missing (actual TLS sessions only)
        is_tls = ((L7 == "tls") | (src_port == 443) | (dst_port == 443)
                  | (has_CHLO == 1) | (has_SHLO == 1))
        M["Aa_5"] = (L4 == "tcp") & is_tls & ((has_CHLO == 0) | (has_SHLO == 0))
        # ── Cc (network control) ────────────────────────────────────────────
        M["Cc_1"] = (L7.isin(List_L3Control) | (L3 == "ipv6")
                     | L3.isin({"arp", "rarp"}) | L4.isin({"icmp", "icmpv6"}))
        M["Cc_2"] = L7.isin(List_NameResolve)
        M["Cc_3"] = L7.isin(List_Windows)
        M["Cc_4"] = L7.isin(List_InfraMgmt)              # pure (no VoIP exception)
        oct1 = pd.to_numeric(dst_ip.str.split(".").str[0], errors="coerce")
        M["Cc_5"] = ((dst_ip == "255.255.255.255") | dst_ip.str.endswith(".255")
                     | ((oct1 >= 224) & (oct1 <= 239))
                     | dst_ip.str.startswith("ff")         # IPv6 multicast (ff00::/8, ff02::, etc.)
                     | dst_ip.str.startswith("fe80")       # IPv6 link-local
                     | dst_ip.str.startswith("169.254."))  # IPv4 link-local
        # ── Cd (background) ─────────────────────────────────────────────────
        M["Cd_1"] = (L7.isin({"bt-dht", "bt-tracker"})
                     | src_port.isin([6969, 61009]) | dst_port.isin([6969, 61009]))
        # msftncsi = Windows NCSI connectivity probe (appears in dns_qry / http_uri).
        #   Note: rarely (≈3%) a session mixes DGA/C2 queries into the same DNS stream; session-level
        #         removal may lose signal (relevant only for DGA datasets).
        M["Cd_2"] = ((http_ua == "winhttp") | tls_sni.isin(List_WinBackground)
                     | dns_qry.str.contains("msftncsi", na=False)
                     | http_uri.str.contains("msftncsi", na=False))
        M["Cd_3"] = tls_sni.str.contains(_MSDELIVERY_PAT, regex=True, na=False)
        M["Cd_4"] = http_ua.str.contains("apt-http")
        # ── De (CDN/SSO) ───────────────────────────────────────────────────
        M["De_1"] = ((tls_sni.str.contains("gstatic") | tls_sni.str.contains("google"))
                     & ~tls_sni.str.contains("googlevideo") & L7.isin({"tls", "gquic"}))
        M["De_2"] = tls_sni.str.contains("mail")
        M["De_3"] = tls_sni.str.contains(_ADTRACKING_PAT, regex=True, na=False)
        M["De_4"] = tls_sni.str.contains(_BROWSERINFRA_PAT, regex=True, na=False)
        # ── Df (browser-dependent) ──────────────────────────────────────────
        M["Df_1"] = http_uri.str.contains("msn.com") | http_uri.str.contains("bing")

        # ── Aa_6 / Aa_7 (label/protocol quality) ────────────────────────────
        # Aa_6: task1 label error (unlabeled / double) → remove improperly labeled sessions
        #        (notably cic17: task1 contains 'unlabeled','double')
        M["Aa_6"] = task1.isin({"unlabeled", "double"})
        # Aa_7: IPv6 session → feat_a (IPv4-header based) cannot build a flow, extraction fails → remove
        M["Aa_7"] = (L3 == "ipv6")

        # ── Bb (dataset/class-dependent special rules; merged from Eg) ──────
        IS_VOIP = class_nm.str.contains("voip")
        IS_P2P  = class_nm.str.contains("bittorrent") | (task2 == "p2p")
        # Bb_1: infrastructure-management protocols (snmp/ntp/rtcp/lsd/chargen/stun),
        #        except stun/rtcp in the VoIP class, which are signal
        M["Bb_1"] = L7.isin(List_InfraMgmt) & ~(IS_VOIP & L7.isin({"stun", "rtcp"}))
        # Bb_2: P2P background (bt-dht/bt-tracker or port 6969/61009),
        #        except the P2P (BitTorrent) class, where it is signal
        M["Bb_2"] = (
            L7.isin({"bt-dht", "bt-tracker"})
            | src_port.isin([6969, 61009]) | dst_port.isin([6969, 61009])
        ) & ~IS_P2P
        # Bb_3: class-protocol mismatch (SCP but not ssh / FTP but sip) → label error
        M["Bb_3"] = (
            (class_nm.str.contains("scp") & (L7 != "ssh")) |
            (class_nm.str.contains("ftp") & (L7 == "sip"))
        ) & ~L7.isin(_List_NetControl)
        # Bb_4: SNI mismatches the label domain (task3) → label error (domain-classification datasets)
        _has_sni = (tls_sni != "") & (tls_sni != "-")
        _has_dom = (task3 != "") & (task3 != "-")
        _valid = _has_sni & _has_dom
        _exact = tls_sni == task3
        _ends = pd.Series(False, index=df.index)
        if _valid.any():
            _ends.loc[_valid] = [
                s.endswith("." + d) for s, d in zip(tls_sni[_valid], task3[_valid])
            ]
        # [Scope] only datasets whose label (task3) is a domain — _valid is that gate.
        M["Bb_4"] = _valid & ~(_exact | _ends)
        # Bb_5: task1=Benign but has a high-precision attack signature → attack mixed into benign (mislabel).
        #        Only high-precision signals are used (to avoid false positives): has_botid / upload_filename / injection URI.
        #        Low-precision signals (tls_heartbeat=benign CDN, ssh/ftp=normal use, ad susp_uri) are
        #        intentionally excluded — using them would wrongly delete benign sessions en masse.
        _is_benign  = task1 == "benign"
        # Scope gate. Every Bb rule must specify one of the three below.
        #   [All attacks]    _is_attack        — all attack classes
        #   [Specific class] class_nm.contains — only the specified classes
        #   [Benign]         _is_benign        — benign labels only
        _is_attack  = ~_is_benign
        _has_upload = (upload_fn != "") & (upload_fn != "-") & (upload_fn != "0")
        _inject = (susp_uri.str.contains(_INJECT_PAT, regex=True, na=False)
                   & ~susp_uri.str.contains(_TRACK_PAT, regex=True, na=False))
        M["Bb_5"] = _is_benign & ((has_botid == 1) | _has_upload | _inject)
        # Bb_6: gateway VPN health-check beacon — single-packet UDP/7001 sent by one host to the CGN (100.64/10, RFC6598)
        #        range, which inherits the active attack-window label (not an attack).
        _o1 = pd.to_numeric(dst_ip.str.split(".").str[0], errors="coerce")
        _o2 = pd.to_numeric(dst_ip.str.split(".").str[1], errors="coerce")
        _cgn = (_o1 == 100) & (_o2 >= 64) & (_o2 <= 127)
        # [Scope] all attack classes — rule that removes sessions that inherited the attack-window label.
        #        The same beacon in benign-labeled sessions is correctly labeled, so it is left untouched.
        M["Bb_6"] = _is_attack & (L4 == "udp") & _cgn & (dst_port == 7001)

        # ── Bb_7 (ustc16): automatic OS traffic of the infected VM ──────────
        # [Scope] all attack classes — automatic OS traffic of the infected VM that
        #        inherited the malware label. The same traffic in benign classes is correctly labeled.
        M["Bb_7"] = _is_attack & (
            L7.isin(List_OSAuto_L7)
            | src_port.isin(List_OSAuto_Port) | dst_port.isin(List_OSAuto_Port)
            | L3.isin({"arp", "rarp"}) | (L4 == "icmpv6")
            | dns_qry.str.contains(_OSAUTO_DNS_PAT, regex=True, na=False)
            | http_uri.str.contains(_OSAUTO_URI_PAT, regex=True, na=False)
        )
        # ── Bb_8 (cic17): remove the entire Infiltration class ──────────────
        M["Bb_8"] = class_nm.str.contains(_INFILTRATION_PAT, regex=True, na=False)

        # ── Bb_9~11 (cic17): attack-tool fingerprints mixed into benign labels ──
        #    All gated on task1=benign. Attack-labeled sessions are left untouched.
        ftp_arg   = self._scol(df, "ftp_arg")
        ssh_proto = self._scol(df, "ssh_proto")
        uri_raw   = self._scol_raw(df, "http_uri")     # case must be preserved
        M["Bb_9"]  = _is_benign & ftp_arg.str.contains(_FTP_PATATOR_PAT,
                                                       regex=True, na=False)
        M["Bb_10"]  = _is_benign & ssh_proto.str.contains(_SSH_TOOL_PAT,
                                                         regex=True, na=False)
        _hulk = uri_raw.str.contains(_HULK_URI_PAT, regex=True, na=False)
        M["Bb_11"] = _is_benign & _hulk
        M["Bb_12"] = _is_benign & ~_hulk & uri_raw.str.contains(
            _GOLDENEYE_URI_PAT, regex=True, na=False)

        # ── Bb_13~14 (cic18): attacks labeled benign due to time-window errors ──
        src_ip18 = self._scol(df, "src_ip")
        src_ip = src_ip18
        _atk = src_ip18.isin(CIC18_ATTACKER_IPS) | dst_ip.isin(CIC18_ATTACKER_IPS)
        # Bb_13 [Benign]: benign sessions involving documented attacker IPs = missed attacks.
        M["Bb_13"] = _is_benign & _atk
        # Bb_14 [Benign]: recon/scanners mixed into benign (targeting the victim network, confirmed by UA/DNS bytes).
        _recon = dns_qry.str.contains(r"version\.bind|hostname\.bind|id\.server",
                                      regex=True, na=False)
        _scanua = http_ua.str.contains(
            r"zgrab|masscan|nmap|nikto|sqlmap|dirbuster|gobuster|hydra|nuclei",
            regex=True, na=False)
        # SIP scanners (friendly-scanner/SIPVicious) — sip_ua column. cic18 benign 80,951.
        sip_ua = self._scol(df, "sip_ua")
        _sipscan = sip_ua.str.contains(r"friendly-scanner|sipvicious|sundayddr",
                                       regex=True, na=False)
        M["Bb_14"] = _is_benign & (_recon | _scanua | _sipscan)

        # ── Bb_15~23 (cic18 Distrinet corrections + pcap verification) ──────
        #   Evidence: distrinet-research.be/CNS2022/CSECICIDS2018 (KU Leuven)
        #         + full session_stat scan + cross-check against actual pcaps.
        cls_l = class_nm                              # lowercase (already lowered by _scol)
        dstp  = dst_port                              # int
        pay   = self._num(df, "payload_size", 0)
        dur   = (self._num(df, "ts_last", 0) - self._num(df, "ts_first", 0))
        has_rst = self._num(df, "has_RST", 0).astype(int)
        uri_l = http_uri

        # [A] labeled as attack but no attack payload ─────────────────────────
        # Bb_15 [Specific class] FTP-Patator, all: port 21 closed, SYN->RST, 0 USER/PASS.
        M["Bb_15"] = cls_l.str.contains("ftp-patator")
        # Bb_16 [Specific class] Slowhttptest, all: misfired at port 21, no successful attack.
        M["Bb_16"] = cls_l.str.contains("slowhttptest")
        # Bb_17 [Specific class] SSH-Patator flows misfired at port 21 (Distrinet: 30 flows).
        M["Bb_17"] = cls_l.str.contains("ssh-patator") & (dstp == 21)
        # Bb_18 [Specific class] LOIC-UDP flows that are mis-processed ICMP (old CICFlowMeter bug).
        M["Bb_18"] = cls_l.str.contains("loic-udp") & (L4 == "icmp")

        # [B] genuine attacks but artifacts/partial contamination ──────────────
        # Bb_19 [Specific class] Slowloris empty flows from connection splitting (120s timeout split).
        M["Bb_19"] = cls_l.str.contains("slowloris") & (pay == 0)
        # Bb_20 [Specific class] GoldenEye early-RST 'attempt' flows (dur<5.05s & RST).
        M["Bb_20"] = (cls_l.str.contains("goldeneye") & (has_rst == 1) & (dur < 5.05))
        # Bb_21 [Specific class] Web-Attack page-resource requests (css/img/js/ico).
        _res = uri_l.str.contains(r"\.(?:css|js|png|jpe?g|gif|ico|woff2?|svg)(?:$|\?)",
                                  regex=True, na=False)
        M["Bb_21"] = cls_l.str.contains("web-attack") & _res
        # Bb_22 [Specific class] Bot empty flows from TCP-seg-offset.
        M["Bb_22"] = cls_l.str.contains("_bot") & (pay == 0)

        # [C] attacks mixed into the benign class (in addition to Bb_13/14) ───
        # Bb_23 [Benign] Infiltration NMAP internal port scan (infected victim scans 21 internal hosts).
        #   Distrinet: 172.31.69.24 / .13 nmap-scan 172.31.69.x, originally labeled benign.
        _inf_src = src_ip.isin({"172.31.69.24", "172.31.69.13"})
        _internal_dst = dst_ip.str.startswith("172.31.69.")
        M["Bb_23"] = _is_benign & _inf_src & _internal_dst & (has_SYN == 1) & (has_SYN_ACK == 0)


        # ── Bb_24~14 (iot23): attack-like background traffic inside the testbed ──
        src_ip = self._scol(df, "src_ip")
        _sv, _so1, _so2, _so3 = _octets(src_ip)
        _dv, _do1, _do2, _do3 = _octets(dst_ip)
        _s_pub = _is_public(_sv, _so1, _so2)
        _d_pub = _is_public(_dv, _do1, _do2)
        _cic = (((_so1 == CIC_LAB_NET[0]) & (_so2 == CIC_LAB_NET[1]) & (_so3 == CIC_LAB_NET[2]))
                | ((_do1 == CIC_LAB_NET[0]) & (_do2 == CIC_LAB_NET[1]) & (_do3 == CIC_LAB_NET[2]))
                ).fillna(False)
        # Bb_24: one side is a public IP = communication with the Internet outside the testbed.
        #        Attacks are internal→internal (evidence a), so it cannot be that attack. The CIC range is excluded.
        # [Scope] all attack classes — removes only device background mixed into attack captures.
        #        benign was captured separately "when there were no attacks" (16h per the paper), so its label is correct.
        M["Bb_24"] = _is_attack & (_s_pub | _d_pub) & ~_cic
        # Bb_26: destination is broadcast/multicast/link-local → cannot be a targeted attack
        #   [Note] dst_ip of encapsulated sessions may be joined as "outer_header,inner_header".
        #   Applying string checks (endswith/startswith) to the raw value tests the 'inner header' (GRE
        #   flood random spoofed IP), producing ~1/256 false positives.
        #   (measured: Mirai_Greeth 9,941 / Greip 4,637 false positives) → always use only the first value.
        _dst1 = dst_ip.str.split(",").str[0].str.strip()
        _bc = ((_dst1 == "255.255.255.255") | _dst1.str.endswith(".255")
               | ((_do1 >= 224) & (_do1 <= 239)).fillna(False)
               | _dst1.str.startswith("ff") | _dst1.str.startswith("fe80")
               | _dst1.str.startswith("169.254."))
        # [Scope] all attack classes
        M["Bb_26"] = _is_attack & _bc & ~_cic
        # Bb_25: internal↔internal pure infrastructure/discovery protocols that are not attack vectors
        _internal = _sv & _dv & ~_s_pub & ~_d_pub
        # [Scope] all attack classes
        M["Bb_25"] = (_is_attack & _internal & ~M["Bb_26"] & ~_cic
                      & L7.isin(List_SafeInfra))

        # ── Bb_27 ~ Bb_29: scans mixed into benign captures ─────────────────
        is_nmap = self._num(df, "is_nmap_probe", 0).astype(int)
        # Bb_27 [Benign]: nmap UDP probes = scans mixed into benign labels (type1 tier A).
        #   is_nmap_probe is determined byte-wise by 03_session_stat using nmap-payloads 7.90.
        M["Bb_27"] = _is_benign & (is_nmap == 1)
        # Bb_28 [Specific class]: single-shot TCP SYN-only scans in the BB_CLASS_SCOPE['Bb_28'] class.
        #   Removed only from this class via the class condition — no effect on other attack classes.
        M["Bb_28"] = (_in_scope(class_nm, "Bb_28")
                      & (L4 == "tcp") & (has_SYN == 1) & (has_SYN_ACK == 0)
                      & (self._num(df, "pkt_count", 0).astype(int) == 1))
        # Bb_29 [Specific class]: SYN-only port sweeps in the BB_CLASS_SCOPE['Bb_29'] class (IoT botnet target ports 23/2323/37215/445, etc.).
        M["Bb_29"] = (_in_scope(class_nm, "Bb_29")
                      & (has_SYN == 1) & (has_SYN_ACK == 0)
                      & (self._num(df, "pkt_count", 0).astype(int) == 1))

        return [(c, M[c].fillna(False)) for c in COMMON_CODES]