"""
02_preprocess/noise_rule/03_noise_rule_cic18.py
──────────────────────────────────────────────
Default refinement rule list for cic18 (CSE-CIC-IDS2018).

  Evidence: (1) Distrinet (KU Leuven) CNS2022 correction document
             https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html
        (2) full session_stat scan (51.2M) + cross-check against actual pcaps
        (3) official schedule https://www.unb.ca/cic/datasets/ids-2018.html

[A] Attack label without attack payload (removed)
  Bb_15 FTP-Patator all     190,300  port 21 closed, SYN→RST, 0 USER/PASS (pcap verified)
  Bb_16 Slowhttptest all    105,550  misfired at port 21, no successful attack (pcap verified)
  Bb_17 SSH-Patator port 21     30   exactly matches Distrinet "30 flows mislabeled"
  Bb_18 LOIC-UDP ICMP          85   old CICFlowMeter mis-processes ICMP as UDP
[B] Attacks, but artifacts/partial contamination
  Bb_19 Slowloris empty flow  2,146   fragments from 120s-timeout connection splitting
  Bb_20 GoldenEye RST attempt ~6,700  dur<5.05s & RST = incomplete attempts under overload
  Bb_21 Web-Attack resources  few     css/img/js/ico page-resource requests
  Bb_22 Bot empty flow          258   TCP-seg-offset fragments
[C] Attacks mixed into benign
  Bb_13 attacker-IP benign    59,908   Bot C2 + attacks missed outside the time window (incl. Infiltration traffic)
  Bb_14 scanner/recon benign  82,298   friendly-scanner/version.bind/zgrab/masscan
  Bb_23 Infiltration internal scan     nmap internal port scan by infected victims (172.31.69.24/.13)
"""

from .base import BaseNoiseRules


class Cic18ClassRules(BaseNoiseRules):
    dataset_name = "cic18"
    default_clean_rules = [
        "Aa_2", "Aa_4", "Aa_6", "Aa_7",         # quality
        "Bb_13", "Bb_14",                        # [C] attacks within benign
        "Bb_15", "Bb_16", "Bb_17", "Bb_18",     # [A] attack label on non-attacks
        "Bb_19", "Bb_20", "Bb_21", "Bb_22",     # [B] artifacts/partial contamination
        "Bb_23",                                 # [C] Infiltration internal scan
    ]

    def mark(self, df):
        return []
