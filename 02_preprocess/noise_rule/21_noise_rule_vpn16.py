"""
02_preprocess/noise_rule/21_noise_rule_vpn16.py
──────────────────────────────────────────────
The 'default refinement rule list' applied to the vpn16 dataset.

  All actual marking logic lives in noise_rule_all.py (04 marks every rule);
  this file only defines, for use when 05 builds denoised without --new_rule,
  the 'default refinement rule list of this dataset (default_clean_rules)'.
"""

from .base import BaseNoiseRules


class Vpn16ClassRules(BaseNoiseRules):
    dataset_name = "vpn16"
    default_clean_rules = [
        "Aa_1", # incomplete 3-way handshake (no SYN/SYN-ACK)
        "Aa_2", # damaged/truncated warning session
        "Aa_4", # malformed packet
        "Aa_5", # TLS but no ClientHello/ServerHello
        "Aa_6", # label error (unlabeled/double)
        "Aa_7", # IPv6 session
        "Bb_1", # infrastructure management protocols (VoIP stun/rtcp exempt)
        "Bb_2", # P2P background (P2P class exempt)
        "Bb_3", # class-protocol mismatch (SCP/FTP label error)
        "Cc_1", # L3 control/IPv6/ARP/ICMP
        "Cc_2", # name resolution (DNS etc.)
        "Cc_3", # Windows protocols
        "Cc_5", # broadcast/multicast
        "Cd_2", # Windows background
        "Cd_3", # MS delivery
        "Cd_4", # apt-http
        "De_1", # Google infrastructure
        "De_2", # mail
        "De_3", # ad tracking
        "De_4", # browser infrastructure
        "Df_1", # msn/bing
    ]

    def mark(self, df):
        # no marking logic (all in noise_rule_all). Provides the list only.
        return []
