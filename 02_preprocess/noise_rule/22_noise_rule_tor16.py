"""
02_preprocess/noise_rule/22_noise_rule_tor16.py
──────────────────────────────────────────────
The 'default refinement rule list' applied to the tor16 dataset.

  All marking logic lives in noise_rule_all.py (04 marks all rules);
  this file only defines, for 05 to apply when building denoised without --new_rule,
  'the default refinement rule list of this dataset (default_clean_rules)'.
"""

from .base import BaseNoiseRules


class Tor16ClassRules(BaseNoiseRules):
    dataset_name = "tor16"
    default_clean_rules = [
        "Aa_1", # incomplete 3-way handshake
        "Aa_2", # damaged/truncated
        "Aa_4", # malformed
        "Aa_5", # missing TLS handshake
        "Aa_6", # label error
        "Aa_7", # IPv6
        "Bb_1", # infrastructure management (VoIP exception)
        "Bb_2", # P2P background (P2P exception)
        "Cc_1", # L3 control
        "Cc_2", # name resolution
        "Cc_3", # Windows
        "Cc_5", # broadcast
        "Cd_2", # Windows background
        "Cd_3", # MS delivery
        "Cd_4", # apt-http
        "De_1", # Google
        # "De_2" excluded — SNI 'mail' matching wrongly removes the signal of mail classes such as Gmail
        "De_3", # ad tracking
        "De_4", # browser infrastructure
        "Df_1", # msn/bing
    ]

    def mark(self, df):
        # no marking logic here (all in noise_rule_all). Only provides the list.
        return []
