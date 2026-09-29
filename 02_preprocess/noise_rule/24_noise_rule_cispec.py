"""
02_preprocess/noise_rule/24_noise_rule_cispec.py
──────────────────────────────────────────────
The 'default refinement rule list' applied to the cispec dataset.

  All actual marking logic lives in noise_rule_all.py (04 marks every rule);
  this file only defines, for use when 05 builds denoised without --new_rule,
  the 'default refinement rule list of this dataset (default_clean_rules)'.
"""

from .base import BaseNoiseRules


class CispecClassRules(BaseNoiseRules):
    dataset_name = "cispec"
    default_clean_rules = [
        "Aa_2", # damaged/truncated
        "Aa_4", # malformed
        "Aa_5", # missing TLS handshake
        "Aa_6", # label error
        "Aa_7", # IPv6
        "Cc_1", # L3 control
        "Cc_2", # name resolution
        "Cc_3", # Windows
        "Cc_4", # infrastructure management
        "Cc_5", # broadcast
        "Cd_1", # P2P
        "Cd_2", # Windows background
        "Cd_3", # MS delivery
        "Cd_4", # apt-http
        "Df_1", # msn/bing
    ]

    def mark(self, df):
        # no marking logic (all in noise_rule_all). Provides the list only.
        return []
