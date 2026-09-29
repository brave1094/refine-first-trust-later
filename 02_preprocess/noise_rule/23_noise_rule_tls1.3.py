"""
02_preprocess/noise_rule/23_noise_rule_tls1.3.py
──────────────────────────────────────────────
'Default refinement rule list' applied to the tls1.3 dataset.

  All actual marking logic lives in noise_rule_all.py (04 marks every rule);
  this file only defines, for when 05 builds denoised without --new_rule,
  'the default refinement rule list of this dataset (default_clean_rules)'.
"""

from .base import BaseNoiseRules


class Tls13ClassRules(BaseNoiseRules):
    dataset_name = "tls1.3"
    default_clean_rules = [
        # Aa_2 re-applied (2026-09-06, author decision): the 2026-08-15 exclusion reason (Aa_2
        #   over-removed "tcp previous segment not captured"+"tcp dup ack" = benign TCP artifacts,
        #   47.7%) was invalidated by the 2026-08-20 Aa_2 redefinition — the current Aa_2 WARN
        #   judges only {size-limited, malformed, truncated} and dup-ack/prev-segment are
        #   intentionally excluded (noise_rule_all.py). So it catches only real capture damage and is safe.
        "Aa_2", # capture damage·truncation (redefined, excl. dup-ack/prev-seg)
        "Aa_4", # malformed
        "Aa_6", # label error
        "Aa_7", # IPv6
        "Bb_4", # SNI-domain mismatch (label error)
        "Cc_1", # L3 control
        "Cc_2", # name resolution
        "Cc_3", # Windows
        "Cc_4", # infrastructure management
        "Cc_5", # broadcast
        "Cd_1", # P2P
        "Cd_2", # Windows background
        "Cd_3", # MS delivery
        "Cd_4", # apt-http
        "De_1", # Google
        "De_2", # mail
        "De_3", # ad tracking
        "De_4", # browser infrastructure
        "Df_1", # msn/bing
    ]

    def mark(self, df):
        # no marking logic (all in noise_rule_all). Provides the list only.
        return []
