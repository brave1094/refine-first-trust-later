"""
02_preprocess/noise_rule/01_noise_rule_ustc16.py
──────────────────────────────────────────────
'Default refinement rule list' to apply to the ustc16 dataset.

  All actual marking logic lives in noise_rule_all.py (04 marks all rules);
  this file only defines, for use by 05 when building denoised without --new_rule,
  'the default refinement rule list of this dataset (default_clean_rules)'.

[Additional rules]
  Bb_7  automatic OS traffic of the infected VM (Windows XP)
        Evidence: label_clean/ustc16/type1_malware_OS_traffic.md — 294 sessions.
        USTC-TFC2016 malware was captured on a Windows XP VM, and the automatic OS
        traffic of that VM (llmnr/nbns/mdns/ssdp/dhcp/arp/icmpv6 + msftncsi/wpad/teredo queries)
        also received the malware class label.
"""

from .base import BaseNoiseRules


class Ustc16ClassRules(BaseNoiseRules):
    dataset_name = "ustc16"
    default_clean_rules = [
        "Aa_2",  # corrupted/truncated
        "Aa_4",  # malformed
        "Aa_6",  # label error (unlabeled/double)
        "Aa_7",  # IPv6
        "Cc_5",  # broadcast/multicast/link-local destination
        "Cd_2",  # Windows background + msftncsi OS probe
        "Bb_7",  # automatic OS traffic of the infected VM (type1, confirmed 294 / reproduced 290)
    ]

    def mark(self, df):
        # No marking logic (all in noise_rule_all). Provides the list only.
        return []
