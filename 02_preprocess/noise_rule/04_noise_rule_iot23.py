"""
02_preprocess/noise_rule/04_noise_rule_iot23.py
──────────────────────────────────────────────
The 'default refinement rule list' applied to the iot23 dataset.

  All actual marking logic lives in noise_rule_all.py (04 marks every rule);
  this file only defines, for use when 05 builds denoised without --new_rule,
  the 'default refinement rule list of this dataset (default_clean_rules)'.

[Additional rules] iot23 = CICIoT2023 (CIC/UNB). Not Aposemat IoT-23 (CTU, Czech Republic).

  Evidence: Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for
        Large-Scale Attacks in IoT Environment", Sensors 23(13), 5941, 2023.
    (a) "In all scenarios, the attacks are performed by malicious IoT devices
         targeting vulnerable IoT devices."     → all attacks are internal→internal
    (b) "an ASUS router connects the network to the Internet"
                                                → devices are connected to the Internet during the experiments
    (c) "A Gigamon Network Tap" / "non-intrusive, and passive way of accessing
         network traffic"                       → the entire network is passively tapped
    (d) "for each attack executed, the entire traffic captured is labeled as
         belonging to that particular attack."  → the whole capture is given that attack label

  ⇒ The constant background traffic of 105 IoT devices is mixed as is into the attack captures, and
     all of it carries the attack label. At session granularity, this is a label error.

  Bb_13  communication with the external Internet — cannot be an internal→internal attack (device↔vendor cloud)
  Bb_14  internal infrastructure protocols     — SSDP/DHCP/NTP/mDNS etc. ARP·DNS excluded as attack vectors
  Bb_15  broadcast/multicast destination — cannot be a targeted attack

  ※ 205.174.165.0/24 (the CIC own range) has an unknown role, so it is conservatively excluded from marking.
"""

from .base import BaseNoiseRules


class Iot23ClassRules(BaseNoiseRules):
    dataset_name = "iot23"
    default_clean_rules = [
        "Aa_2",   # damaged/truncated
        "Aa_4",   # malformed
        "Aa_6",   # label error
        "Aa_7",   # IPv6
        "Bb_24",  # external Internet communication
        "Bb_25",  # internal infrastructure protocols
        "Bb_26",  # broadcast/multicast destination
    ]

    def mark(self, df):
        # no marking logic (all in noise_rule_all). Provides the list only.
        return []
