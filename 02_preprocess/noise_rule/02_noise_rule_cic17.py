"""
02_preprocess/noise_rule/02_noise_rule_cic17.py
──────────────────────────────────────────────
'Default refinement rule list' to apply to the cic17 dataset.

  The actual marking logic is all in noise_rule_all.py (04 marks every rule);
  this file only defines, for when 05 builds denoised without --new_rule,
  'the default refinement rule list of this dataset (default_clean_rules)'.

[Additional rules] Bb_8 removes a class; Bb_9~12 are task1=benign gates — sessions with attack labels are untouched.
  Bb_9   FTP-Patator dictionary-attack arguments (literal \t in ftp_arg)
         Evidence: label_clean/cic17/type2_FTP-Patator.md — 46 sessions.
  Bb_10  SSH-Patator tool fingerprint (ssh_proto such as paramiko, excluding JSCH/OpenSSH)
         Evidence: label_clean/cic17/type3_SSH-Patator.md — 27 sessions.
  Bb_11  HULK DoS URI (/?UPPER3~10=UPPER3~10)
         Evidence: label_clean/cic17/type4_Hulk.md — 1,916 sessions.
         Measured reproduction: benign 1,916 (exactly matches the confirmed count),
                    Hulk label 156,545/156,546 = 100.0%.
  Bb_12  GoldenEye DoS URI (root + 1~5 random parameters, keys containing uppercase/digits)
         Evidence: label_clean/cic17/type5_GoldenEye.md — 190 sessions.
         Measured reproduction: benign 181 (all inside the official attack window Wed 11:10~11:19, 0 false positives).
                    Of the 190 confirmed, 9 are missed because their keys happen to be all lowercase.

  ※ Bb_9/Bb_10 require the ftp_arg / ssh_proto columns newly extracted
     by 03_session_stat_cic17.py. They do not fire before re-extraction.
"""

from .base import BaseNoiseRules


class Cic17ClassRules(BaseNoiseRules):
    dataset_name = "cic17"
    default_clean_rules = [
        "Aa_2",   # corrupted/truncated
        "Aa_4",   # malformed
        "Aa_6",   # label error (unlabeled/double)
        "Aa_7",   # IPv6
        "Bb_5",   # attack signatures inside benign labels (botid/upload/injection URI)
        "Bb_8",   # remove the entire Infiltration class   (type1)
        "Bb_9",   # FTP-Patator argument signature      (type2,    46 sessions)
        "Bb_10",  # SSH-Patator tool fingerprint        (type3,    27 sessions)
        "Bb_11",  # HULK DoS URI                    (type4, 1,916 sessions)
        "Bb_12",  # GoldenEye DoS URI               (type5,   190 sessions)
    ]

    def mark(self, df):
        # no marking logic here (all in noise_rule_all). Provides the list only.
        return []
