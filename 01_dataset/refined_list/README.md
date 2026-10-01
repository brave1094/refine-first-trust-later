# Refined session lists

The sessions of each public dataset that remain after the default refinement rules of the dataset, with the rule
engine of this repository (`02_preprocess/noise_rule/`, rules listed in `default_clean_rules` of
`noise_rule/<NN>_noise_rule_<ds>.py`). They are the refined sets behind Table 8 of the article; the per-class counts
match `99_documents/results/dataset_stats/<ds>/02_refined_<ds>.csv` exactly (`00_stat/check.txt`).

| Folder | Content |
|---|---|
| `<ds>/refined_<ds>_<k>.csv` | `filename`, `proto`, `stream` of every refined session, 1,000,000 rows per file (`k` = 1, 2, ...) |
| `00_stat/<ds>.csv` | per class (task 3): sessions before and after refinement, and the total |
| `00_stat/check.txt` | comparison with the released Table 8 counts |

A session is identified by the capture file and its transport stream:
- `filename`: the file in `01_pcap/` (cic17, cic18, iot23, ustc16: one capture holds many sessions) or the session file
  written by step 3 in `02_session/` (vpn16, tor16, tls1.3, cispec), with the names given by `02_preprocess/00_rename.py`;
- `proto`, `stream`: `tcp` or `udp` and the tshark stream number (`tcp.stream` / `udp.stream`) inside that file.

Here: vpn16, tor16, tls1.3, cispec, ustc16, cic17. The lists of cic18 (50,081,816 sessions, 51 files) and iot23
(1,152,187,614 sessions, 1,153 files) are too large for this repository and are on Zenodo: [doi:10.5281/zenodo.23082133](https://doi.org/10.5281/zenodo.23082133)
(`00_stat/cic18.csv` and `00_stat/iot23.csv` are here).
