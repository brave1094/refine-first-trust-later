# Datasets (step 1)

The captures are **not redistributed** here. Download them from their publishers, cite the dataset papers, and follow
each publisher's terms. This repository provides, for every public dataset, the scripts that arrange the official
release into the layout of the pipeline, the per-class counts to check the result against, and the content keys of all
sessions used in the article (`01_dataset/<dataset>/session_keys.csv`).

The two private datasets of the article (Private #1 and Private #2) cannot be shared; no data, list, or result of them
is included.

## Overview

| ID | Dataset | Paper | Official download | Classes* | Sessions* |
|---|---|---|---|---:|---:|
| vpn16 | ISCX VPN-nonVPN 2016 | Draper-Gil et al., ICISSP 2016 | <https://www.unb.ca/cic/datasets/vpn.html> | 15 | 305,301 |
| tor16 | ISCX Tor-nonTor 2016 | Lashkari et al., ICISSP 2017 | <https://www.unb.ca/cic/datasets/tor.html> | 15 | 62,593 |
| tls1.3 | CSTNET-TLS 1.3 | Lin et al., WWW 2022 (ET-BERT) | <https://github.com/linwhitehat/ET-BERT> (folder `CSTNET-TLS 1.3`) | 120 | 46,382 |
| cispec | CipherSpectrum | Wickramasinghe et al., IEEE S&P 2025 | <https://cgi.cse.unsw.edu.au/~cspectrum/> (short form, CC BY-NC 4.0) | 41 | 123,000 |
| ustc16 | USTC-TFC2016 | Wang et al., ICOIN 2017 | <https://github.com/yungshenglu/USTC-TFC2016> | 20 | 595,783 |
| cic17 | CIC-IDS2017 | Sharafaldin et al., ICISSP 2018 | <https://www.unb.ca/cic/datasets/ids-2017.html> | 15 | 1,571,244 |
| cic18 | CSE-CIC-IDS2018 | Sharafaldin et al., ICISSP 2018 | <https://www.unb.ca/cic/datasets/ids-2018.html> (AWS) | 14 | 51,234,354 |
| iot23 | CIC IoT Dataset 2023 | Neto et al., Sensors 2023 | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | 30 | 1,155,285,412 |

\* task-3 classes and sessions before refinement (Table 1 of the article); per-class counts are in
`99_documents/results/dataset_stats/<dataset>/01_origin_<dataset>.csv`.

## Layout expected by the pipeline

```
<NM_DATASET_ROOT>/                     default: <repository>/00_assets/datasets
└── <dataset dir>/                     e.g. 21_ISCX-VPN-2016 (see 00_assets/datasets/_tools/expected_layout.json)
    └── 01_pcap/<class>/<class>_<i>.pcap        class = <task1>_<task2>_<task3>, e.g. VPN_Chat_AIM_none
```
Steps 3-6 write `02_session/`, `03_session_stat/`, `04_session_noisy_labeled/` next to `01_pcap/`.

## Per dataset

| ID | What to download | How to arrange | Rule reliability |
|---|---|---|---|
| vpn16 | the VPN and NonVPN PCAP archives | `python 00_assets/datasets/_tools/arrange_pcaps.py --dataset vpn16 --src <extracted>` | reconstructed keyword rules — check with `verify_counts.py` |
| tor16 | the Tor and NonTor PCAP archives (keep the `Tor/`, `NonTor/` folders) | `arrange_pcaps.py --dataset tor16 --src <extracted>` | reconstructed keyword rules — check with `verify_counts.py` |
| tls1.3 | the CSTNET-TLS 1.3 pcaps (one folder per domain) | `arrange_pcaps.py --dataset tls1.3 --src <extracted>` | exact (domain folders) |
| cispec | `aes-128-gcm.zip`, `aes-256-gcm.zip`, `chacha20-poly1305.zip` | `python 00_assets/datasets/_tools/cispec/relabel_cispec.py`, then `cispec/fix_chacha20.py` and `cispec/verify_sni.py` | exact; the release holds **41** domains (the paper and page state 40) |
| ustc16 | `Benign/` and `Malware/` (extract the `.7z` files) | `arrange_pcaps.py --dataset ustc16 --src <repository clone>` | exact |
| cic17 | the five `*-WorkingHours*.pcap` files and the labelled flows (`TrafficLabelling` CSVs, "GeneratedLabelledFlows") | `arrange_pcaps.py --dataset cic17 --src <pcaps>`; labels are joined in step 3 (`CIC17_LABELS=<csv folder>`) | exact |
| cic18 | `Original Network Traffic and Log data/<Day>/pcap.zip` for the nine days (e.g. `aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/" <dst>`) | `python 00_assets/datasets/_tools/cic18/apply_rename_map.py --src <extracted day folders>`; labels from the official attack schedule in step 3 | exact (4,006-row name map) |
| iot23 | the PCAP folders (one per attack, plus benign) | `arrange_pcaps.py --dataset iot23 --src <extracted>` | normalised folder-name match — check with `verify_counts.py` |

pcapng files are converted to classic pcap with `python 00_assets/datasets/_tools/convert_pcapng.py --dataset <id>` (needs
`editcap`), because the session statistics use dpkt.

## Checking the arrangement

After step 3, `python 00_assets/datasets/_tools/verify_counts.py --dataset <id>` compares your per-class session counts with
ours. Differences point to captures that are misplaced, missing, or extra; files that `arrange_pcaps.py` could not
assign are listed by it and can be placed by hand (the file name does not matter).

## Using the released session lists with your file names

The released lists name sessions by our file names. `python 02_preprocess/align_session_lists.py --dataset <id>` recomputes a
content key for every session of your `03_session_stat` — transport protocol, the two endpoints, and the first-packet
time in microseconds, which does not depend on file names or on the tshark version — and rewrites the lists to your
names (originals kept as `list_*.published.csv`). Aligning our own data to itself reproduces every row exactly.

Note on tor16: the official release contains identical copies of some captures, so identical sessions occur more than
once (up to four times) and some test sessions have an identical copy in the training list (7-12 % of the test lists).
Excluding them changes the tor16 result of Illusion 1 from +2.15 to +1.96 pp (95 % CI [1.38, 2.53]; all seven models
positive). The alignment keeps the copies one-to-one.
