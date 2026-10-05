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
| ustc16 | USTC-TFC2016 | Wang et al., ICOIN 2017 | <https://github.com/davidyslu/USTC-TFC2016> | 20 | 595,783 |
| cic17 | CIC-IDS2017 | Sharafaldin et al., ICISSP 2018 | <https://www.unb.ca/cic/datasets/ids-2017.html> | 15 | 1,571,244 |
| cic18 | CSE-CIC-IDS2018 | Sharafaldin et al., ICISSP 2018 | <https://www.unb.ca/cic/datasets/ids-2018.html> (AWS) | 14 | 51,234,354 |
| iot23 | CIC IoT Dataset 2023 | Neto et al., Sensors 2023 | <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | 30 | 1,155,285,412 |

\* classes of task 3 (the finest of the three label levels task1 > task2 > task3) and sessions before refinement (Table 1 of the article); per-class counts are in
`99_documents/results/dataset_stats/<dataset>/01_origin_<dataset>.csv`.

## Layout expected by the pipeline

```
<NM_DATASET_ROOT>/                     default: <repository>/00_assets/datasets
└── <dataset dir>/                     e.g. 21_ISCX-VPN-2016 (see 00_assets/datasets/_tools/expected_layout.json)
    └── 01_pcap/<class>/<class>_<i>.pcap        class = <task1>_<task2>_<task3> (each part may contain "_"),
                                                e.g. VPN_Chat_AIM_none: task1 VPN, task2 Chat, task3 AIM
```
Steps 3-4 write `02_session/`, `03_session_stat/` (`03_session_stat_labeled/` for cic17 and cic18) and
`04_session_noisy_labeled/` next to `01_pcap/`; step 6 writes the model inputs to `01_dataset/`.

## Per dataset

| ID | What to download | How to arrange | Rule reliability |
|---|---|---|---|
| vpn16 | the five archives NonVPN-PCAPs-01..03.zip, VPN-PCAPS-01.zip, VPN-PCAPs-02.zip | `python 02_preprocess/00_rename.py --dataset vpn16 --src <extracted>` | exact: `02_preprocess/lib_rename/vpn16.csv` (140 official files; all 139 placed files byte-identical to ours) |
| tor16 | Tor.zip and NonTor.tar.xz | `python 02_preprocess/00_rename.py --dataset tor16 --src <extracted>` | exact: `02_preprocess/lib_rename/tor16.csv` (95 official files; 106 placed files byte-identical to ours, incl. one capture cut at 16 MiB and 11 repaired copies in `.salvaged/`) |
| tls1.3 | the Google Drive folder `cstnet-tls 1.3` linked from the ET-BERT repository (one folder per domain; e.g. `rclone copy`) | `python 02_preprocess/00_rename.py --dataset tls1.3 --src <downloaded folder>` | exact: `02_preprocess/lib_rename/tls1.3.csv` (all 46,372 files byte-identical to ours) |
| cispec | `aes-128-gcm.zip`, `aes-256-gcm.zip`, `chacha20-poly1305.zip` | `python 02_preprocess/00_rename.py --dataset cispec --src <extracted>` (the scripts in `00_assets/datasets/_tools/cispec/` are the ones we used; the map gives the same result) | exact: `02_preprocess/lib_rename/cispec.csv.gz` (all 123,000 placed files byte-identical to ours; the folder `chacha20` of aes-256-gcm.zip is getpocket.com, verified by SNI); the release holds **41** domains (the paper and page state 40) |
| ustc16 | `Benign/` and `Malware/` of github.com/davidyslu/USTC-TFC2016 (extract every `.7z`) | `python 02_preprocess/00_rename.py --dataset ustc16 --src <repository clone>` | exact: `02_preprocess/lib_rename/ustc16.csv` (all 20 placed files byte-identical to ours; SMB and Weibo are the official parts merged in time order with mergecap) |
| cic17 | the five `*-WorkingHours*.pcap` files and the labelled flows (`TrafficLabelling` CSVs, "GeneratedLabelledFlows") | `python 02_preprocess/00_rename.py --dataset cic17 --src <pcaps>`; labels are joined in step 3 (`CIC17_LABELS=<csv folder>`) | one file per day (`02_preprocess/lib_rename/cic17.csv`: `<Day>-WorkingHours.pcap` -> `0N_<Day>.pcap`; size and first packet are checked) |
| cic18 | `Original Network Traffic and Log data/<Day>/pcap.zip` for the nine days (e.g. `aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/" <dst>`) | `python 02_preprocess/00_rename.py --dataset cic18 --src <extracted day folders>`; labels from the official attack schedule in step 3 | name map made when we placed the captures (`02_preprocess/lib_rename/cic18.csv`, 4,006 rows) |
| iot23 | the 34 folders of `PCAP/` (registration on the download page; 309 files, 588 GB) | `python 02_preprocess/00_rename.py --dataset iot23 --src <PCAP folder>` | `02_preprocess/lib_rename/iot23.csv`: every official file identified by its first 64 KiB (pcap header and first packets, all 309 distinct); size and first packet are checked when placed |

`00_rename.py` converts the official pcapng files to classic pcap (the session statistics use dpkt). Captures placed
without it can be converted with `python 00_assets/datasets/_tools/convert_pcapng.py --dataset <id>` (needs `editcap`).

## Refined session lists

`01_dataset/refined_list/` lists, for every public dataset, the sessions that remain after its default refinement rules
(`filename, proto, stream`; the refined sets of Table 4, identical per class). cic18 and iot23 are on Zenodo:
[doi:10.5281/zenodo.23082133](https://doi.org/10.5281/zenodo.23082133). See `01_dataset/refined_list/README.md`.

## Session lists of cic18 and iot23

The full (`00_filelist`) and sizectrl (`00_filelist_sm`) lists of cic18 and iot23 are restored from the inputs the seven
models were trained on (the xgboost feature files of 2026-08-22): one row per session, in the order of training. For
large datasets, `05_make_filelist.py` reads the shuffled captures in one bucket per `--workers`, each with its own seed,
and stops reading once every class is full; its sample therefore also depends on `--workers`, `--chunk_rows`, and
`--early_stop_mult`. These settings were not recorded for those runs, and a later run of step 5 with the same code and
data drew a different sample, so rerunning step 5 (`LISTS=rebuild`) does not give the lists of the article for these
two datasets; use the released lists (`LISTS=published`). The strat lists (`00_filelist_strat`) are the ones used for
training. The models of those runs numbered the classes differently from `label_map.json` (iot23: `SlowLoris` last;
cic18: a map of 13 classes): the class names are the same, only the order of the output units differs.

## Checking the arrangement

After step 3, `python 00_assets/datasets/_tools/verify_counts.py --dataset <id>` compares your per-class session counts with
ours. Differences point to captures that are misplaced, missing, or extra; `02_preprocess/00_rename.py` already lists the
official files it could not find (missing) or does not know (not in the map), and checks size and first packet of every file.

## Using the released session lists with your file names

The released lists name sessions by our file names. `python 02_preprocess/align_session_lists.py --dataset <id>` recomputes a
content key for every session of your `03_session_stat` — transport protocol, the two endpoints, and the first-packet
time in microseconds, which does not depend on file names or on the tshark version — and rewrites the lists to your
names (originals kept as `list_*.published.csv`). Aligning our own data to itself reproduces every row exactly.

Note on tor16: the official release contains identical copies of some captures, so identical sessions occur more than
once (up to four times) and some test sessions have an identical copy in the training list (7-12 % of the test lists).
Excluding them changes the tor16 result of Illusion 1 from +2.15 to +1.96 pp (95 % CI [1.38, 2.53]; all seven models
positive). The alignment keeps the copies one-to-one.
