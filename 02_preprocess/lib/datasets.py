#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/datasets.py
─────────────────────────────────────────────────────────────────────────────
Central dataset registry (path resolution only, minimal version).

Maps alias → actual directory name under the dataset root, and
provides the standard subdirectory paths used by each pipeline stage.

  root(name)        : dataset root          <ROOT>/<dirname>
  pcap_dir(name)    : original pcap         root/01_pcap
  session_dir(name) : per-session pcap      root/02_session
  stat_dir(name)    : session stats CSV     root/03_session_stat
  labeled_dir(name) : noise labeling CSV    root/04_session_noisy_labeled
  json_dir(name)    : (old pipeline) JSON   root/04_json
  names()           : list of registered aliases

The dataset root is the env var NM_DATASET_ROOT (default: <repository>/00_assets/datasets, see 99_documents/DATASETS.md).
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
from pathlib import Path

DATA_ROOT = RP.DATASETS

DATASET_MAP = {
    "ustc16":  "01_USTC-TFC_2016",
    "cic17":   "02_CIC-IDS-2017",
    "cic18":   "03_CIC-IDS-2018",
    "iot23":   "04_CIC_IoT_Dataset_2023",
    
    "vpn16":   "21_ISCX-VPN-2016",
    "tor16":   "22_ISCX-TOR-2016",
    "tls1.3":  "23_CSTNET_TLS1.3",
    "cispec":  "24_CipherSpectrum",
    
    
}
ALL_DATASETS = list(DATASET_MAP.keys())

PCAP_SUBDIR    = "01_pcap"
SESSION_SUBDIR = "02_session"
STAT_SUBDIR    = "03_session_stat"
LABELED_SUBDIR = "04_session_noisy_labeled"
JSON_SUBDIR    = "04_json"


def names():
    return list(DATASET_MAP.keys())


def root(name: str) -> Path:
    if name not in DATASET_MAP:
        raise KeyError(f"unknown dataset alias: {name!r} (choose from {ALL_DATASETS})")
    return DATA_ROOT / DATASET_MAP[name]


def pcap_dir(name: str) -> Path:
    return root(name) / PCAP_SUBDIR


def session_dir(name: str) -> Path:
    return root(name) / SESSION_SUBDIR


def stat_dir(name: str) -> Path:
    return root(name) / STAT_SUBDIR


def labeled_dir(name: str) -> Path:
    return root(name) / LABELED_SUBDIR


def json_dir(name: str) -> Path:
    return root(name) / JSON_SUBDIR
