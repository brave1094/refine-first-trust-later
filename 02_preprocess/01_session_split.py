#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_session_split.py

For each dataset directory,  01_pcap/<class>/*.pcap  are
split per session and saved to  02_session/<class>/ .

Principle (same as the original method):
    01_pcap/<class>/original.pcap
        └─(SplitCap)─▶  02_tmp/<class>/original.pcap.TCP_....pcap  (per-session split)
                            └─(tshark)─▶  02_session/<class>/..._stream_N.pcap
    02_tmp is deleted after completion

Class directories (a,b,c,d,e ...) are kept as is in both input and output.

Usage:
    python 01_session_split.py --dataset all
    python 01_session_split.py --dataset iot23 cic18
    python 01_session_split.py --dataset ustc16 --workers 8
"""

import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import sys
import argparse
import subprocess
import resource
import shutil
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

ROOT = RP.DATASETS

# SplitCap (Netresec, run with mono) is not redistributed: bash 00_assets/tools/setup_splitcap.sh fetches it.
SCRIPT_DIR = Path(__file__).resolve().parent
SPLITCAP = os.environ.get("SPLITCAP", str(RP.TOOLS / "SplitCap" / "SplitCap.exe"))

NUM_WORKERS = 16

PCAP_SUBDIR = "01_pcap"
TMP_SUBDIR = "02_tmp"
SESSION_SUBDIR = "02_session"

# --dataset key → actual dataset directory name
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

# ============================================================
# ULIMIT (generous, since SplitCap opens a file per session)
# ============================================================

def set_ulimit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(655350, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

# ============================================================
# pcapng → pcap normalization (original untouched; converted copy is created in _norm)
# SplitCap only reads classic libpcap safely, so convert only when needed
# ============================================================

def check_magic(path: Path) -> str:
    with open(path, "rb") as f:
        return f.read(4).hex()

CLASSIC_PCAP_MAGIC = ("d4c3b2a1", "a1b2c3d4", "4d3cb2a1", "a1b23c4d")

def to_splitcap_input(src: Path, norm_dir: Path):
    """Return (classic pcap path, conversion error|None). Convert into norm_dir if pcapng/unknown."""
    try:
        magic = check_magic(src)
    except Exception as e:
        return src, f"[convert] failed to read magic: {e}"
    if magic in CLASSIC_PCAP_MAGIC:
        return src, None
    norm_dir.mkdir(parents=True, exist_ok=True)
    out = norm_dir / (src.stem + ".pcap")
    if not out.exists():
        try:
            subprocess.run(
                ["editcap", "-F", "pcap", str(src), str(out)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except Exception as e:
            return src, f"[convert] editcap failed (trying the original): {e}"
    return out, None

# ============================================================
# STAGE A. SplitCap : 01_pcap/<class>/*  →  02_tmp/<class>/
# ============================================================

def _scdone_marker(tmp_class_dir: Path, src: Path) -> Path:
    """Path of the SplitCap completion marker per original pcap (hidden file)."""
    return Path(tmp_class_dir) / f".{src.name}.scdone"

def splitcap_one(item):
    src, tmp_class_dir, norm_dir = item
    src = Path(src)
    tmp_class_dir = Path(tmp_class_dir)
    tmp_class_dir.mkdir(parents=True, exist_ok=True)

    marker = _scdone_marker(tmp_class_dir, src)
    if marker.exists():
        return (str(src), None)        # original already split → skipped on resume

    inp, conv_err = to_splitcap_input(src, Path(norm_dir))
    cmd = ["mono", SPLITCAP, "-r", str(inp), "-o", str(tmp_class_dir)]
    try:
        r = subprocess.run(cmd, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        return (str(src), f"[splitcap] execution failed: {e}")
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return (str(src), f"[splitcap] rc={r.returncode}: {tail[-1] if tail else ''}")

    # success → write completion marker (this original is skipped on the next run)
    try:
        marker.touch()
    except Exception:
        pass
    return (str(src), conv_err)   # if there is a conversion warning, record only that (processing continued)

# ============================================================
# STAGE B. tshark : 02_tmp/<class>/*  →  02_session/<class>/
#   determine proto from the SplitCap output filename, then save per stream
# ============================================================

def detect_proto(name: str):
    if ".TCP_" in name:
        return "tcp"
    if ".UDP_" in name:
        return "udp"
    return None

def tshark_split_one(item):
    src, sess_class_dir = item
    src = Path(src)
    sess_class_dir = Path(sess_class_dir)
    sess_class_dir.mkdir(parents=True, exist_ok=True)

    proto = detect_proto(src.name)
    if proto is None:
        return (str(src), None)  # not a TCP/UDP session file (ARP etc.) — normal skip

    base = src.stem

    # extract stream id
    try:
        r = subprocess.run(
            ["tshark", "-r", str(src), "-T", "fields", "-e", f"{proto}.stream"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except Exception as e:
        return (str(src), f"[tshark] stream extraction execution failed: {e}")
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return (str(src), f"[tshark] stream extraction rc={r.returncode}: {tail[-1] if tail else ''}")

    stream_ids = sorted({int(x) for x in r.stdout.split() if x.isdigit()})
    if not stream_ids:
        return (str(src), f"[tshark] no stream id (proto={proto})")

    fails = []
    for s in stream_ids:
        out = sess_class_dir / f"{base}_stream_{s}.pcap"
        if out.exists():   # skip for re-run after Ctrl+C
            continue
        tmp = out.with_suffix(out.suffix + ".tmp")
        try:
            subprocess.run(
                ["tshark", "-r", str(src), "-Y", f"{proto}.stream=={s}",
                 "-F", "pcap", "-w", str(tmp)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        except Exception as e:
            fails.append(f"stream={s}({e})")
            continue
        if tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(out)
        else:
            fails.append(f"stream={s}(0 bytes/creation failed)")
            try:
                tmp.unlink()
            except Exception:
                pass

    if fails:
        return (str(src), f"[tshark] session save failed: {', '.join(fails)}")
    return (str(src), None)

# ============================================================
# process one dataset
# ============================================================

def list_classes(pcap_dir: Path):
    return sorted([d for d in pcap_dir.iterdir() if d.is_dir()])

def find_pcaps(class_dir: Path):
    return sorted(list(class_dir.rglob("*.pcap")) + list(class_dir.rglob("*.pcapng")))

def process_dataset(key: str, workers: int):
    dname = DATASET_MAP[key]
    base = ROOT / dname
    pcap_dir = base / PCAP_SUBDIR
    tmp_dir = base / TMP_SUBDIR
    sess_dir = base / SESSION_SUBDIR

    print(f"\n{'='*70}\n[DATASET] {key}  →  {dname}\n{'='*70}")

    if not pcap_dir.is_dir():
        print(f"  [SKIP] {pcap_dir} not found")
        return

    classes = list_classes(pcap_dir)
    if not classes:
        print(f"  [WARN] no class directories in {pcap_dir}. (a,b,c... layout required)")
        return
    print(f"  {len(classes)} classes: {', '.join(c.name for c in classes)}")

    norm_root = tmp_dir / "_norm"

    # ---- STAGE A : SplitCap ----
    a_items = []
    for c in classes:
        for f in find_pcaps(c):
            a_items.append((str(f), tmp_dir / c.name, norm_root / c.name))

    if not a_items:
        print("  [WARN] no pcap to process.")
        return

    print(f"  [STAGE A] SplitCap session split  ({len(a_items)} pcap)")
    errors = []   # [(path, reason), ...]
    with Pool(workers, initializer=set_ulimit) as pool:
        for res in tqdm(pool.imap_unordered(splitcap_one, a_items),
                        total=len(a_items), desc=f"  {key} splitcap"):
            if res and res[1]:
                errors.append(res)

    # ---- STAGE B : tshark stream split ----
    b_items = []
    for c in classes:
        tc = tmp_dir / c.name
        if tc.is_dir():
            for f in sorted(tc.glob("*.pcap")):
                b_items.append((str(f), sess_dir / c.name))

    print(f"  [STAGE B] tshark session(stream) save  ({len(b_items)} session)")
    with Pool(workers, initializer=set_ulimit) as pool:
        for res in tqdm(pool.imap_unordered(tshark_split_one, b_items),
                        total=len(b_items), desc=f"  {key} tshark"):
            if res and res[1]:
                errors.append(res)

    # ---- failure log (inside the dataset directory) ----
    if errors:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = base / f"error_step1_{key}_{ts}.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(f"# 01_session_split failure log — dataset={key} ({dname}), {ts}\n")
                lf.write(f"# total {len(errors)}  (format: reason<TAB>path)\n")
                for path, reason in errors:
                    lf.write(f"{reason}\t{path}\n")
            print(f"  [WARN] {len(errors)} failures → {log_path}")
        except Exception as e:
            print(f"  [WARN] {len(errors)} failures — failed to write log: {e}")

    # ---- STAGE C : 02_tmp cleanup ----
    #   delete only on full success (0 failures). If there are failures, keep 02_tmp (+completion markers) so that
    #   on re-run, finished originals are skipped via markers and only the failures are resumed.
    #   (if the process dies midway, this point is never reached, so 02_tmp is kept → resumable)
    if errors:
        print(f"  [KEEP] 02_tmp kept — on re-run, finished originals are skipped and only failures resume: {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  no failures — 02_tmp cleanup done")

    # result summary
    total = sum(len(list((sess_dir / c.name).glob("*.pcap"))) for c in classes if (sess_dir / c.name).is_dir())
    print(f"  [DONE] {key}: 02_session session files {total} / failures {len(errors)}")

# ============================================================
# MAIN
# ============================================================

def parse_keys(values):
    if "all" in values:
        return list(DATASET_MAP.keys())
    unknown = [v for v in values if v not in DATASET_MAP]
    if unknown:
        sys.exit(f"[ERROR] unknown dataset key: {', '.join(unknown)}\n"
                 f"        available: {', '.join(DATASET_MAP.keys())}, all")
    # deduplicate + keep input order
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def main():
    ap = argparse.ArgumentParser(description="per-dataset session-level pcap split (SplitCap + tshark)")
    ap.add_argument("--dataset", nargs="+", required=True,
                    metavar="KEY",
                    help="ustc16 cic17 cic18 iot23 vpn16 tor16 tls1.3 cispec / all")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS,
                    help=f"number of parallel workers (default {NUM_WORKERS})")
    args = ap.parse_args()
    if not os.path.exists(SPLITCAP) or shutil.which("mono") is None:
        sys.exit(f"[ERROR] SplitCap ({SPLITCAP}) and mono are needed: run  bash 00_assets/tools/setup_splitcap.sh  "
                 "(downloads SplitCap from https://www.netresec.com/?page=SplitCap) and install mono "
                 "(https://www.mono-project.com/download/stable/), or set SPLITCAP to SplitCap.exe")

    set_ulimit()

    keys = parse_keys(args.dataset)
    print(f"[START] target datasets: {', '.join(keys)}  (workers={args.workers})")

    for key in keys:
        process_dataset(key, args.workers)

    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()