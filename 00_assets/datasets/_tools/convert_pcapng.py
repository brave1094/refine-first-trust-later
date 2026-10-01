#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_assets/datasets/_tools/convert_pcapng.py (02_preprocess/00_rename.py already converts the official pcapng files)
─────────────────────────────────────────────────────────────────────────────
Checks every file under a dataset's 01_pcap/ and converts only the pcapng ones to classic pcap
(dpkt cannot read pcapng, so feature extraction would fail -> unify to pcap beforehand).

  - pcapng detection: by the magic number (0a 0d 0d 0a); the extension (.pcap) is not trusted.
  - conversion: editcap writes a pcap to a temporary file -> verified -> replaces the original (same file name).
  - classic pcap files are left untouched.

Usage:
  python3 convert_pcapng.py --dataset vpn16
  python3 convert_pcapng.py --dataset vpn16 --workers 8
  python3 convert_pcapng.py --dataset cic18 --dry-run     # count only, no conversion
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from multiprocessing import Pool

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

sys.path.insert(0, str(RP.PREPROCESS))
from lib import datasets as ds

PCAP_MAGICS = {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",   # classic pcap (LE/BE)
               b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}   # nanosecond pcap
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"


def file_kind(path: str) -> str:
    """File format by magic number: 'pcap' | 'pcapng' | 'other'."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except Exception:
        return "other"
    if head == PCAPNG_MAGIC:
        return "pcapng"
    if head in PCAP_MAGICS:
        return "pcap"
    return "other"


def _kind_with_path(path: str):
    """(path, kind) — wrapper that also returns the file name in the parallel dry run."""
    return (path, file_kind(path))


def _convert_one(path: str):
    """Convert one pcapng to classic pcap and replace the original.
    return: (path, status)  status ∈ {converted, skip_notpcapng, fail_*}"""
    kind = file_kind(path)
    if kind != "pcapng":
        return (path, "skip_notpcapng")

    src = Path(path)
    # temporary file on the same file system (atomic replacement)
    tmp_dir = src.parent
    fd, tmp_path = tempfile.mkstemp(suffix=".pcap.tmp", dir=str(tmp_dir))
    os.close(fd)
    try:
        # editcap: pcapng → classic pcap (-F pcap)
        r = subprocess.run(
            ["editcap", "-F", "pcap", str(src), tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=3600)
        if r.returncode != 0:
            os.unlink(tmp_path)
            return (path, f"fail_editcap:{r.stderr.decode()[:60].strip()}")
        # verify that the result really is a classic pcap
        if file_kind(tmp_path) != "pcap":
            os.unlink(tmp_path)
            return (path, "fail_not_pcap_after")
        if os.path.getsize(tmp_path) == 0:
            os.unlink(tmp_path)
            return (path, "fail_empty")
        # replace the original (os.replace is atomic on the same file system)
        os.replace(tmp_path, str(src))
        return (path, "converted")
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return (path, "fail_timeout")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return (path, f"fail_exc:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser(
        description="convert pcapng files in a dataset's 01_pcap to classic pcap (file names kept)")
    ap.add_argument("--dataset", required=True, help="dataset alias (vpn16, cic18, ...)")
    ap.add_argument("--workers", type=int, default=8, help="parallel processes (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only count the pcapng files, convert nothing")
    args = ap.parse_args()

    pcap_dir = ds.pcap_dir(args.dataset)
    if not pcap_dir.exists():
        print(f"[ERR] no 01_pcap folder: {pcap_dir}")
        sys.exit(1)

    # collect every .pcap recursively
    files = [str(p) for p in pcap_dir.rglob("*.pcap")]
    print(f"[{args.dataset}] {pcap_dir}")
    print(f"  .pcap files: {len(files):,}")

    if not files:
        print("  no files. done.")
        return

    # ── dry run: count formats only (magic numbers, in parallel) ──
    if args.dry_run:
        from collections import Counter
        cnt = Counter()
        pcapng_files, other_files = [], []
        with Pool(max(1, args.workers)) as pool:
            for path, kind in tqdm(
                    pool.imap_unordered(_kind_with_path, files, chunksize=64),
                    total=len(files), desc="check", unit="file"):
                cnt[kind] += 1
                if kind == "pcapng":
                    pcapng_files.append(path)
                elif kind == "other":
                    other_files.append(path)
        print(f"  formats: {dict(cnt)}")
        print(f"  -> to convert (pcapng): {cnt.get('pcapng', 0):,}")
        if pcapng_files:
            print("  [pcapng files]")
            for p in pcapng_files:
                print(f"    {p}")
        if other_files:
            print(f"  [other files — neither pcap nor pcapng, check them]")
            for p in other_files:
                # other files: size + first 8 bytes
                try:
                    sz = os.path.getsize(p)
                    with open(p, "rb") as f:
                        head = f.read(8).hex()
                except Exception:
                    sz, head = -1, "?"
                print(f"    {p}  (size={sz}B, first 8 bytes={head})")
        return

    # ── conversion (parallel) ──
    from collections import Counter
    stat = Counter()
    fails = []
    converted = []
    n_workers = max(1, args.workers)
    with Pool(n_workers) as pool:
        for path, status in tqdm(
                pool.imap_unordered(_convert_one, files, chunksize=8),
                total=len(files), desc=f"[{args.dataset}] convert", unit="file"):
            key = status.split(":")[0]
            stat[key] += 1
            if key.startswith("fail"):
                fails.append((path, status))
            elif key == "converted":
                converted.append(path)

    print("\n" + "=" * 60)
    print(f"[{args.dataset}] conversion done")
    print(f"  converted (pcapng->pcap): {stat.get('converted', 0):,}")
    print(f"  skipped (already pcap) : {stat.get('skip_notpcapng', 0):,}")
    fail_total = sum(v for k, v in stat.items() if k.startswith("fail"))
    print(f"  failed                 : {fail_total:,}")
    if converted:
        print("  [converted files]")
        for p in converted:
            print(f"    {p}")
    if fails:
        print("  [failed files]")
        for path, status in fails[:50]:
            print(f"    {status}  {path}")
        if fail_total > 50:
            print(f"    ... and {fail_total - 50} more")


if __name__ == "__main__":
    main()