#!/usr/bin/env python3
"""
Fix the single mislabeled CipherSpectrum class:
    none_aes256gcm_chacha20  ->  none_aes256gcm_getpocket.com
(verified via SNI: actual domain = getpocket.com)

Renames the directory and every file inside (prefix swap, index preserved).
Run with the 01_pcap folder of cispec as argument, or set ROOT below.
"""

import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
from pathlib import Path
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else RP.DATASETS / "24_CipherSpectrum" / "01_pcap"
OLD = "none_aes256gcm_chacha20"
NEW = "none_aes256gcm_getpocket.com"
DRY_RUN = False   # True -> print only, rename nothing


def main():
    old_dir = ROOT / OLD
    new_dir = ROOT / NEW

    if not old_dir.is_dir():
        print(f"[ERR] not found: {old_dir}")
        sys.exit(1)
    if new_dir.exists():
        print(f"[ABORT] target already exists (merge needed): {new_dir}")
        sys.exit(1)

    files = sorted(p for p in old_dir.iterdir() if p.is_file())
    print(f"{OLD}  ->  {NEW}")
    print(f"files: {len(files)}   mode: {'DRY-RUN' if DRY_RUN else 'RENAME'}")

    if DRY_RUN:
        for p in files[:3]:
            print(f"  {p.name}  ->  {NEW + p.name[len(OLD):]}")
        print("  ...")
        print("** DRY-RUN: nothing changed. Set DRY_RUN=False to apply. **")
        return

    # 1) rename the directory
    old_dir.rename(new_dir)

    # 2) rename each file: swap OLD prefix -> NEW prefix, keep "_<n>.ext"
    renamed = 0
    for p in sorted(new_dir.iterdir()):
        if p.is_file() and p.name.startswith(OLD + "_"):
            p.rename(new_dir / (NEW + p.name[len(OLD):]))
            renamed += 1

    count = len(list(new_dir.glob("*.pcap")))
    print(f"renamed files: {renamed}")
    print(f"count in {NEW}: {count}   (expected 1000)")
    for f in sorted(new_dir.glob("*.pcap"))[:3]:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
