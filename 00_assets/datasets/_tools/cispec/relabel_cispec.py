#!/usr/bin/env python3
"""
CipherSpectrum: extract the 3 cipher-suite zips and relabel into class dirs.

Layout produced (under OUT_DIR):
    none_<cipher>_<class>/
        none_<cipher>_<class>_1.pcap
        none_<cipher>_<class>_2.pcap
        ...

- <cipher> is derived from the zip filename:
      aes-128-gcm.zip       -> aes128gcm
      chacha20-poly1305.zip -> chacha20poly1305
      aes-256-gcm.zip       -> aes256gcm
- <class> is taken from the DOMAIN DIRECTORY NAME inside each zip
  (NOT from any token in the original filename) -> this is what fixes the
  previous 'chacha20' mislabel: a cipher name can never leak into the class.
- .DS_Store / __MACOSX junk is deleted.
- per (cipher, class) the index restarts at _1, _2, _3, ...
"""

import re
import sys
import shutil
import zipfile
from pathlib import Path

# ---------------- config ----------------
SRC_DIR = Path(".")               # folder containing the 3 zip files
OUT_DIR = Path("../01_pcap")      # extraction + rename target
TMP_DIR = Path("../_cispec_extract_tmp")

PCAP_EXTS = {".pcap", ".pcapng", ".cap"}

DRY_RUN = False        # True -> only print what WOULD happen, move nothing
RESET_OUTPUT = False   # True -> delete existing 'none_<cipher>_*' class dirs
                       #         (scoped to ciphers we are writing) before writing
# ----------------------------------------


def norm_cipher(zip_stem: str) -> str:
    """'aes-128-gcm' -> 'aes128gcm', 'chacha20-poly1305' -> 'chacha20poly1305'."""
    return re.sub(r"[^0-9a-z]", "", zip_stem.lower())


def natkey(name: str):
    """Natural sort key so _2 comes before _10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def cipher_root(extracted: Path) -> Path:
    """Return the directory that holds the class (domain) folders."""
    dirs = [p for p in extracted.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    return dirs[0] if len(dirs) == 1 else extracted   # single wrapper dir, else flat


def main():
    zips = sorted(SRC_DIR.glob("*.zip"))
    if not zips:
        print(f"[ERR] no .zip found in {SRC_DIR.resolve()}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"out dir : {OUT_DIR.resolve()}")
    print(f"zips    : {[z.name for z in zips]}")
    print(f"mode    : {'DRY-RUN' if DRY_RUN else 'WRITE'}\n")

    grand_files = 0
    grand_skipped = 0

    for zp in zips:
        cipher = norm_cipher(zp.stem)
        print(f"=== {zp.name}  ->  cipher='{cipher}' ===")

        ex = TMP_DIR / zp.stem
        if ex.exists():
            shutil.rmtree(ex)
        ex.mkdir(parents=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(ex)

        # drop macOS junk
        for junk in ex.rglob(".DS_Store"):
            junk.unlink()
        mac = ex / "__MACOSX"
        if mac.exists():
            shutil.rmtree(mac)

        root = cipher_root(ex)
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                            key=lambda p: p.name.lower())
        if not class_dirs:
            print(f"  [WARN] no class dirs under {root}")
            shutil.rmtree(ex, ignore_errors=True)
            continue

        # optional scoped reset
        if RESET_OUTPUT and not DRY_RUN:
            for old in OUT_DIR.glob(f"none_{cipher}_*"):
                if old.is_dir():
                    shutil.rmtree(old)

        zip_files = 0
        for cdir in class_dirs:
            cls = cdir.name                       # e.g. 'adblockplus.org'
            label = f"none_{cipher}_{cls}"        # class_name = task1_task2_task3
            dst = OUT_DIR / label

            pcaps = sorted(
                [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() in PCAP_EXTS],
                key=lambda p: natkey(p.name),
            )
            others = [p for p in cdir.rglob("*") if p.is_file() and p.suffix.lower() not in PCAP_EXTS]
            grand_skipped += len(others)

            if not DRY_RUN:
                dst.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(pcaps, start=1):
                    shutil.move(str(src), str(dst / f"{label}_{i}{src.suffix.lower()}"))

            zip_files += len(pcaps)
            print(f"  {cls:34s} {len(pcaps):5d} -> {label}/"
                  + (f"   [skip {len(others)} non-pcap]" if others else ""))

        grand_files += zip_files
        print(f"  -- {zp.name}: {len(class_dirs)} classes, {zip_files} pcap files\n")
        shutil.rmtree(ex, ignore_errors=True)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    print(f"DONE. pcap files relabeled: {grand_files}"
          + (f"   (skipped {grand_skipped} non-pcap)" if grand_skipped else ""))
    if DRY_RUN:
        print("** DRY-RUN: nothing was moved. Set DRY_RUN=False to apply. **")


if __name__ == "__main__":
    main()