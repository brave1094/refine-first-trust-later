#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/salvage.py
─────────────────────────────────────────────────────────────────────────────
Corrupted pcap salvage utility (for 03_session_stat only).

For pcaps whose first tshark parse failed (cut short / oversized / malformed etc.),
try to salvage by rewriting (reindexing) with editcap; if that fails, retry with tcpdump
rewriting only the readable portion.

  salvage_pcap(pcap: Path) -> (salv_path | None, note)
    - success: (salvaged temp file path (str), note on the method used)
    - failure: (None, failure note)

The salvaged copy is a temp file. The caller (03) always unlinks it after reparsing, so
this only creates the file and returns the path. The original is never touched.

editcap / tcpdump are looked up in PATH (same as how 03 invokes tshark).
If missing, that method is skipped.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Cap (s) that prevents the salvage tools from running forever on pathological files.
SALVAGE_TIMEOUT = int(os.environ.get("NM_SALVAGE_TIMEOUT", "120"))

# Minimum valid pcap size (more than the 24B global header = at least 1 packet).
_MIN_VALID_BYTES = 25

_HAS_EDITCAP = shutil.which("editcap") is not None
_HAS_TCPDUMP = shutil.which("tcpdump") is not None


def _mk_tmp(src: Path) -> str:
    """Create a temp pcap path for the salvaged copy (empty file). Created in the same directory as the original
    to reduce rename/move cost on tshark reparsing (the caller unlinks on failure)."""
    fd, path = tempfile.mkstemp(prefix=".salv_", suffix=".pcap",
                                dir=str(src.parent))
    os.close(fd)
    return path


def _ok(path: str) -> bool:
    try:
        return os.path.getsize(path) >= _MIN_VALID_BYTES
    except OSError:
        return False


def _run(cmd) -> bool:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=SALVAGE_TIMEOUT)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def salvage_pcap(pcap: Path):
    """Try to salvage a corrupted pcap. Returns (salvaged_path|None, note)."""
    pcap = Path(pcap)
    out = _mk_tmp(pcap)

    # 1) editcap: rewrites header/index. Robust to oversized snaplen·minor malformation.
    if _HAS_EDITCAP:
        # editcap may write the leading part even with rc!=0 on a truncated last packet, so
        # output validity (_ok) is checked as well.
        _run(["editcap", "-F", "pcap", str(pcap), out])
        if _ok(out):
            return out, "editcap"

    # 2) tcpdump: rewrites up to the readable point. Robust to cut short (truncated file end).
    if _HAS_TCPDUMP:
        _run(["tcpdump", "-r", str(pcap), "-w", out])
        if _ok(out):
            return out, "tcpdump"

    # all failed → clean up the temp file, then None.
    try:
        os.unlink(out)
    except OSError:
        pass
    return None, "unsalvageable"
