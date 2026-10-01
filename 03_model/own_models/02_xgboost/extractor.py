#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/own_models/02_xgboost/extractor.py
─────────────────────────────────────────────────────────────────────────────
XGBoost feature extractor based on direct pcap parsing (--feat plugin style).

  --feat a  →  uses extract_features(packets) from feat/xgboost_feature_a.py.
  One session pcap = one flow; one feature row is produced per session.

The output CSV has a 'common 8 columns' + 'feature columns' layout:
  Label | filename | Stream_num | protocol | srcip | srcport | dstip | dstport | <features...>
    - The common 8 columns are filled from the filelist (group_key/proto/stream) + parsed meta.
    - Feature columns do not include ip/port/protocol (already in the common 8 columns).

Session pcap location:
  session-mode session (stream="-") : {session_dir}/<class>/<filename>  (1 file = 1 session)
  whole-mode session (stream!="-")  : from {session_dir}/<class>/<filename>
                                  temporarily extract '{proto}.stream=={stream}' then parse (requires tshark)

Called by 06_make_dataset.py via build_all(...), or runnable as a standalone CLI.

Standalone run:
  python3 extractor.py --feat a \
      --list 01_dataset/vpn16/00_filelist/list_train_noisy.csv \
      --session-dir datasets/21_ISCX-VPN-2016/02_session \
      --out train_features.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
FEAT_DIR = _HERE / "feat"

COMMON_COLS = ["Label", "filename", "Stream_num", "protocol",
               "srcip", "srcport", "dstip", "dstport"]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ═══════════════ feat plugin loading ═══════════════
def load_feat_module(feat: str):
    path = FEAT_DIR / f"xgboost_feature_{feat}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"feat definition not found: {path}\n"
            f"       (xgboost_feature_{feat}.py must exist in feat/)")
    name = f"_xgb_feat_{feat}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════ pcap index / stream extraction ═══════════════
_INDEX_CACHE = {}


def build_pcap_index(*roots) -> dict:
    """
    pcap filename → path index. Multiple roots are indexed in order,
    earlier roots take priority (setdefault) — 02_session (individual sessions) first, else 01_pcap (whole).

      session-mode session (stream="-") : matched to individual session files in 02_session
      whole-mode session (stream!="-")  : 02_session has no whole-file name, so
                                      matched to the whole pcap in 01_pcap → worker extracts the stream
    Within one process each root combination is scanned only once (cached) — avoids repeating for the 4 splits.
    """
    key = tuple(str(r) for r in roots if r is not None)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    idx = {}
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for pat in ("*.pcap", "*.pcapng"):
            for p in root.rglob(pat):
                idx.setdefault(p.name, p)
    _INDEX_CACHE[key] = idx
    return idx


def _tshark_available():
    from shutil import which
    return which("tshark") is not None


def _extract_stream(pcap_path: Path, proto: str, stream: str, tmp_dir: Path):
    out = tmp_dir / f"_stream_{proto}_{stream}.pcap"
    try:
        subprocess.run(
            ["tshark", "-r", str(pcap_path), "-Y", f"{proto}.stream=={stream}",
             "-F", "pcap", "-w", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=300)
    except Exception:
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def _stream_map(pcap_path: str):
    """Run tshark once → {frame_number: (proto, stream_id)} mapping.
    Builds the frame→stream mapping in advance so the whole pcap is not reread per session.
    Extracts tcp.stream / udp.stream together with each frame."""
    try:
        p = subprocess.run(
            ["tshark", "-r", pcap_path, "-T", "fields",
             "-e", "frame.number", "-e", "tcp.stream", "-e", "udp.stream"],
            capture_output=True, text=True, timeout=1800)
    except Exception:
        return {}
    fmap = {}
    for line in p.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        fn, tcp_s, udp_s = parts[0], parts[1], parts[2]
        if not fn:
            continue
        try:
            frame_no = int(fn)
        except ValueError:
            continue
        if tcp_s != "":
            fmap[frame_no] = ("tcp", tcp_s)
        elif udp_s != "":
            fmap[frame_no] = ("udp", udp_s)
    return fmap


def _empty_feature_row(mod, label, filename, stream, proto="-"):
    """Row for sessions without a valid flow (fragmentation/GRE etc.).
    Only the minimum of the common 8 columns is filled; all features are MISSING('-') → treated as NaN in training.
    Byte models represent this session with raw bytes, so the feature models also keep the
    sample itself to match the sample count across the 8 models (the representational limit of feature models remains as NaN)."""
    MISSING = getattr(mod, "MISSING", "-")
    proto_num = {"tcp": 6, "udp": 17}.get(proto, "-")
    row = {"Label": label, "filename": filename, "Stream_num": stream,
           "protocol": proto_num, "srcip": "-", "srcport": "-",
           "dstip": "-", "dstport": "-"}
    for c in mod.feature_columns():
        row[c] = MISSING
    return row


def _work_whole_file(job):
    """
    job = (pcap_path, [(group_key, filename, proto, stream), ...])
    Reads one whole pcap 'only once' (tshark stream map + dpkt pass) and extracts
    features for all sessions in it. return: [row dict, ...]
    Sessions without a valid flow are kept as NaN-feature rows (sample-aligned with the byte models).
    """
    import dpkt
    pcap_path, sessions = job
    mod = _W["mod"]

    # 1) tshark once → frame→(proto,stream) mapping
    fmap = _stream_map(pcap_path)
    if not fmap:
        return []

    # 2) single dpkt pass → group packets by stream_id (only needed streams)
    wanted = {(p, s) for (_, _, p, s) in sessions}
    buckets = {key: [] for key in wanted}
    try:
        with open(pcap_path, "rb") as f:
            rdr = dpkt.pcap.Reader(f)
            try:
                lt = rdr.datalink()
            except Exception:
                lt = dpkt.pcap.DLT_EN10MB
            for i, (ts, buf) in enumerate(rdr, start=1):
                key = fmap.get(i)                 # (proto, stream_id)
                if key is None or key not in buckets:
                    continue
                buckets[key].append((ts, buf, lt))
    except Exception:
        return []

    # 3) per-session feature extraction
    rows = []
    for (gk, filename, proto, stream) in sessions:
        pkts = buckets.get((proto, stream))
        if not pkts:
            continue
        res = mod.extract_features(pkts)
        if res is None:
            # whole-mode sessions are fixed as tcp/udp streams by tshark → IP guaranteed.
            # if no valid flow (fragment/GRE), keep a NaN-feature row.
            rows.append(_empty_feature_row(mod, gk, filename, stream, proto))
            continue
        m = res["meta"]
        row = {"Label": gk, "filename": filename, "Stream_num": stream,
               "protocol": m["protocol"], "srcip": m["src_ip"],
               "srcport": m["src_port"], "dstip": m["dst_ip"],
               "dstport": m["dst_port"]}
        row.update(res["features"])
        rows.append(row)
    return rows


def _read_pcap(path):
    """pcap → list of (ts, buf, linktype) tuples (dpkt). Consumed by process_packet of feat.
    linktype is passed along so feat can distinguish Ethernet/RawIP/SLL when parsing.
    Formats dpkt.pcap cannot read (pcapng etc.) yield an empty list."""
    import dpkt
    out = []
    with open(path, "rb") as f:
        try:
            rdr = dpkt.pcap.Reader(f)
        except Exception:
            return []
        try:
            linktype = rdr.datalink()
        except Exception:
            linktype = dpkt.pcap.DLT_EN10MB
        for ts, buf in rdr:
            out.append((ts, buf, linktype))
    return out


# ═══════════════ worker (1 session → feature row) ═══════════════
_W = {}

def _init_worker(feat):
    import warnings
    warnings.filterwarnings("ignore", message="Precision loss occurred")
    _W["mod"] = load_feat_module(feat)


def _work(job):
    """
    job = (label, filename, proto, stream, pcap_path)
    return: dict(common 8 columns + features) or None

    NOTE: _work is the session-mode-only path (whole mode uses _work_whole_file).
    In session mode 1 file = 1 session, so the file is read directly
    regardless of the stream value. (stream is used only as metadata — previously, stream!='-'
    triggered tshark re-extraction, which failed entirely in environments without tshark)
    """
    label, filename, proto, stream, pcap_path = job
    mod = _W["mod"]

    try:
        packets = _read_pcap(str(pcap_path))
    except Exception:
        return None

    if not packets:
        return None
    res = mod.extract_features(packets)
    if res is None:
        # no valid flow: if there is at least one IP packet, keep a NaN-feature row (same as byte models),
        # if only non-IP (ARP etc.), drop (byte models also drop as not-views).
        has_ip = any(mod._unpack_ip(ts, buf, lt) is not None
                     for (ts, buf, lt) in packets)
        return (_empty_feature_row(mod, label, filename, stream, proto)
                if has_ip else None)

    m = res["meta"]
    row = {
        "Label":      label,
        "filename":   filename,
        "Stream_num": stream,
        "protocol":   m["protocol"],
        "srcip":      m["src_ip"],
        "srcport":    m["src_port"],
        "dstip":      m["dst_ip"],
        "dstport":    m["dst_port"],
    }
    row.update(res["features"])
    return row


# ═══════════════ process one list (split) → CSV ═══════════════
def extract_list(feat_mod, feat: str, list_csv: Path, index_roots, out_csv: Path,
                 workers: int = 1, extract_mode: str = "session"):
    """index_roots: list of roots to use for the pcap index (whole=[01_pcap], session=[02_session]).
    whole mode: reads each whole pcap only once and extracts all sessions in it (fast)."""
    lst = pd.read_csv(list_csv, dtype=str, na_filter=False, encoding="utf-8-sig")
    if not isinstance(index_roots, (list, tuple)):
        index_roots = [index_roots]
    pcap_index = build_pcap_index(*index_roots)

    feat_cols = feat_mod.feature_columns()
    all_cols = COMMON_COLS + feat_cols
    MISSING = getattr(feat_mod, "MISSING", "-")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_empty, missing = 0, 0, 0

    # ═══ whole mode: group by whole pcap file and read each only once ═══
    if extract_mode == "whole":
        from collections import defaultdict
        file_groups = defaultdict(list)          # pcap_path → [(gk, fn, proto, stream)]
        for _, r in lst.iterrows():
            p = pcap_index.get(r["filename"])
            if p is None:
                missing += 1
                continue
            file_groups[str(p)].append(
                (r["group_key"], r["filename"], r["proto"], r["stream"]))
        file_jobs = [(pcap_path, sess) for pcap_path, sess in file_groups.items()]
        total_sess = sum(len(s) for _, s in file_jobs)

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, restval=MISSING,
                                    extrasaction="ignore")
            writer.writeheader()
            if workers <= 1:
                _init_worker(feat)
                for job in tqdm(file_jobs, desc=out_csv.stem, unit="file"):
                    for row in _work_whole_file(job):
                        writer.writerow(row); n_ok += 1
            else:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=workers,
                                         initializer=_init_worker,
                                         initargs=(feat,)) as ex:
                    for rows in tqdm(ex.map(_work_whole_file, file_jobs),
                                     total=len(file_jobs), desc=out_csv.stem,
                                     unit="file"):
                        for row in rows:
                            writer.writerow(row); n_ok += 1
        n_empty = total_sess - n_ok
        print(f"  [extractor] {out_csv.name}: rows={n_ok} "
              f"missing_pcap={missing} empty/failed={n_empty} "
              f"features={len(feat_cols)}  (whole: {len(file_jobs)} files)")
        return n_ok, missing, n_empty

    # ═══ session mode: 1 session file = 1 job (original) ═══
    jobs = []
    for _, r in lst.iterrows():
        p = pcap_index.get(r["filename"])
        if p is None:
            missing += 1
            continue
        jobs.append((r["group_key"], r["filename"], r["proto"], r["stream"], str(p)))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, restval=MISSING,
                                extrasaction="ignore")
        writer.writeheader()

        if workers <= 1:
            _init_worker(feat)
            it = (_work(j) for j in tqdm(jobs, desc=out_csv.stem, unit="sess"))
            for row in it:
                if row is None:
                    n_empty += 1
                    continue
                writer.writerow(row)
                n_ok += 1
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(feat,)) as ex:
                for row in tqdm(ex.map(_work, jobs, chunksize=8),
                                total=len(jobs), desc=out_csv.stem, unit="sess"):
                    if row is None:
                        n_empty += 1
                        continue
                    writer.writerow(row)
                    n_ok += 1

    print(f"  [extractor] {out_csv.name}: rows={n_ok} "
          f"missing_pcap={missing} empty/failed={n_empty} "
          f"features={len(feat_cols)}")
    return n_ok, missing, n_empty


# ═══════════════ 06_make_dataset.py entry point ═══════════════
def build_all(*, dataset: str, task: str, mode: str, mode_dir: Path, out_dir: Path,
              label_map: dict, session_dir: Path, feat: str = "a",
              workers: int = 1, pcap_dir: Path = None,
              extract_mode: str = "session", **_):
    """
    Extract train/test for one mode via pcap parsing and save as CSV.
    Output: {out_dir}/{train,test}_features.csv  (+ feature_columns.json)

    extract_mode:
      "session" : use individual session files in 02_session (datasets with session_id='-')
      "whole"   : extract sessions from the whole pcap in 01_pcap via {proto}.stream=={stream}
                  (datasets with session_id=tcp_N/udp_N: ustc16/cic17/cic18/iot23)
    """
    mode_dir, out_dir = Path(mode_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_mod = load_feat_module(feat)

    # choose the pcap index source by mode
    if extract_mode == "whole":
        index_roots = [pcap_dir]              # whole pcap only
    else:
        index_roots = [session_dir]           # individual session files only

    print(f"  [extractor] feat={feat}  features={len(feat_mod.feature_columns())}  "
          f"extract_mode={extract_mode}  roots={[str(r) for r in index_roots if r]}")

    summary = {}
    for split in ("train", "test"):
        list_csv = mode_dir / f"list_{split}.csv"
        out_csv  = out_dir / f"{split}_features.csv"
        summary[split] = extract_list(feat_mod, feat, list_csv,
                                      index_roots, out_csv, workers,
                                      extract_mode=extract_mode)

    import json
    (out_dir / "feature_columns.json").write_text(
        json.dumps(COMMON_COLS + feat_mod.feature_columns(),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="XGBoost feature extraction via direct pcap parsing")
    ap.add_argument("--feat", default="a", help="feat code (feat/xgboost_feature_{feat}.py)")
    ap.add_argument("--list", required=True, help="filelist CSV (e.g. list_train.csv)")
    ap.add_argument("--session-dir", default=None, help="session pcap root (02_session)")
    ap.add_argument("--pcap-dir", default=None, help="original whole pcap root (01_pcap)")
    ap.add_argument("--extract-mode", choices=["session", "whole"], default="session",
                    help="session=individual files in 02_session / whole=extract streams from whole pcaps in 01_pcap")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.extract_mode == "whole":
        if not args.pcap_dir:
            ap.error("--extract-mode whole requires --pcap-dir")
        roots = [Path(args.pcap_dir)]
    else:
        if not args.session_dir:
            ap.error("--extract-mode session requires --session-dir")
        roots = [Path(args.session_dir)]

    feat_mod = load_feat_module(args.feat)
    print(f"[extractor] feat={args.feat} extract_mode={args.extract_mode} "
          f"features={len(feat_mod.feature_columns())}")
    extract_list(feat_mod, args.feat, Path(args.list), roots,
                 Path(args.out), args.workers, extract_mode=args.extract_mode)


if __name__ == "__main__":
    main()