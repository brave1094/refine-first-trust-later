#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/parser/pcap_source.py
─────────────────────────────────────────────────────────────────────────────
filelist-based session source + multiprocessing runner.

  extract_mode:
    session : one individual session file in 02_session = one session (stream='-')
    whole   : extract {proto}.stream=={stream} sessions from monolithic pcaps in 01_pcap.
              Each file is read only once: one tshark stream map + one dpkt pass
              (reuses the proven pattern from 02_xgboost/extractor.py).

  run_split(list_csv, index_roots, extract_mode, workers, sample_fn_name,
            shaper_key, opt, on_result, desc)
    - workers call make_sample(packets, meta, opt) of the shaping module
    - on_result(meta, sample) callback in the main process (writer etc.)
    - meta: filelist row dict (filename/session_id/proto/stream/group_key/task*)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ═══════════════ pcap reading ═══════════════
def read_pcap(path):
    """pcap → [(ts, buf, linktype)] (dpkt). [] on failure / unsupported pcapng"""
    import dpkt
    out = []
    with open(path, "rb") as f:
        try:
            rdr = dpkt.pcap.Reader(f)
        except Exception:
            try:
                f.seek(0)
                rdr = dpkt.pcapng.Reader(f)
            except Exception:
                return []
        try:
            linktype = rdr.datalink()
        except Exception:
            linktype = 1
        try:
            for ts, buf in rdr:
                out.append((ts, buf, linktype))
        except Exception:
            pass
    return out


_INDEX_CACHE = {}


def build_pcap_index(*roots) -> dict:
    """pcap file name → path index (earlier roots take precedence). Cached within the process
    (so that the 4 splits of a dataset do not repeat the same rglob)."""
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


def _stream_map(pcap_path: str):
    """one tshark pass → {frame_number: (proto, stream_id)}"""
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
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            frame_no = int(parts[0])
        except ValueError:
            continue
        if parts[1] != "":
            fmap[frame_no] = ("tcp", parts[1])
        elif parts[2] != "":
            fmap[frame_no] = ("udp", parts[2])
    return fmap


# ═══════════════ worker globals (set by initializer) ═══════════════
_W = {}


def _init_worker(shaper_key: str, opt: dict):
    """Worker init: load the shaping module + store options."""
    import importlib
    import sys
    from pathlib import Path as _P
    lib_dir = _P(__file__).resolve().parent.parent          # …/02_preprocess/lib/
    pre_dir = lib_dir.parent                                 # …/02_preprocess/
    for p in (str(pre_dir), str(lib_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module(f"lib.shaping.shaping_{shaper_key}")
    _W["mod"] = mod
    _W["opt"] = opt


def _work_session(job):
    """job = (meta_dict, pcap_path) — one session in session mode."""
    meta, pcap_path = job
    try:
        packets = read_pcap(pcap_path)
        if not packets:
            return meta, None
        sample = _W["mod"].make_sample(packets, meta, _W["opt"])
        return meta, sample
    except Exception:
        return meta, None


def _load_bucket_cache(cache_dir, pcap_path):
    """Load cached buckets ((None, cache_file) if missing or failed)."""
    import os as _os
    try:
        _os.makedirs(cache_dir, exist_ok=True)
        cf = _os.path.join(cache_dir, _os.path.basename(pcap_path) + ".pkl")
        if _os.path.exists(cf):
            import pickle
            with open(cf, "rb") as fh:
                return pickle.load(fh), cf
        return None, cf
    except Exception:
        return None, None


def _save_bucket_cache(cache_file, buckets):
    """Atomically cache buckets (harmless if it fails)."""
    try:
        import os as _os, pickle
        tmp = cache_file + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(buckets, fh, protocol=4)
        _os.replace(tmp, cache_file)
    except Exception:
        pass


def _work_whole_file(job):
    """job = (pcap_path, [meta_dict, ...]) — all sessions inside one monolithic pcap.
    ★ extraction cache: buckets (raw packets per session) are model-independent → only the first model runs the tshark+dpkt scan,
      later models reuse the pkl (avoids a 400GB rescan). Only when opt['_extract_cache'] is set."""
    pcap_path, metas = job
    cache_dir = _W.get("opt", {}).get("_extract_cache") if isinstance(_W.get("opt"), dict) else None

    buckets = cache_file = None
    if cache_dir:
        buckets, cache_file = _load_bucket_cache(cache_dir, pcap_path)

    if buckets is None:                                      # cache miss → actual scan
        fmap = _stream_map(pcap_path)
        if not fmap:
            return [(m, None) for m in metas]
        wanted = {(m["proto"], m["stream"]) for m in metas}
        buckets = {k: [] for k in wanted}
        try:
            import dpkt
            with open(pcap_path, "rb") as f:
                rdr = dpkt.pcap.Reader(f)
                try:
                    lt = rdr.datalink()
                except Exception:
                    lt = 1
                for i, (ts, buf) in enumerate(rdr, start=1):
                    key = fmap.get(i)
                    if key is None or key not in buckets:
                        continue
                    buckets[key].append((ts, buf, lt))
        except Exception:
            return [(m, None) for m in metas]
        if cache_file:
            _save_bucket_cache(cache_file, buckets)

    out = []
    for m in metas:
        pkts = buckets.get((m["proto"], m["stream"]))
        if not pkts:
            out.append((m, None))
            continue
        try:
            out.append((m, _W["mod"].make_sample(pkts, m, _W["opt"])))
        except Exception:
            out.append((m, None))
    return out


# ═══════════════ multi-model (parse once → multiple shapings) ═══════════════
def _init_worker_multi(shaper_keys, opt):
    import importlib, sys
    from pathlib import Path as _P
    lib_dir = _P(__file__).resolve().parent.parent; pre_dir = lib_dir.parent
    for p in (str(pre_dir), str(lib_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _W["mods"] = {k: importlib.import_module(f"lib.shaping.shaping_{k}")
                  for k in shaper_keys}
    _W["opt"] = opt


def _samples_from_pkts(pkts, meta):
    """raw packets → parse_session once → build_from_views per model (dict)."""
    from . import dpkt_parser as dp
    views = dp.parse_session(pkts, _W["opt"].get("parser", "dpkt"), max_packets=20)
    out = {}
    for k, mod in _W["mods"].items():
        try:
            out[k] = mod.build_from_views(views, meta, _W["opt"])
        except Exception:
            out[k] = None
    return out


def _work_session_multi(job):
    meta, pcap_path = job
    try:
        packets = read_pcap(pcap_path)
        if not packets:
            return meta, None
        return meta, _samples_from_pkts(packets, meta)
    except Exception:
        return meta, None


def _work_whole_file_multi(job):
    """Same as _work_whole_file (cache reuse), but shapes for all models with one parse per session."""
    pcap_path, metas = job
    cache_dir = _W.get("opt", {}).get("_extract_cache") if isinstance(_W.get("opt"), dict) else None
    buckets = cache_file = None
    if cache_dir:
        buckets, cache_file = _load_bucket_cache(cache_dir, pcap_path)
    if buckets is None:
        fmap = _stream_map(pcap_path)
        if not fmap:
            return [(m, None) for m in metas]
        wanted = {(m["proto"], m["stream"]) for m in metas}
        buckets = {k: [] for k in wanted}
        try:
            import dpkt
            with open(pcap_path, "rb") as f:
                rdr = dpkt.pcap.Reader(f)
                try:
                    lt = rdr.datalink()
                except Exception:
                    lt = 1
                for i, (ts, buf) in enumerate(rdr, start=1):
                    key = fmap.get(i)
                    if key is None or key not in buckets:
                        continue
                    buckets[key].append((ts, buf, lt))
        except Exception:
            return [(m, None) for m in metas]
        if cache_file:
            _save_bucket_cache(cache_file, buckets)
    out = []
    for m in metas:
        pkts = buckets.get((m["proto"], m["stream"]))
        if not pkts:
            out.append((m, None))
            continue
        out.append((m, _samples_from_pkts(pkts, m)))
    return out


def run_split_multi(list_csv: Path, index_roots, extract_mode: str, workers: int,
                    shaper_keys, opt: dict, on_result, desc: str = ""):
    """Multi-model version of run_split. Calls on_result(meta, {model: sample}).
    parse_session runs only once per session → shaping for all shaper_keys (key speedup)."""
    lst = pd.read_csv(list_csv, dtype=str, na_filter=False, encoding="utf-8-sig")
    if not isinstance(index_roots, (list, tuple)):
        index_roots = [index_roots]
    pcap_index = build_pcap_index(*index_roots)
    n_ok = n_missing = n_failed = 0
    from concurrent.futures import ProcessPoolExecutor

    if extract_mode == "whole":
        from collections import defaultdict
        groups = defaultdict(list)
        for _, r in lst.iterrows():
            p = pcap_index.get(r["filename"])
            if p is None:
                n_missing += 1; continue
            groups[str(p)].append(dict(r))
        jobs = list(groups.items())
        if workers <= 1:
            _init_worker_multi(shaper_keys, opt)
            it = (_work_whole_file_multi(j) for j in jobs)
        else:
            ex = ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker_multi,
                                     initargs=(shaper_keys, opt))
            it = ex.map(_work_whole_file_multi, jobs)
        for results in tqdm(it, total=len(jobs), desc=desc, unit="file"):
            for meta, samples in results:
                if samples is None:
                    n_failed += 1
                else:
                    on_result(meta, samples); n_ok += 1
        if workers > 1:
            ex.shutdown()
        return n_ok, n_missing, n_failed

    # session mode
    jobs = []
    for _, r in lst.iterrows():
        p = pcap_index.get(r["filename"])
        if p is None:
            n_missing += 1; continue
        jobs.append((dict(r), str(p)))
    if workers <= 1:
        _init_worker_multi(shaper_keys, opt)
        it = (_work_session_multi(j) for j in jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker_multi,
                                 initargs=(shaper_keys, opt))
        it = ex.map(_work_session_multi, jobs, chunksize=8)
    for meta, samples in tqdm(it, total=len(jobs), desc=desc, unit="sess"):
        if samples is None:
            n_failed += 1
        else:
            on_result(meta, samples); n_ok += 1
    if workers > 1:
        ex.shutdown()
    return n_ok, n_missing, n_failed


# ═══════════════ split runner ═══════════════
def run_split(list_csv: Path, index_roots, extract_mode: str, workers: int,
              shaper_key: str, opt: dict, on_result, desc: str = ""):
    """
    Runs the shaping module make_sample for every session in the filelist and
    calls on_result(meta, sample) in the main process.

    returns (n_ok, n_missing, n_failed)
    """
    lst = pd.read_csv(list_csv, dtype=str, na_filter=False, encoding="utf-8-sig")
    if not isinstance(index_roots, (list, tuple)):
        index_roots = [index_roots]
    pcap_index = build_pcap_index(*index_roots)

    n_ok = n_missing = n_failed = 0

    if extract_mode == "whole":
        from collections import defaultdict
        groups = defaultdict(list)
        for _, r in lst.iterrows():
            p = pcap_index.get(r["filename"])
            if p is None:
                n_missing += 1
                continue
            groups[str(p)].append(dict(r))
        jobs = list(groups.items())

        if workers <= 1:
            _init_worker(shaper_key, opt)
            for job in tqdm(jobs, desc=desc, unit="file"):
                for meta, sample in _work_whole_file(job):
                    if sample is None:
                        n_failed += 1
                    else:
                        on_result(meta, sample)
                        n_ok += 1
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker,
                                     initargs=(shaper_key, opt)) as ex:
                for results in tqdm(ex.map(_work_whole_file, jobs),
                                    total=len(jobs), desc=desc, unit="file"):
                    for meta, sample in results:
                        if sample is None:
                            n_failed += 1
                        else:
                            on_result(meta, sample)
                            n_ok += 1
        return n_ok, n_missing, n_failed

    # session mode
    jobs = []
    for _, r in lst.iterrows():
        p = pcap_index.get(r["filename"])
        if p is None:
            n_missing += 1
            continue
        jobs.append((dict(r), str(p)))

    if workers <= 1:
        _init_worker(shaper_key, opt)
        for job in tqdm(jobs, desc=desc, unit="sess"):
            meta, sample = _work_session(job)
            if sample is None:
                n_failed += 1
            else:
                on_result(meta, sample)
                n_ok += 1
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_init_worker,
                                 initargs=(shaper_key, opt)) as ex:
            for meta, sample in tqdm(ex.map(_work_session, jobs, chunksize=8),
                                     total=len(jobs), desc=desc, unit="sess"):
                if sample is None:
                    n_failed += 1
                else:
                    on_result(meta, sample)
                    n_ok += 1
    return n_ok, n_missing, n_failed
