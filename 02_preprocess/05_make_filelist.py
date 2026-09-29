#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_make_filelist.py
─────────────────────────────────────────────────────────────────────────────
04_session_noisy_labeled labeling CSV → filelist(train/test) + label_map generation.

  noisy    = all sessions (clean + noisy, everything)
  denoised = only sessions whose noise-rule one-hot columns ( Aa_*/Bb_*/Cc_*/Cd_*/De_*/Df_*/Eg_* )
             are all 0 (= refined set. Rule list: dataset default or --new_rule selection)
  --mode {noisy,denoised,both} : default both = generate both modes at once.
  (--denoised is a backward-compatible alias for --mode denoised)

  The selected sessions are split into train:test while preserving class (stratify) ratios
  (deterministic: same seed + same data → same result).
  noisy/denoised are each an independent stratified split.

label(group_key) = value of the column chosen by --task. SCIE experiments fix task3 (default).

Output : 01_dataset/{dataset}/00_filelist/
  list_train_noisy.csv  list_train_denoised.csv
  list_test_noisy.csv   list_test_denoised.csv
  label_map.json          # shared by noisy/denoised (consistent ids across exp1~4 cross experiments)

List schema (1 session = 1 row):
  filename, session_id, proto, stream, group_key, task1, task2, task3
    - proto/stream : parsed from session_id(tcp_N/udp_N/'-').
      downstream, whole-mode pcaps are filtered with '{proto}.stream=={stream}'
      to extract only that session. In session mode ('-'), 1 file = 1 session.

Usage:
  python3 05_make_filelist.py --dataset vpn16                              # task3, both modes
  python3 05_make_filelist.py --dataset all --sample_per_class 5000
  python3 05_make_filelist.py --dataset vpn16 --mode denoised --new_rule --Aa_1 --Cc_1
  python3 05_make_filelist.py --dataset tls1.3 --dry-run
"""
import argparse
import json
import re
import sys
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):            # without tqdm installed, iterate as is
        return it

CODE_DIR = Path(__file__).resolve().parent          # …/02_preprocess/
WORK_DIR = CODE_DIR.parent                          # repository root
LIST_BASE = WORK_DIR / "01_dataset"

sys.path.insert(0, str(CODE_DIR))
from lib import datasets as ds
try:
    from noise_rule import get_default_clean_rules
except Exception:
    def get_default_clean_rules(dataset):      # fallback when noise_rule is unavailable
        return []

INPUT_SUBDIR = "04_session_noisy_labeled"

# taxonomy one-hot noise columns (common Aa/Cc/Cd/De/Df + class rule Eg)
NOISE_RE = re.compile(r"^(Aa|Bb|Cc|Cd|De|Df|Eg)_\d+$")

# rule codes that 05 can select individually with --new_rule (based on ORDER_COMMON).
# each code is enabled with the --Aa_1 … --Df_1 flags.
RULE_CODES = [
    "Aa_1", "Aa_2", "Aa_3", "Aa_4", "Aa_5", "Aa_6",
    "Cc_1", "Cc_2", "Cc_3", "Cc_4", "Cc_5",
    "Cd_1", "Cd_2", "Cd_3", "Cd_4",
    "De_1", "De_2", "De_3", "De_4",
    "Df_1",
]

# ── task label values to 'exclude' per dataset ────────────────────────────────────────
# rows with a listed value in the given task column (the column chosen by --task) are fully excluded from candidates.
#   cic17/cic18 labeling leaves unlabeled/conflicting sessions with special markers in task1:
#     damaged / unlabeled / double / none
#   damaged is data corruption, so always excluded. unlabeled/double are also caught by Aa_6 refinement,
#   but when extracting by task1 the label itself is meaningless, so they are excluded by value too.
EXCLUDE_LABELS = {
    "cic17": {"damaged", "unlabeled", "double", "none"},
    "cic18": {"damaged", "double", "none"},
    # tor16: only 5~6 original sessions, and even those are low quality (Aa_1 incomplete handshake/DNS), so
    #        denoised=0 after refinement. Untrainable class, so excluded entirely.
    "tor16": {"twitter", "aim", "google"},
}

META_COLS = ["filename", "session_id", "task1", "task2", "task3"]
OUT_COLS  = ["filename", "session_id", "proto", "stream",
             "group_key", "task1", "task2", "task3"]
DEFAULT_CHUNK = 500_000


def detect_noise_cols(header):
    return [c for c in header if NOISE_RE.match(c)]


# ── 04 CSV streaming load + per-class reservoir sampling ─────────────────────────
def _task_key(row_task1, row_task2, row_task3, task):
    if task == "full":
        return row_task1 + "_" + row_task2 + "_" + row_task3
    return {"task1": row_task1, "task2": row_task2, "task3": row_task3}[task]


def _reservoir_worker(job):
    """
    Worker: reads part of the files, applies filters, and builds partial per-class reservoirs.
    job = (file_list, usecols, eval_cols, excl, task, mode, cap, unlimited, seed)
    Returns: (part_reservoir, part_counts, n_total, n_excluded, n_selected)
      part_reservoir = {key: [(fn,sid,t1,t2,t3), ...]}  (at most cap per class)
      part_counts    = {key: seen}                        (passed sessions per class)
    """
    import zlib
    (file_list, usecols, eval_cols, excl, task, mode, cap, unlimited,
     seed, chunk_rows, early_target) = job

    reservoirs = {}          # key -> {"rows":[], "seen":int, "rng":Generator}
    counts = {}
    n_total = n_excluded = n_selected = 0

    def _rng_for(key):
        h = zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF
        return np.random.default_rng(seed + h)

    # early stop (skip saturated classes):
    #   a class that has filled cap×mult(=early_target) is 'saturated'. Sessions of saturated classes are
    #   no longer stored; only seen is counted (saves compute). If all passed sessions of a file are in saturated classes,
    #   leave that file early and move to the next. Even in 'file=class' layouts such as iot23, each file is read
    #   only as much as needed, so all files are visited but the amount read drops sharply.
    def _is_sat(key):
        if unlimited or early_target is None:
            return False
        s = reservoirs.get(key)
        return s is not None and s["seen"] >= early_target

    for fp in file_list:
        file_stop = False
        for ch in pd.read_csv(fp, dtype=str, na_filter=False, encoding="utf-8-sig",
                              usecols=usecols, chunksize=chunk_rows):
            n_total += len(ch)
            t1 = ch["task1"].to_numpy(); t2 = ch["task2"].to_numpy(); t3 = ch["task3"].to_numpy()
            if task == "full":
                gk = np.char.add(np.char.add(np.char.add(np.char.add(t1, "_"), t2), "_"), t3)
            else:
                gk = {"task1": t1, "task2": t2, "task3": t3}[task]
            if excl:
                gk_low = np.char.lower(gk.astype(str))
                keep_mask = ~np.isin(gk_low, list(excl))
                n_excluded += int((~keep_mask).sum())
            else:
                keep_mask = np.ones(len(ch), dtype=bool)
            if eval_cols:
                noisy = (ch[eval_cols].to_numpy() == "1").any(axis=1)
            else:
                noisy = np.zeros(len(ch), dtype=bool)
            if mode == "denoised":
                keep_mask &= ~noisy
            passed = keep_mask
            n_selected += int(passed.sum())

            fn = ch["filename"].to_numpy(); sid = ch["session_id"].to_numpy()
            all_passed_saturated = True          # whether all passed rows of this chunk are in saturated classes
            for i in np.nonzero(passed)[0]:
                key = str(gk[i])
                counts[key] = counts.get(key, 0) + 1
                slot = reservoirs.get(key)
                if slot is None and not unlimited:
                    slot = {"rows": [], "seen": 0, "rng": _rng_for(key)}
                    reservoirs[key] = slot
                if unlimited:
                    reservoirs.setdefault(key, {"rows": []})["rows"].append(
                        (fn[i], sid[i], str(t1[i]), str(t2[i]), str(t3[i])))
                    all_passed_saturated = False
                    continue
                # saturated class → only count seen and skip
                if early_target is not None and slot["seen"] >= early_target:
                    slot["seen"] += 1
                    continue
                all_passed_saturated = False
                rec = (fn[i], sid[i], str(t1[i]), str(t2[i]), str(t3[i]))
                slot["seen"] += 1
                if len(slot["rows"]) < cap:
                    slot["rows"].append(rec)
                else:
                    j = int(slot["rng"].integers(0, slot["seen"]))
                    if j < cap:
                        slot["rows"][j] = rec
            del ch
            # if this file is single-class (iot23), no need to read it further once that class is saturated
            if (early_target is not None and not unlimited
                    and passed.any() and all_passed_saturated):
                file_stop = True
                break
        # file_stop → next file (full traversal continues)

    part = {k: (v["rows"] if not unlimited else v["rows"])
            for k, v in reservoirs.items()}
    return part, counts, n_total, n_excluded, n_selected


def stream_load_sample(dataset, input_glob, chunk_rows, clean_rules,
                       task, mode, excl_labels, spc, seed, show_progress=False,
                       file_workers=1, early_stop_mult=3):
    """
    Streams file by file and applies filtering/sampling to limit how much is held in memory.
    If file_workers > 1, many files, e.g. 1156, are read by parallel processes.

    Processing order (same for sequential/parallel):
      1) group_key(task label)  2) excluded labels  3) denoised noise removal  4) per-class reservoir

    Early stop (early_stop_mult): once a class has cap×mult passed rows it is 'saturated',
      and when all (assigned) classes are saturated, reading stops. Files are shuffled with the seed
      to remove per-class file bias (e.g. in iot23 file=attack type), so it is fast without front bias.
      Larger mult is closer to fully uniform (slower); smaller is faster (slightly biased). 0 disables it (full).

    Parallel merge: each worker returns partial per-class reservoirs (at most cap) + passed counts (seen) →
      the main process resamples weighted by seen, merging equivalently to a fully uniform sample.
    """
    in_dir = Path(ds.root(dataset)) / INPUT_SUBDIR
    if not in_dir.exists():
        return None, f"input dir not found: {in_dir}"
    pattern = input_glob or f"session_stat_{dataset}_*_labeled.csv"
    files = sorted(in_dir.glob(pattern))
    if not files:
        return None, f"no CSV matched: {in_dir}/{pattern}"

    header = list(pd.read_csv(files[0], nrows=0, encoding="utf-8-sig").columns)
    noise_cols = detect_noise_cols(header)
    eval_cols = [c for c in clean_rules if c in noise_cols]
    usecols = META_COLS + eval_cols

    excl = {v.lower() for v in (excl_labels or set())}
    unlimited = (spc is None or spc <= 0)
    cap = None if unlimited else int(spc)

    # early-stop target: cap × mult. If mult<=0 or unlimited, disabled (None=full traversal).
    early_target = None
    if not unlimited and early_stop_mult and early_stop_mult > 0:
        early_target = cap * int(early_stop_mult)

    # file shuffle — removes per-class file bias (e.g. iot23 filename=attack type).
    # fixed seed keeps reproducibility. Key to avoiding front bias under early stop.
    files = list(files)
    np.random.default_rng(seed).shuffle(files)

    nw = max(1, int(file_workers))
    # distribute shuffled files round-robin across workers (classes spread evenly per worker)
    buckets = [[] for _ in range(nw)]
    for i, fp in enumerate(files):
        buckets[i % nw].append(fp)
    # unique seed per bucket (reflects bucket index) → reproducible
    jobs = [(b, usecols, eval_cols, excl, task, mode, cap, unlimited,
             seed + 1000 * bi, chunk_rows, early_target)
            for bi, b in enumerate(buckets) if b]

    n_total = n_excluded = n_selected = 0
    merged = {}          # key -> {"rows":[], "seen":int, "rng":Generator}

    def _rng_for(key):
        import zlib
        h = zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF
        return np.random.default_rng(seed + h)

    def _merge(part, counts):
        # absorb partial reservoir into the main reservoir weighted by seen
        for key, recs in part.items():
            seen = counts.get(key, len(recs))
            slot = merged.get(key)
            if slot is None:
                slot = {"rows": [], "seen": 0, "rng": _rng_for(key)}
                merged[key] = slot
            if unlimited:
                slot["rows"].extend(recs)
                slot["seen"] += seen
                continue
            # recs is a cap-size sample of the seen rows this part represents. Apply to the full reservoir:
            # each rec represents 'seen/len(recs)' rows, extending Algorithm R.
            weight = max(1, seen // max(1, len(recs)))
            for rec in recs:
                slot["seen"] += weight
                if len(slot["rows"]) < cap:
                    slot["rows"].append(rec)
                else:
                    j = int(slot["rng"].integers(0, slot["seen"]))
                    if j < cap:
                        slot["rows"][j] = rec

    total_files = len(files)
    if nw == 1:
        # sequential — iterate files directly and advance the progress bar per file
        import zlib
        cap_reservoirs = {}       # key -> {"rows":[], "seen":int, "rng":Generator}

        def _rng2(key):
            h = zlib.adler32(key.encode("utf-8")) & 0xFFFFFFFF
            return np.random.default_rng(seed + h)

        bar = tqdm(total=total_files, desc=f"[{dataset}/{task}/{mode}]",
                   unit="file", disable=not show_progress)
        for fp in files:                          # shuffled order
            for ch in pd.read_csv(fp, dtype=str, na_filter=False,
                                  encoding="utf-8-sig", usecols=usecols,
                                  chunksize=chunk_rows):
                n_total += len(ch)
                t1 = ch["task1"].to_numpy(); t2 = ch["task2"].to_numpy()
                t3 = ch["task3"].to_numpy()
                if task == "full":
                    gk = np.char.add(np.char.add(np.char.add(
                        np.char.add(t1, "_"), t2), "_"), t3)
                else:
                    gk = {"task1": t1, "task2": t2, "task3": t3}[task]
                if excl:
                    gk_low = np.char.lower(gk.astype(str))
                    keep_mask = ~np.isin(gk_low, list(excl))
                    n_excluded += int((~keep_mask).sum())
                else:
                    keep_mask = np.ones(len(ch), dtype=bool)
                if eval_cols:
                    noisy = (ch[eval_cols].to_numpy() == "1").any(axis=1)
                else:
                    noisy = np.zeros(len(ch), dtype=bool)
                if mode == "denoised":
                    keep_mask &= ~noisy
                n_selected += int(keep_mask.sum())
                fn = ch["filename"].to_numpy(); sid = ch["session_id"].to_numpy()
                all_passed_saturated = True
                for i in np.nonzero(keep_mask)[0]:
                    key = str(gk[i])
                    if unlimited:
                        cap_reservoirs.setdefault(
                            key, {"rows": []})["rows"].append(
                            (fn[i], sid[i], str(t1[i]), str(t2[i]), str(t3[i])))
                        all_passed_saturated = False
                        continue
                    slot = cap_reservoirs.get(key)
                    if slot is None:
                        slot = {"rows": [], "seen": 0, "rng": _rng2(key)}
                        cap_reservoirs[key] = slot
                    if early_target is not None and slot["seen"] >= early_target:
                        slot["seen"] += 1
                        continue
                    all_passed_saturated = False
                    rec = (fn[i], sid[i], str(t1[i]), str(t2[i]), str(t3[i]))
                    slot["seen"] += 1
                    if len(slot["rows"]) < cap:
                        slot["rows"].append(rec)
                    else:
                        j = int(slot["rng"].integers(0, slot["seen"]))
                        if j < cap:
                            slot["rows"][j] = rec
                del ch
                # all passed rows of this file are in saturated classes → stop reading this file, go to next file
                if (early_target is not None and not unlimited
                        and keep_mask.any() and all_passed_saturated):
                    break
            bar.update(1)                        # one file done
            if show_progress:
                kept = sum(len(s["rows"]) for s in cap_reservoirs.values())
                bar.set_postfix_str(
                    f"read={n_total:,} kept={kept:,} cls={len(cap_reservoirs)}")
        bar.close()
        merged = cap_reservoirs
    else:
        from multiprocessing import Pool
        # progress bar total = number of files. Advance by the bucket file count when each bucket finishes.
        bar = tqdm(total=total_files, desc=f"[{dataset}/{task}/{mode}]",
                   unit="file", disable=not show_progress)
        job_file_counts = [len(j[0]) for j in jobs]
        with Pool(nw) as pool:
            # imap(order preserved) → fixed merge order → deterministic result
            for idx, (part, counts, nt, ne, ns) in enumerate(
                    pool.imap(_reservoir_worker, jobs)):
                n_total += nt; n_excluded += ne; n_selected += ns
                _merge(part, counts)
                if show_progress:
                    kept = sum(len(s["rows"]) for s in merged.values())
                    bar.update(job_file_counts[idx])
                    bar.set_postfix_str(f"read={n_total:,} kept={kept:,} cls={len(merged)}")
        bar.close()

    # merged → DataFrame
    rows = []
    for key, slot in merged.items():
        for (fn, sid, a1, a2, a3) in slot["rows"]:
            rows.append((fn, sid, a1, a2, a3, key))
    if rows:
        sampled = pd.DataFrame(rows, columns=["filename", "session_id",
                                              "task1", "task2", "task3", "group_key"])
    else:
        sampled = pd.DataFrame(columns=["filename", "session_id",
                                        "task1", "task2", "task3", "group_key"])
    # (approx.) actual passed count per class: if unlimited, stored rows = actual; if capped, seen (approx. when saturated).
    #   basis for the --at_least check. Minority classes never saturate, so they are counted exactly.
    class_counts = {}
    for key, slot in merged.items():
        if unlimited:
            class_counts[key] = len(slot["rows"])
        else:
            class_counts[key] = int(slot.get("seen", len(slot["rows"])))

    stats = {"total": n_total, "excluded": n_excluded,
             "selected": n_selected, "sampled": len(sampled),
             "class_counts": class_counts}
    return (sampled, noise_cols, files, eval_cols, stats), None


# ── group_key(label) construction ─────────────────────────────────────────────────────
def add_group_key(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "full":
        df["group_key"] = df[["task1", "task2", "task3"]].agg("_".join, axis=1)
    else:
        df["group_key"] = df[task]
    return df


# ── session_id → proto / stream ─────────────────────────────────────────────
def add_proto_stream(df: pd.DataFrame) -> pd.DataFrame:
    sid = df["session_id"].astype(str)
    sp = sid.str.split("_", n=1, expand=True)
    if sp.shape[1] == 2:
        df["proto"]  = np.where(sid == "-", "-", sp[0])
        df["stream"] = np.where(sid == "-", "-", sp[1].fillna("-"))
    else:
        df["proto"] = "-"
        df["stream"] = "-"
    return df


# ── deterministic sampling of at most n per class (before stratify) ─────────────────────
def sample_per_class(df: pd.DataFrame, key_col: str, n: int, seed: int):
    """
    Keep at most n per class (all if n<=0). Deterministic:
    within each class, sort by (filename, session_id), shuffle with seed → first n.
    """
    if n is None or n <= 0 or len(df) == 0:
        return df
    rng = np.random.default_rng(seed)
    work = df.sort_values([key_col, "filename", "session_id"], kind="mergesort")
    keep = []
    for _, grp in work.groupby(key_col, sort=True):
        idx = grp.index.to_numpy()
        idx = idx[rng.permutation(len(idx))]
        keep.extend(idx[:n].tolist())
    return df.loc[df.index.isin(keep)].reset_index(drop=True)


# ── stratified train/test split (deterministic, no sklearn needed) ───────────────────
def _session_test_frac(filename, session_id, seed):
    """[0,1) hash value determined only by session identity (crc32, deterministic, process-independent)."""
    key = f"{seed}|{filename}|{session_id}".encode("utf-8")
    return (zlib.crc32(key) & 0xFFFFFFFF) / 0xFFFFFFFF


def stratified_split(df: pd.DataFrame, key_col: str, test_ratio: float, seed: int):
    """
    ★ Leak prevention (rewritten 2026-08-16): the train/test role is decided by 'session identity (filename+session_id)'
      alone. → whichever mode, noisy or denoised, processes it, the same session always
      gets the same role, so cross-mode leakage of denoised_test sessions into noisy_train is 0.
      (The old version used an independent stratified split per mode → denoised_test ⊂ noisy_train leakage.)

      Absolute hash threshold (session_frac < test_ratio → test). Per-class ratios are approximately preserved
      by hash uniformity (large classes are very close to test_ratio). key_col is kept for signature compatibility.
      Same seed + same session → always the same result.
    """
    if len(df) == 0:
        empty = df.copy()
        return empty, empty

    keys = (str(seed) + "|" + df["filename"].astype(str)
            + "|" + df["session_id"].astype(str))
    frac = keys.map(lambda k: (zlib.crc32(k.encode("utf-8")) & 0xFFFFFFFF)
                    / 0xFFFFFFFF)
    test_mask = (frac < test_ratio).to_numpy()

    train = df.loc[~test_mask].reset_index(drop=True)
    test  = df.loc[test_mask].reset_index(drop=True)
    return train, test


def process_dataset(dataset, task, mode, input_glob, chunk_rows,
                    test_ratio, seed, spc, dry_run, new_rule, selected_rules,
                    show_progress=False, file_workers=1, early_stop_mult=3,
                    at_least=100, label_map_scope="task"):
    # decide which refinement rules to apply
    if new_rule:
        clean_rules = list(selected_rules)
    else:
        clean_rules = get_default_clean_rules(dataset)

    excl_labels = EXCLUDE_LABELS.get(dataset, set())
    # streaming load + per-class reservoir sampling (does not load everything into memory)
    res, err = stream_load_sample(dataset, input_glob, chunk_rows, clean_rules,
                                  task, mode, excl_labels, spc, seed, show_progress,
                                  file_workers, early_stop_mult)
    if err:
        return f"[SKIP] {dataset}: {err}"
    sel, noise_cols, files, eval_cols, stats = res
    total = stats["total"]
    n_excluded = stats["excluded"]
    n_selected = stats["selected"]
    n_sampled = stats["sampled"]

    sel = add_proto_stream(sel)

    # ── exclude minority classes (--at_least) ───────────────────────────────────────
    #   if a class has fewer than at_least actual passed rows, the whole class is excluded.
    #   (it appears in none of label_map/train/test). Disabled if at_least<=0.
    #   Disabled if at_least=None (default).
    if at_least is None:
        at_least = 0
    class_counts = stats.get("class_counts", {})
    dropped = {}
    if at_least and at_least > 0 and len(sel):
        present = list(sel["group_key"].unique())
        # decide by actual passed count (falls back to current sel row count if absent)
        cnt_of = {k: int(class_counts.get(k, int((sel["group_key"] == k).sum())))
                  for k in present}
        keep_classes = {k for k, c in cnt_of.items() if c >= at_least}
        dropped = {k: c for k, c in cnt_of.items() if c < at_least}
        if dropped:
            sel = sel[sel["group_key"].isin(keep_classes)].reset_index(drop=True)

    tr, te = stratified_split(sel.reset_index(drop=True),
                              "group_key", test_ratio, seed)

    # label_map = group_key of selected sessions. Created if missing, union-updated if present.
    #   scope="task": shared under task (same ids for noisy/denoised).
    #   scope="mode": independent under task/mode (only classes kept in each mode, exactly matching the lists).
    keys_here = sorted(set(sel["group_key"].unique())) if len(sel) else []
    # new layout: per-mode lists + shared label_map under 01_dataset/{dataset}/00_filelist/
    fl_dir = LIST_BASE / dataset / "00_filelist"
    label_map_path = (fl_dir / "label_map.json" if label_map_scope == "task"
                      else fl_dir / f"label_map_{mode}.json")
    if label_map_path.exists():
        label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        for k in keys_here:
            if k not in label_map:
                label_map[k] = len(label_map)
    else:
        label_map = {k: i for i, k in enumerate(keys_here)}

    out_dir = fl_dir
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        tr[OUT_COLS].to_csv(out_dir / f"list_train_{mode}.csv",
                            index=False, encoding="utf-8-sig")
        te[OUT_COLS].to_csv(out_dir / f"list_test_{mode}.csv",
                            index=False, encoding="utf-8-sig")
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    t = total or 1
    tag = "[DRY-RUN] " if dry_run else ""
    spc_txt = f"{spc}" if (spc and spc > 0) else "unlimited"
    lines = [
        f"\n{tag}[{dataset}] task={task} mode={mode}",
        f"  input files      : {len(files)}  (noise cols: {len(noise_cols)})",
    ]
    if mode == "denoised":
        src = "explicit(--new_rule)" if new_rule else "dataset default"
        applied = eval_cols if eval_cols else ["(none)"]
        lines.append(f"  clean rules      : {', '.join(applied)}  [{src}]")
    if n_excluded:
        lines.append(f"  excluded labels  : {n_excluded:,} "
                     f"({', '.join(sorted(EXCLUDE_LABELS.get(dataset, set())))})")
    lines += [
        f"  total sessions   : {total:,}",
        f"  mode selected    : {n_selected:,} "
        f"({n_selected/t*100:.1f}%)"
        + ("" if mode == "noisy"
           else f"   noisy-dropped: {total-n_selected:,}"),
        f"  sample_per_class : {spc_txt}  → after sampling: {n_sampled:,}",
        f"  at_least         : {at_least if at_least and at_least > 0 else 'disabled'}"
        + (f"  → excluded classes: {len(dropped)}" if dropped else ""),
        f"  classes          : {len(label_map)}",
        f"  train/test       : {len(tr):,} / {len(te):,}",
    ]
    if dropped:
        for k in sorted(dropped):
            lines.append(f"    [excluded] {k:28s} {dropped[k]:,} (< {at_least})")
    if not dry_run:
        lines.append(f"  written to       : {out_dir}")
        lines.append(f"  label_map        : {label_map_path}")
    # per-class count summary (after sampling)
    for k in keys_here:
        c = int((sel["group_key"] == k).sum())
        lines.append(f"    {k:30s} {mode}={c:,}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="04 labeling CSV → {noisy|denoised} train/test filelist")
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="dataset alias (multiple allowed, 'all'=all)")
    ap.add_argument("--task", default="task3",
                    choices=["task1", "task2", "task3", "full"],
                    help="task column used as label (default task3 — fixed for SCIE experiments)")
    ap.add_argument("--mode", choices=["noisy", "denoised", "both"],
                    default="both",
                    help="generation mode (default both = noisy and denoised)")
    ap.add_argument("--denoised", action="store_true",
                    help="(backward compat) same as --mode denoised")
    ap.add_argument("--sample_per_class", type=int, default=0,
                    help="max samples per class (0=unlimited). Deterministic sampling before split.")
    ap.add_argument("--at_least", type=int, default=None,
                    help="min samples per class. Classes whose actual passed count is below this are "
                         "excluded entirely. 0=disabled (keep all). "
                         "unset=0 (disabled). "
                         "if given, the value is enforced for all datasets.")
    ap.add_argument("--label_map_scope", choices=["task", "mode"], default="task",
                    help="label_map.json location/scope. task=shared under task (noisy/denoised "
                         "same ids, default). mode=independent under task/mode (only classes kept in each mode, "
                         "exactly matching the lists). Use mode to apply at_least differently per mode.")
    ap.add_argument("--input-glob", default=None,
                    help="input CSV glob (default: session_stat_{dataset}_*_labeled.csv)")
    ap.add_argument("--test-ratio", type=float, default=0.2,
                    help="test ratio per class (default 0.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="split/sampling seed (default 42)")
    ap.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK,
                    help=f"chunk streaming rows (default {DEFAULT_CHUNK:,})")
    ap.add_argument("--workers", type=int, default=1,
                    help="number of parallel processes over datasets (default 1=sequential). "
                         "processes multiple datasets concurrently.")
    ap.add_argument("--file_workers", type=int, default=1,
                    help="number of processes reading files in parallel within one dataset (default 1). "
                         "use to quickly read datasets with many files (1156) such as iot23. "
                         "e.g.: --file_workers 16")
    ap.add_argument("--early_stop_mult", type=int, default=3,
                    help="early-stop multiplier (default 3). Once a class collects sample_per_class×mult rows "
                         "it is saturated; when all classes are saturated the remaining files are skipped. "
                         "files are shuffled with the seed, so it speeds up without front bias. "
                         "tune with 3/5/10 etc. (larger = more uniform/slower, smaller = faster). "
                         "0=disabled (full traversal, fully uniform). Ignored if sample_per_class is unlimited.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print statistics only, without writing")
    # ── refinement rule selection ──
    ap.add_argument("--new_rule", action="store_true",
                    help="ignore the dataset default refinement rules; use the --Aa_1 … flags below and "
                         "apply only the enabled rules. (if unset, the default rules of noise_rule_{ds}.py apply)")
    grp = ap.add_argument_group(
        "refinement rule flags (with --new_rule)",
        "enable rules for the denoised decision individually. Sessions with 1 in an enabled rule column are removed.")
    for code in RULE_CODES:
        grp.add_argument(f"--{code}", action="store_true",
                         help=f"apply rule {code}")
    args = ap.parse_args()

    # collect rules enabled via --new_rule
    selected_rules = [c for c in RULE_CODES if getattr(args, c, False)]
    if args.new_rule and not selected_rules:
        print("[WARN] --new_rule given but no rule flag enabled → no refinement (keep all)")

    if args.denoised:
        modes = ["denoised"]
    elif args.mode == "both":
        modes = ["noisy", "denoised"]
    else:
        modes = [args.mode]

    requested = []
    for d in args.dataset:
        requested.extend(ds.ALL_DATASETS if d == "all" else [d])
    seen = set()
    targets = [d for d in requested if not (d in seen or seen.add(d))]

    # warn about settings where the cap/early stop suppresses counts so large classes may be wrongly excluded by at_least.
    #   with unlimited (spc=0), actual passed counts are exact, so it is safe.
    if (args.at_least and args.at_least > 0 and args.sample_per_class
            and args.sample_per_class > 0):
        eff = args.sample_per_class * (args.early_stop_mult
                                       if args.early_stop_mult and args.early_stop_mult > 0
                                       else 1)
        if eff < args.at_least:
            print(f"[WARN] sample_per_class×early_stop_mult({eff}) < at_least"
                  f"({args.at_least}) → early stop caps large-class counts, so they "
                  f"may be wrongly excluded. For an exact --at_least check, "
                  f"set sample_per_class to 0 (unlimited) or increase early_stop_mult.")

    print(f"[targets] {targets} | task={args.task} modes={modes} "
          f"sample_per_class={args.sample_per_class or 'unlimited'} "
          f"at_least={'disabled' if args.at_least is None else (args.at_least or 'disabled')} "
          f"test_ratio={args.test_ratio} seed={args.seed} "
          f"workers={args.workers} dry_run={args.dry_run}")
    print("=" * 60)

    for mode in modes:
        work = partial(process_dataset, task=args.task, mode=mode,
                       input_glob=args.input_glob, chunk_rows=args.chunk_rows,
                       test_ratio=args.test_ratio, seed=args.seed,
                       spc=args.sample_per_class, dry_run=args.dry_run,
                       new_rule=args.new_rule, selected_rules=selected_rules,
                       file_workers=args.file_workers,
                       early_stop_mult=args.early_stop_mult,
                       at_least=args.at_least,
                       label_map_scope=args.label_map_scope)

        n_workers = max(1, min(args.workers, len(targets)))
        if n_workers > 1 and len(targets) > 1:
            # independent per dataset (separate input CSVs and output dirs) → safe for process parallelism.
            with Pool(n_workers) as pool:
                for line in pool.imap(work, targets):
                    print(line)
        else:
            for dataset in targets:
                print(work(dataset, show_progress=True))
    print("\n" + "=" * 60
          + f"\n[done] {len(targets)} dataset(s) × {len(modes)} mode(s)\n")


if __name__ == "__main__":
    main()