#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_cic17_label.py

Write CIC-IDS-2017 author labels into my session feature CSVs' task1/task2/task3.

  task1 : Benign | Malware
  task2 : attack family   (DoS, Web-Attack, Patator, ... ; Benign->Benign)
  task3 : specific technique (Hulk, GoldenEye, XSS, ... ; Benign->Benign)
          mapping table is LABEL_MAP near the top of this file (easy to edit)

Input  feature CSVs : <in-dir>            (default ../03_session_stat)
CIC label CSVs      : <cic-dir>/*.csv     (default ./TrafficLabelling_)
Output              : <out-dir>           (default <in-dir>_labeled, same filenames)

Join rule (one label per session):
  key   = day + UNORDERED 5-tuple {(ip,port),(ip,port)} + L4 proto
  guard = CIC flow timestamp within [ts_first-TOL, ts_last+TOL]
  AM/PM = CIC 12h clock lacks AM/PM marker, so each CIC flow is indexed at both
          its parsed time and +12h; the pcap window selects the correct twin.
  tz    = CIC local = ADT (UTC-3); my ts_* are UTC  -> default --cic-utc-offset -3

Per-session category (mutually exclusive, sums to total):
  damaged   : column count != header, or ts_first/ts_last unparsable
  unlabeled : 0 matched CIC labels (non-tcp/udp, or no CIC flow for the key)
  labeled   : exactly 1 distinct matched label
  double    : >= 2 distinct matched labels (conflict)

For 'double', the written label is resolved by --conflict:
  priority (default) : most frequent non-BENIGN label (else BENIGN)
  join               : task3 = 'A|B|...' (sorted distinct), task1=malware if any attack
"""

import argparse, csv, os, glob, sys, datetime
from collections import defaultdict, Counter
from multiprocessing import Pool
from tqdm import tqdm

csv.field_size_limit(10**9)

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
PROTO_MAP = {"tcp": "6", "udp": "17"}
TS_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
              "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M")
HALF_DAY = 43200.0

# ---- globals shared with workers via fork (set before Pool is created) ------
CIC_INDEX = None        # {(day,ip1,p1,ip2,p2,proto): [(utc_ts, label), ...]}
COL = None              # {column_name: index}
NCOLS = None
TOL = 120.0
CONFLICT = "priority"

# ---- label -> (task1, task2, task3) -----------------------------------------
# task1: benign|malware   task2: attack family   task3: specific technique
# No spaces / no underscores; words joined with '-'. Edit freely.
LABEL_MAP = {
    "BENIGN":                     ("Benign",  "Benign",       "Benign"),
    "FTP-Patator":                ("Malware", "Patator",      "FTP-Patator"),
    "SSH-Patator":                ("Malware", "Patator",      "SSH-Patator"),
    "DoS Hulk":                   ("Malware", "DoS",          "Hulk"),
    "DoS GoldenEye":              ("Malware", "DoS",          "GoldenEye"),
    "DoS slowloris":              ("Malware", "DoS",          "Slowloris"),
    "DoS Slowhttptest":           ("Malware", "DoS",          "Slowhttptest"),
    "Heartbleed":                 ("Malware", "Heartbleed",   "Heartbleed"),
    "Web Attack - Brute Force":   ("Malware", "Web-Attack",   "Brute-Force"),
    "Web Attack - XSS":           ("Malware", "Web-Attack",   "XSS"),
    "Web Attack - Sql Injection": ("Malware", "Web-Attack",   "Sql-Injection"),
    "Infiltration":               ("Malware", "Infiltration", "Infiltration"),
    "Bot":                        ("Malware", "Bot",          "Bot"),
    "PortScan":                   ("Malware", "PortScan",     "PortScan"),
    "DDoS":                       ("Malware", "DDoS",         "LOIC"),
}


def map_label(cic_label):
    """CIC label string -> (task1, task2, task3); fallback for unknowns."""
    if cic_label in LABEL_MAP:
        return LABEL_MAP[cic_label]
    safe = "-".join(cic_label.split())          # unknown: strip spaces
    return ("Malware", safe, safe)


def norm_label(s):
    s = s.strip().replace("\x96", "-").replace("\x97", "-")
    return " ".join(s.split())


def parse_ts(s, off):
    s = s.strip()
    if not s:
        return None
    for fmt in TS_FORMATS:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            tz = datetime.timezone(datetime.timedelta(hours=off))
            return dt.replace(tzinfo=tz).timestamp()
        except ValueError:
            pass
    return None


def day_of(name):
    low = name.lower()
    for d in DAYS:
        if d.lower() in low:
            return d
    return None


def canon_key(day, ip1, p1, ip2, p2, proto):
    a, b = (ip1, p1), (ip2, p2)
    if b < a:
        a, b = b, a
    return (day, a[0], a[1], b[0], b[1], proto)


def build_cic_index(cic_dir, utc_offset, days_needed):
    idx = defaultdict(list)
    files = sorted(glob.glob(os.path.join(cic_dir, "*.csv")))
    if not files:
        sys.exit(f"[ERR] no CSVs under {cic_dir!r}")
    by_day = Counter()
    for f in files:
        day = day_of(os.path.basename(f))
        if day is None:
            print(f"[WARN] unknown day, skip: {os.path.basename(f)}")
            continue
        if days_needed and day not in days_needed:
            continue
        n = 0
        with open(f, encoding="latin-1") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) < 8:
                    continue
                ts = parse_ts(row[6], utc_offset)
                if ts is None:
                    continue
                k = canon_key(day, row[1].strip(), row[2].strip(),
                              row[3].strip(), row[4].strip(), row[5].strip())
                idx[k].append((ts, norm_label(row[-1])))   # AM/PM twin checked at query
                n += 1
        by_day[day] += n
        print(f"[CIC] {os.path.basename(f):<52} {day:<9} rows={n:,}")
    for k in idx:
        idx[k].sort(key=lambda x: x[0])
    print(f"[CIC] keys={len(idx):,}  rows_by_day={dict(by_day)}")
    return idx


def resolve(label_pool):
    """label_pool: list of CIC labels -> (task1, task2, task3)."""
    distinct = set(label_pool)
    if CONFLICT == "join" and len(distinct) >= 2:
        attacks = [l for l in label_pool if l.upper() != "BENIGN"]
        win = Counter(attacks).most_common(1)[0][0] if attacks else "BENIGN"
        t1, t2, _ = map_label(win)
        t3 = "|".join(map_label(l)[2] for l in sorted(distinct))
        return t1, t2, t3
    # priority (also used for single label)
    attacks = [l for l in label_pool if l.upper() != "BENIGN"]
    win = Counter(attacks).most_common(1)[0][0] if attacks else "BENIGN"
    return map_label(win)


def _hit(ts, tf, tl):
    """True if ts (or its +12h AM/PM twin) falls in the session window."""
    return (tf - TOL <= ts <= tl + TOL) or (tf - TOL <= ts + HALF_DAY <= tl + TOL)


def _dist(ts, tf, tl):
    a = min(abs(ts - tf), abs(ts - tl))
    b = min(abs(ts + HALF_DAY - tf), abs(ts + HALF_DAY - tl))
    return min(a, b)


def process_chunk(rows):
    """Edit task1/2/3 in place, return (rows, category_counter, label_counter)."""
    ci, cnt, lbl = COL, Counter(), Counter()
    i_t1, i_t2, i_t3 = ci["task1"], ci["task2"], ci["task3"]
    for row in rows:
        if len(row) != NCOLS:
            cnt["damaged"] += 1
            continue                      # cannot safely edit; leave as-is
        try:
            tf = float(row[ci["ts_first"]]); tl = float(row[ci["ts_last"]])
        except ValueError:
            row[i_t1], row[i_t2], row[i_t3] = "damaged", "none", "none"
            cnt["damaged"] += 1
            continue
        proto = PROTO_MAP.get(row[ci["L4"]].strip().lower())
        cands = None
        if proto is not None:
            key = canon_key(day_of(row[ci["filename"]]),
                            row[ci["src_ip"]], row[ci["src_port"]],
                            row[ci["dst_ip"]], row[ci["dst_port"]], proto)
            cands = CIC_INDEX.get(key)
        if not cands:
            row[i_t1], row[i_t2], row[i_t3] = "unlabeled", "none", "none"
            cnt["unlabeled"] += 1
            continue
        pool = [l for ts, l in cands if _hit(ts, tf, tl)]
        if not pool:
            pool = [min(cands, key=lambda x: _dist(x[0], tf, tl))[1]]
        if len(set(pool)) >= 2:                 # conflict (2+ labels) -> marked double
            row[i_t1], row[i_t2], row[i_t3] = "double", "none", "none"
            cnt["double"] += 1
            lbl["double"] += 1
            continue
        t1, t2, t3 = resolve(pool)
        row[i_t1], row[i_t2], row[i_t3] = t1, t2, t3
        cnt["labeled"] += 1
        lbl[f"{t2}/{t3}" if t1.lower() == "malware" else t1] += 1
    return rows, cnt, lbl


def gen_chunks(reader, size):
    buf = []
    for row in reader:
        buf.append(row)
        if len(buf) >= size:
            yield buf; buf = []
    if buf:
        yield buf


def count_rows(path):
    with open(path, "rb") as f:
        return sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 20), b"")) - 1


def process_file(in_path, out_path, pool, chunk_size):
    total = max(count_rows(in_path), 0)
    cnt, lbl = Counter(), Counter()
    with open(in_path, encoding="utf-8-sig", newline="") as fh, \
         open(out_path, "w", encoding="utf-8-sig", newline="") as fo:
        r = csv.reader(fh)
        header = next(r)
        w = csv.writer(fo)
        w.writerow(header)
        bar = tqdm(total=total, desc=os.path.basename(in_path), unit="row")
        for rows_out, c, l in pool.imap(process_chunk, gen_chunks(r, chunk_size)):
            for row in rows_out:
                w.writerow(row)
            cnt += c; lbl += l
            bar.update(sum(c.values()))
        bar.close()
    return cnt, lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic-dir", default="./TrafficLabelling_")
    ap.add_argument("--in-dir", default="../03_session_stat")
    ap.add_argument("--out-dir", default=None,
                    help="default: <in-dir>_labeled")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cic-utc-offset", type=int, default=-3)
    ap.add_argument("--time_correction", type=float, default=0.0,
                    help="seconds by which the session window is widened on both sides (default 0 = none). "
                         "e.g. 60 -> [ts_first-60, ts_last+60]")
    ap.add_argument("--conflict", choices=("priority", "join"), default="priority")
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()

    global CIC_INDEX, COL, NCOLS, TOL, CONFLICT
    TOL, CONFLICT = args.time_correction, args.conflict

    # output folder: _labeled without correction, _labeled_cor_{N} with it
    if args.out_dir:
        out_dir = args.out_dir
    else:
        base = os.path.normpath(args.in_dir) + "_labeled"
        cor = int(args.time_correction) if args.time_correction == int(args.time_correction) \
              else args.time_correction
        out_dir = base + (f"_cor_{cor}" if args.time_correction > 0 else "")
    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(args.in_dir, "*.csv")))
    if not inputs:
        sys.exit(f"[ERR] no CSVs under {args.in_dir!r}")

    # header / colmap from first file (assumed identical schema)
    with open(inputs[0], encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    COL = {name: i for i, name in enumerate(header)}
    NCOLS = len(header)
    for need in ("filename", "task1", "task2", "task3", "L4",
                 "src_ip", "src_port", "dst_ip", "dst_port",
                 "ts_first", "ts_last"):
        if need not in COL:
            sys.exit(f"[ERR] required column missing: {need}")

    # which days are present across inputs (from filename column)
    days_needed = set()
    for p in inputs:
        days_needed.add(day_of(os.path.basename(p)) or "")
    days_needed = {d for d in days_needed if d}
    if not days_needed:                      # day not in filename -> scan column
        for p in inputs:
            with open(p, encoding="utf-8-sig", newline="") as fh:
                r = csv.reader(fh); next(r)
                for row in r:
                    d = day_of(row[COL["filename"]])
                    if d:
                        days_needed.add(d)
    print(f"[PLAN] inputs={len(inputs)} days={sorted(days_needed)} -> {out_dir}")

    CIC_INDEX = build_cic_index(args.cic_dir, args.cic_utc_offset, days_needed)

    grand = Counter()
    glbl = Counter()
    with Pool(args.workers) as pool:          # fork AFTER index built -> COW share
        for p in inputs:
            outp = os.path.join(out_dir, os.path.basename(p))
            cnt, lbl = process_file(p, outp, pool, args.chunk)
            grand += cnt; glbl += lbl
            print(f"[FILE] {os.path.basename(p)}  {dict(cnt)}")

    m = grand["labeled"]; n = grand["unlabeled"]
    j = grand["double"]; k = grand["damaged"]
    l = m + n + j + k
    print("\n" + "=" * 40)
    print(f"labeled_session : {m}")
    print(f"unlabeled_session : {n}")
    print(f"double_labeled_session : {j}")
    print(f"damaged_session : {k}")
    print(f"total_session : {l}")
    print("=" * 40)
    print(f"[task3/label dist] {dict(glbl.most_common())}")


if __name__ == "__main__":
    main()