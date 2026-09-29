#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_check_cic17_labeled.py

Audit the CIC-IDS-2017 labeling quality.

Reads my labeled session CSVs from ../03_session_stat_labeled/ and RE-DERIVES the
set of original CIC labels each session matches (by re-running the same join the
labeler used against ./TrafficLabelling_/). The labeled CSV keeps only the final
resolved label, so the conflict/mixed breakdown has to be recomputed here.

Reports:
  - total / damaged / no-label(unmatched)
  - single-label (= properly labeled) count
  - multi-label (>=2 distinct CIC labels) count, split into
        Benign+attack   (+ which attack)
        attack+attack   (+ which combo)
  - per-class distribution of the FINAL labels written in the CSV (task1/2/3)

Run with --labeled-dir <cic17 dir>/03_session_stat_labeled and --cic-dir <official
TrafficLabelling CSV dir> (the defaults are relative to the working directory).
"""

import argparse, csv, os, glob, sys, datetime
from collections import defaultdict, Counter
from tqdm import tqdm

csv.field_size_limit(10**9)

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
PROTO_MAP = {"tcp": "6", "udp": "17"}
TS_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
              "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M")
HALF_DAY = 43200.0
TOL = 120.0


def norm_label(s):
    return " ".join(s.strip().replace("\x96", "-").replace("\x97", "-").split())


def parse_ts(s, off):
    s = s.strip()
    if not s:
        return None
    for fmt in TS_FORMATS:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(tzinfo=datetime.timezone(
                datetime.timedelta(hours=off))).timestamp()
        except ValueError:
            pass
    return None


def day_of(name):
    low = name.lower()
    for d in DAYS:
        if d.lower() in low:
            return d
    return None


def canon_key(day, i1, p1, i2, p2, pr):
    a, b = (i1, p1), (i2, p2)
    if b < a:
        a, b = b, a
    return (day, a[0], a[1], b[0], b[1], pr)


def build_cic_index(cic_dir, off, days_needed):
    idx = defaultdict(list)
    files = sorted(glob.glob(os.path.join(cic_dir, "*.csv")))
    if not files:
        sys.exit(f"[ERR] no CSVs under {cic_dir!r}")
    for f in files:
        day = day_of(os.path.basename(f))
        if day is None or (days_needed and day not in days_needed):
            continue
        with open(f, encoding="latin-1") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) < 8:
                    continue
                ts = parse_ts(row[6], off)
                if ts is None:
                    continue
                k = canon_key(day, row[1].strip(), row[2].strip(),
                              row[3].strip(), row[4].strip(), row[5].strip())
                idx[k].append((ts, norm_label(row[-1])))
        print(f"[CIC] indexed {os.path.basename(f)}")
    return idx


def _hit(ts, tf, tl):
    return (tf - TOL <= ts <= tl + TOL) or (tf - TOL <= ts + HALF_DAY <= tl + TOL)


def _dist(ts, tf, tl):
    return min(min(abs(ts - tf), abs(ts - tl)),
               min(abs(ts + HALF_DAY - tf), abs(ts + HALF_DAY - tl)))


def count_rows(path):
    with open(path, "rb") as f:
        return sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 20), b"")) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", default="../03_session_stat_labeled")
    ap.add_argument("--cic-dir", default="./TrafficLabelling_")
    ap.add_argument("--cic-utc-offset", type=int, default=-3)
    ap.add_argument("--tol", type=float, default=120.0)
    args = ap.parse_args()
    global TOL
    TOL = args.tol

    inputs = sorted(glob.glob(os.path.join(args.labeled_dir,
                                           "session_stat_cic17_*.csv")))
    if not inputs:
        sys.exit(f"[ERR] no session_stat_cic17_*.csv under {args.labeled_dir!r}")

    # days present -> only index those CIC files
    days = set()
    for p in inputs:
        with open(p, encoding="utf-8-sig", newline="") as fh:
            r = csv.reader(fh); h = next(r); ci = {n: i for i, n in enumerate(h)}
            for row in r:
                d = day_of(row[ci["filename"]])
                if d:
                    days.add(d)
    print(f"[PLAN] files={len(inputs)} days={sorted(days)}")
    idx = build_cic_index(args.cic_dir, args.cic_utc_offset, days)

    total = damaged = nolabel = single = multi = 0
    ben_attack = attack_attack = 0
    perclass = Counter()          # final (task1,task2,task3) in the CSV
    ba_combo = Counter()          # benign+attack: which attack
    aa_combo = Counter()          # attack+attack: which combo

    for p in inputs:
        nrows = max(count_rows(p), 0)
        with open(p, encoding="utf-8-sig", newline="") as fh:
            r = csv.reader(fh); h = next(r); ci = {n: i for i, n in enumerate(h)}
            NC = len(h)
            for row in tqdm(r, total=nrows, desc=os.path.basename(p), unit="row"):
                total += 1
                if len(row) != NC:
                    damaged += 1
                    continue
                perclass[(row[ci["task1"]], row[ci["task2"]], row[ci["task3"]])] += 1
                try:
                    tf = float(row[ci["ts_first"]]); tl = float(row[ci["ts_last"]])
                except ValueError:
                    damaged += 1
                    continue
                proto = PROTO_MAP.get(row[ci["L4"]].strip().lower())
                cands = None
                if proto is not None:
                    key = canon_key(day_of(row[ci["filename"]]),
                                    row[ci["src_ip"]], row[ci["src_port"]],
                                    row[ci["dst_ip"]], row[ci["dst_port"]], proto)
                    cands = idx.get(key)
                if not cands:
                    nolabel += 1
                    continue
                pool = [l for ts, l in cands if _hit(ts, tf, tl)]
                if not pool:
                    pool = [min(cands, key=lambda x: _dist(x[0], tf, tl))[1]]
                S = set(pool)
                if len(S) == 1:
                    single += 1
                else:
                    multi += 1
                    atk = sorted(l for l in S if l.upper() != "BENIGN")
                    if "BENIGN" in {x.upper() for x in S} and atk:
                        ben_attack += 1
                        for a in atk:
                            ba_combo[a] += 1
                    else:
                        attack_attack += 1
                        aa_combo[" + ".join(atk)] += 1

    print("\n" + "=" * 52)
    print("CIC17 LABEL AUDIT")
    print("=" * 52)
    print(f"total_session            : {total}")
    print(f"damaged_session          : {damaged}")
    print(f"no_label_session         : {nolabel}   (no matching CIC flow)")
    print(f"single_label_session     : {single}   (properly labeled)")
    print(f"multi_label_session(>=2) : {multi}")
    print(f"    Benign + attack      : {ben_attack}")
    print(f"    attack + attack      : {attack_attack}")
    print("-" * 52)
    print("attack+attack combos (count):")
    for k, v in aa_combo.most_common():
        print(f"    {v:8d}  {k}")
    print("Benign+attack, mixed-in attack (count):")
    for k, v in ba_combo.most_common():
        print(f"    {v:8d}  {k}")
    print("-" * 52)
    print("per-class distribution (final task1/task2/task3 in CSV):")
    for (t1, t2, t3), v in perclass.most_common():
        print(f"    {v:9d}  {t1}/{t2}/{t3}")


if __name__ == "__main__":
    main()