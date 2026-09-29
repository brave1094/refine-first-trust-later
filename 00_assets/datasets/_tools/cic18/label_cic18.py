#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_cic18_labeling.py

Label my CSE-CIC-IDS2018 session feature CSVs' task1/task2/task3 by the OFFICIAL
ATTACK SCHEDULE (not by joining the author CSVs).

Why schedule-based (not a CSV join like cic17):
  The CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" CSVs have NO
  Source/Destination IP and no Flow ID (only Dst Port, Protocol, Timestamp,
  features, Label). A session-level 5-tuple join is therefore impossible.
  CIC itself labeled per-flow using the attack schedule + src/dst IP + port +
  protocol, so applying that schedule directly reproduces their labels exactly.

Calibration (measured from the data, not assumed):
  - IP scheme : attacker = public valid IP (18.x/13.x/52.x),
                victim   = internal 172.31.69.x  (both aliases listed below)
  - timezone  : CIC18 local = UTC-4 (EDT). DoS-Hulk flood sits at 17:56-17:58 UTC
                = 13:56-13:58 local, inside the scheduled 13:45-14:19 window.
                schedule local + 4h = UTC.  (--utc-offset -4)

Rule (per session):
  attack if  ts_first in [start-TOL, end+TOL]  AND  {src,dst} matches
             {attacker set, victim set} (unordered)  for some schedule entry.
  else Benign.   No external CSV to miss, so "Benign" is the default (not
  "unlabeled"); the only non-classified buckets are damaged / double.

  task1: Benign|Malware   task2: family   task3: technique  (LABEL_MAP below)

Output : <out-dir> (default <in-dir>_labeled), same filenames, task1/2/3 edited.
Counts :
  labeled_session : matched exactly 1 attack (Malware)
  benign_session  : no attack matched (Benign default)
  double_session  : matched >=2 distinct attacks (overlapping windows; ~0)
  damaged_session : column count != header, or ts_first unparsable
  total_session
"""

import argparse, csv, os, glob, sys, datetime
from collections import Counter
from multiprocessing import Pool
from tqdm import tqdm

csv.field_size_limit(10**9)

# ---- label -> (task1, task2, task3) -----------------------------------------
LABEL_MAP = {
    "Benign":           ("Benign",  "Benign",       "Benign"),
    "FTP-BruteForce":   ("Malware", "Patator",      "FTP-Patator"),
    "SSH-BruteForce":   ("Malware", "Patator",      "SSH-Patator"),
    "DoS-GoldenEye":    ("Malware", "DoS",          "GoldenEye"),
    "DoS-Slowloris":    ("Malware", "DoS",          "Slowloris"),
    "DoS-Hulk":         ("Malware", "DoS",          "Hulk"),
    "DoS-SlowHTTPTest": ("Malware", "DoS",          "Slowhttptest"),
    "DDoS-LOIC-HTTP":   ("Malware", "DDoS",         "LOIC-HTTP"),
    "DDoS-LOIC-UDP":    ("Malware", "DDoS",         "LOIC-UDP"),
    "DDoS-HOIC":        ("Malware", "DDoS",         "HOIC"),
    "BruteForce-Web":   ("Malware", "Web-Attack",   "Brute-Force"),
    "BruteForce-XSS":   ("Malware", "Web-Attack",   "XSS"),
    "SQL-Injection":    ("Malware", "Web-Attack",   "Sql-Injection"),
    "Infiltration":     ("Malware", "Infiltration", "Infiltration"),
    "Bot":              ("Malware", "Bot",          "Bot"),
}

# ---- IP groups (internal 172.31.x + public valid IP aliases) -----------------
VIC_25 = {"172.31.69.25", "18.217.21.148"}
VIC_28 = {"172.31.69.28", "18.218.83.150"}
VIC_24 = {"172.31.69.24", "18.221.148.137"}
VIC_13 = {"172.31.69.13", "18.216.254.154"}
DDOS_ATK = {"18.218.115.60", "18.219.9.1", "18.219.32.43", "18.218.55.126",
            "52.14.136.135", "18.219.5.43", "18.216.200.189", "18.218.229.235",
            "18.218.11.51", "18.216.24.42"}
BOT_VIC = {"172.31.69.23", "172.31.69.17", "172.31.69.14", "172.31.69.12",
           "172.31.69.10", "172.31.69.8", "172.31.69.6", "172.31.69.26",
           "172.31.69.29", "172.31.69.30",
           "18.217.218.111", "18.222.10.237", "18.222.86.193", "18.222.62.221",
           "13.59.9.106", "18.222.102.2", "18.219.212.0", "18.216.105.13",
           "18.219.163.126", "18.216.164.12"}

# ---- attack schedule: (date, start, end[local], attacker set, victim set, attack)
SCHEDULE_RAW = [
    ("2018-02-14", "10:32", "12:09", {"172.31.70.4", "18.221.219.4"},  VIC_25, "FTP-BruteForce"),
    ("2018-02-14", "14:01", "15:31", {"172.31.70.6", "13.58.98.64"},   VIC_25, "SSH-BruteForce"),
    ("2018-02-15", "09:26", "10:09", {"172.31.70.46", "18.219.211.138"}, VIC_25, "DoS-GoldenEye"),
    ("2018-02-15", "10:59", "11:40", {"172.31.70.8", "18.217.165.70"}, VIC_25, "DoS-Slowloris"),
    ("2018-02-16", "10:12", "11:08", {"172.31.70.23", "13.59.126.31"}, VIC_25, "DoS-SlowHTTPTest"),
    ("2018-02-16", "13:45", "14:19", {"172.31.70.16", "18.219.193.20"}, VIC_25, "DoS-Hulk"),
    ("2018-02-20", "10:12", "11:17", DDOS_ATK, VIC_25, "DDoS-LOIC-HTTP"),
    ("2018-02-20", "13:13", "13:32", DDOS_ATK, VIC_25, "DDoS-LOIC-UDP"),
    ("2018-02-21", "10:09", "10:43", DDOS_ATK, VIC_28, "DDoS-LOIC-UDP"),
    ("2018-02-21", "14:05", "15:05", DDOS_ATK, VIC_28, "DDoS-HOIC"),
    ("2018-02-22", "10:17", "11:24", {"18.218.115.60"}, VIC_28, "BruteForce-Web"),
    ("2018-02-22", "13:50", "14:29", {"18.218.115.60"}, VIC_28, "BruteForce-XSS"),
    ("2018-02-22", "16:15", "16:29", {"18.218.115.60"}, VIC_28, "SQL-Injection"),
    ("2018-02-23", "10:03", "11:03", {"18.218.115.60"}, VIC_28, "BruteForce-Web"),
    ("2018-02-23", "13:00", "14:10", {"18.218.115.60"}, VIC_28, "BruteForce-XSS"),
    ("2018-02-23", "15:05", "15:18", {"18.218.115.60"}, VIC_28, "SQL-Injection"),
    ("2018-02-28", "10:50", "12:05", {"13.58.225.34"},  VIC_24, "Infiltration"),
    ("2018-02-28", "13:42", "14:40", {"13.58.225.34"},  VIC_24, "Infiltration"),
    ("2018-03-01", "09:57", "10:55", {"13.58.225.34"},  VIC_13, "Infiltration"),
    ("2018-03-01", "14:00", "15:37", {"13.58.225.34"},  VIC_13, "Infiltration"),
    ("2018-03-02", "10:11", "11:34", {"18.219.211.138"}, BOT_VIC, "Bot"),
    ("2018-03-02", "14:24", "15:55", {"18.219.211.138"}, BOT_VIC, "Bot"),
]

# ---- globals shared with workers via fork -----------------------------------
SCHED = None        # [(start_epoch, end_epoch, atk_set, vic_set, attack_key), ...]
COL = None
NCOLS = None
TOL = 120.0
CONFLICT = "priority"


def build_schedule(utc_offset):
    tz = datetime.timezone(datetime.timedelta(hours=utc_offset))
    out = []
    for date, st, en, atk, vic, akey in SCHEDULE_RAW:
        y, m, d = map(int, date.split("-"))
        sh, sm = map(int, st.split(":"))
        eh, em = map(int, en.split(":"))
        s = datetime.datetime(y, m, d, sh, sm, tzinfo=tz).timestamp()
        e = datetime.datetime(y, m, d, eh, em, tzinfo=tz).timestamp()
        out.append((s, e, atk, vic, akey))
    return out


def process_chunk(rows):
    """Edit task1/2/3 in place; return (rows, category_counter, label_counter)."""
    ci, cnt, lbl = COL, Counter(), Counter()
    i1, i2, i3 = ci["task1"], ci["task2"], ci["task3"]
    i_s, i_d, i_t = ci["src_ip"], ci["dst_ip"], ci["ts_first"]
    for row in rows:
        if len(row) != NCOLS:
            cnt["damaged"] += 1
            continue
        try:
            tf = float(row[i_t])
        except ValueError:
            row[i1], row[i2], row[i3] = "damaged", "none", "none"
            cnt["damaged"] += 1
            continue
        s, d = row[i_s], row[i_d]
        hits = set()
        for st, en, atk, vic, akey in SCHED:
            if tf < st - TOL or tf > en + TOL:
                continue
            if (s in atk and d in vic) or (s in vic and d in atk):
                hits.add(akey)
        if not hits:
            t1, t2, t3 = LABEL_MAP["Benign"]; cnt["benign"] += 1
        elif len(hits) == 1:
            t1, t2, t3 = LABEL_MAP[next(iter(hits))]; cnt["labeled"] += 1
        else:                                    # overlapping schedules (2+) -> marked double
            t1, t2, t3 = "double", "none", "none"; cnt["double"] += 1
        row[i1], row[i2], row[i3] = t1, t2, t3
        lbl["double" if t1 == "double" else (f"{t2}/{t3}" if t1 == "Malware" else t1)] += 1
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
    ap.add_argument("--in-dir", default="../03_session_stat")
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>_labeled")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--utc-offset", type=int, default=-4, help="CIC18 local UTC offset (EDT=-4)")
    ap.add_argument("--time_correction", type=float, default=0.0,
                    help="seconds by which the session window is widened on both sides (default 0 = none). "
                         "e.g. 60 -> [start-60, end+60]")
    ap.add_argument("--chunk", type=int, default=5000)
    args = ap.parse_args()

    global SCHED, COL, NCOLS, TOL
    TOL = args.time_correction
    SCHED = build_schedule(args.utc_offset)

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

    with open(inputs[0], encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    COL = {n: i for i, n in enumerate(header)}
    NCOLS = len(header)
    for need in ("task1", "task2", "task3", "src_ip", "dst_ip", "ts_first"):
        if need not in COL:
            sys.exit(f"[ERR] required column missing: {need}")
    print(f"[PLAN] inputs={len(inputs)} offset={args.utc_offset}h time_correction={TOL}s -> {out_dir}")

    grand, glbl = Counter(), Counter()
    with Pool(args.workers) as pool:
        for p in inputs:
            outp = os.path.join(out_dir, os.path.basename(p))
            cnt, lbl = process_file(p, outp, pool, args.chunk)
            grand += cnt; glbl += lbl
            print(f"[FILE] {os.path.basename(p)}  {dict(cnt)}")

    m = grand["labeled"]; b = grand["benign"]; j = grand["double"]; k = grand["damaged"]
    print("\n" + "=" * 40)
    print(f"labeled_session : {m}")
    print(f"benign_session : {b}")
    print(f"double_labeled_session : {j}")
    print(f"damaged_session : {k}")
    print(f"total_session : {m + b + j + k}")
    print("=" * 40)
    print(f"[task2/task3 dist] {dict(glbl.most_common())}")


if __name__ == "__main__":
    main()