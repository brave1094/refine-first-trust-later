#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_check_cic18_labeled.py

Audit the CSE-CIC-IDS2018 labeling quality.

Reads my labeled session CSVs from ../03_session_stat_labeled/ and RE-DERIVES the
set of attacks each session matches against the embedded official schedule (same
logic the labeler used). cic18 is schedule-based, so:
  - there is no "no-label" bucket: a session that matches no attack IS Benign
    (the default), which is a real label, not a miss.
  - "Benign + attack" mixing cannot occur: the schedule only yields attack
    matches, so a multi-label session is always attack + attack.

Reports:
  - total / damaged
  - benign (no attack matched) / single-attack (properly labeled attack)
  - multi-attack (>=2 distinct attacks) + which combos
  - per-class distribution of the FINAL labels in the CSV (task1/2/3)

Run with --labeled-dir <cic18 dir>/03_session_stat_labeled (the default is relative to
the working directory).
"""

import argparse, csv, os, glob, sys, datetime
from collections import Counter
from tqdm import tqdm

csv.field_size_limit(10**9)
TOL = 120.0

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


def build_schedule(off):
    tz = datetime.timezone(datetime.timedelta(hours=off))
    out = []
    for date, st, en, atk, vic, akey in SCHEDULE_RAW:
        y, m, d = map(int, date.split("-"))
        sh, sm = map(int, st.split(":")); eh, em = map(int, en.split(":"))
        s = datetime.datetime(y, m, d, sh, sm, tzinfo=tz).timestamp()
        e = datetime.datetime(y, m, d, eh, em, tzinfo=tz).timestamp()
        out.append((s, e, atk, vic, akey))
    return out


def count_rows(path):
    with open(path, "rb") as f:
        return sum(b.count(b"\n") for b in iter(lambda: f.read(1 << 20), b"")) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled-dir", default="../03_session_stat_labeled")
    ap.add_argument("--utc-offset", type=int, default=-4)
    ap.add_argument("--tol", type=float, default=120.0)
    args = ap.parse_args()
    global TOL
    TOL = args.tol
    SCHED = build_schedule(args.utc_offset)

    inputs = sorted(glob.glob(os.path.join(args.labeled_dir,
                                           "session_stat_cic18_*.csv")))
    if not inputs:
        sys.exit(f"[ERR] no session_stat_cic18_*.csv under {args.labeled_dir!r}")
    print(f"[PLAN] files={len(inputs)} offset={args.utc_offset}h tol={TOL}s")

    total = damaged = benign = single = multi = 0
    perclass = Counter()
    aa_combo = Counter()

    for p in inputs:
        nrows = max(count_rows(p), 0)
        with open(p, encoding="utf-8-sig", newline="") as fh:
            r = csv.reader(fh); h = next(r); ci = {n: i for i, n in enumerate(h)}
            NC = len(h)
            i_s, i_d, i_t = ci["src_ip"], ci["dst_ip"], ci["ts_first"]
            for row in tqdm(r, total=nrows, desc=os.path.basename(p), unit="row"):
                total += 1
                if len(row) != NC:
                    damaged += 1
                    continue
                perclass[(row[ci["task1"]], row[ci["task2"]], row[ci["task3"]])] += 1
                try:
                    tf = float(row[i_t])
                except ValueError:
                    damaged += 1
                    continue
                s, d = row[i_s], row[i_d]
                S = set()
                for st, en, atk, vic, akey in SCHED:
                    if tf < st - TOL or tf > en + TOL:
                        continue
                    if (s in atk and d in vic) or (s in vic and d in atk):
                        S.add(akey)
                if not S:
                    benign += 1
                elif len(S) == 1:
                    single += 1
                else:
                    multi += 1
                    aa_combo[" + ".join(sorted(S))] += 1

    print("\n" + "=" * 52)
    print("CIC18 LABEL AUDIT")
    print("=" * 52)
    print(f"total_session             : {total}")
    print(f"damaged_session           : {damaged}")
    print(f"benign_session            : {benign}   (no attack matched = default)")
    print(f"single_attack_session     : {single}   (properly labeled attack)")
    print(f"multi_attack_session(>=2) : {multi}   (attack + attack)")
    print(f"no_label_session          : 0   (N/A: schedule-based, Benign is the default)")
    print(f"Benign+attack mixed       : 0   (N/A: schedule yields no Benign match)")
    print("-" * 52)
    print("attack+attack combos (count):")
    for k, v in aa_combo.most_common():
        print(f"    {v:8d}  {k}")
    print("-" * 52)
    print("per-class distribution (final task1/task2/task3 in CSV):")
    for (t1, t2, t3), v in perclass.most_common():
        print(f"    {v:9d}  {t1}/{t2}/{t3}")


if __name__ == "__main__":
    main()