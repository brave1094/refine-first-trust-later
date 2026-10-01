# -*- coding: utf-8 -*-
"""Compare your per-class session counts with ours (run after step 3, or after step 4).

Counts sessions per class (task3) in <root>/<dataset dir>/03_session_stat/ (03_session_stat_labeled/ for cic17 and
cic18, whose classes come from the official labels) and compares them with
99_documents/results/dataset_stats/<dataset>/01_origin_<dataset>.csv, the counts behind the article's dataset tables.
A class that differs points to a capture that was placed in the wrong class, is missing, or is extra.

usage: python 00_assets/datasets/_tools/verify_counts.py --dataset vpn16 [--root <NM_DATASET_ROOT>]"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, glob, json, os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(RP.ROOT)
LAYOUT = json.load(open(os.path.join(HERE, "expected_layout.json"), encoding="utf-8"))
csv.field_size_limit(2**31 - 1)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, choices=sorted(LAYOUT))
ap.add_argument("--root", default=str(RP.DATASETS))
a = ap.parse_args()
ref = {r["class"]: int(r["count"]) for r in csv.DictReader(open(
    os.path.join(str(RP.DATASET_STATS), a.dataset, f"01_origin_{a.dataset}.csv"), encoding="utf-8-sig"))}
mine = Counter()
sub = "03_session_stat_labeled" if a.dataset in ("cic17", "cic18") else "03_session_stat"
for f in glob.glob(os.path.join(a.root, LAYOUT[a.dataset]["dir"], sub, f"session_stat_{a.dataset}_*.csv")):
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            mine[r["task3"]] += 1
ok = 0
print(f"{'class (task3)':40s} {'yours':>12s} {'ours':>12s}")
for c in sorted(set(ref) | set(mine)):
    same = mine.get(c, 0) == ref.get(c, 0)
    ok += same
    print(f"{c:40s} {mine.get(c, 0):12d} {ref.get(c, 0):12d}{'' if same else '   <-- differs'}")
print(f"{ok}/{len(set(ref) | set(mine))} classes identical | sessions yours {sum(mine.values())} ours {sum(ref.values())}")
