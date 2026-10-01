# -*- coding: utf-8 -*-
"""Map the released session lists onto your own session names (run after step 3, before step 6).

The released lists (01_dataset/<dataset>/00_filelist*/list_*.csv) name sessions by our file names. Your captures,
arranged from the official release, may carry other file names. This script recomputes the content key of every
session in your 03_session_stat (02_preprocess/session_keys.py), looks up the key of every listed session in
01_dataset/<dataset>/session_keys.csv, and rewrites filename/session_id in the lists to your names.

usage:
    python 02_preprocess/align_session_lists.py --dataset vpn16 [--root <NM_DATASET_ROOT>] [--dry-run]

The originals are kept once as list_*.published.csv. A report (01_dataset/<dataset>/align_report.txt) gives, per list,
how many sessions were found, how many are missing (not in your captures), and pkt_count mismatches.
When the same content occurs more than once (identical copies of a session, as in the official tor16 release), the
listed copies are matched one-to-one to your copies in natural file-name order, so the lists keep their rows."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, glob, os, re, shutil, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from session_keys import endpoints, key_from_parts, key_from_stat_row  # noqa: E402

DIRS = {"ustc16": "01_USTC-TFC_2016", "cic17": "02_CIC-IDS-2017", "cic18": "03_CIC-IDS-2018", "iot23": "04_CIC_IoT_Dataset_2023",
        "vpn16": "21_ISCX-VPN-2016", "tor16": "22_ISCX-TOR-2016", "tls1.3": "23_CSTNET_TLS1.3", "cispec": "24_CipherSpectrum"}
csv.field_size_limit(2**31 - 1)


def natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DIRS))
    ap.add_argument("--root", default=str(RP.DATASETS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds_dir = os.path.join(REPO, "01_dataset", a.dataset)

    ours = {}                                                    # (filename, session_id) -> (key, pkt_count)
    for r in csv.DictReader(open(os.path.join(ds_dir, "session_keys.csv"), encoding="utf-8")):
        ours[(r["filename"], r["session_id"])] = (key_from_parts(r["l4"], r["endpoint_a"], r["endpoint_b"], r["ts_first"]), r["pkt_count"])
    wanted = {k for k, _ in ours.values()}

    theirs = defaultdict(list)                                   # key -> [(filename, session_id, pkt_count), ...]
    stats = sorted(glob.glob(os.path.join(a.root, DIRS[a.dataset], "03_session_stat", f"session_stat_{a.dataset}_*.csv")), key=natkey)
    if not stats:
        sys.exit("no 03_session_stat files found — run step 3 first")
    for f in stats:
        with open(f, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                k = key_from_stat_row(r)
                if k not in wanted:
                    continue
                theirs[k].append((r["filename"], r["session_id"], r["pkt_count"]))

    report = [f"{a.dataset}: listed sessions {len(ours)} | found in your captures {sum(1 for k, _ in ours.values() if k in theirs)}"
              f" | contents present more than once in your captures {sum(len(v) > 1 for v in theirs.values())}"]
    for lf in sorted(glob.glob(os.path.join(ds_dir, "00_filelist*", "list_*.csv"))):
        if lf.endswith(".published.csv"):
            continue
        pub = lf[:-4] + ".published.csv"
        src = pub if os.path.exists(pub) else lf
        rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
        # identical copies: the i-th listed copy (natural order of our names) -> the i-th copy of yours
        #   (a copy that also carries the same file name and session id is taken first)
        assign, taken = {}, defaultdict(set)
        pending = []
        for r in rows:
            name = (r["filename"], r["session_id"])
            k, _ = ours.get(name, (None, None))
            same = [c for c in theirs.get(k, []) if (c[0], c[1]) == name]
            if same:
                assign[name] = same[0]
                taken[k].add(name)
            elif k in theirs:
                pending.append((name, k))
        for name, k in sorted(pending, key=lambda x: (natkey(x[0][0]), x[0][1])):
            free = [c for c in theirs[k] if (c[0], c[1]) not in taken[k]] or theirs[k]
            assign[name] = free[0]
            taken[k].add((free[0][0], free[0][1]))
        out, miss, pk = [], 0, 0
        for r in rows:                                            # the output keeps the row order of the list
            m = assign.get((r["filename"], r["session_id"]))
            if m is None:
                miss += 1
                continue
            fn, sid, n2 = m
            pk += str(ours[(r["filename"], r["session_id"])][1]) != str(n2)
            row = {**r, "filename": fn, "session_id": sid}
            if "stream" in r:                                    # proto/stream locate the session in whole captures
                row["proto"], row["stream"] = sid.split("_", 1)
            out.append(row)
        report.append(f"  {os.path.relpath(lf, ds_dir)}: {len(out)}/{len(rows)} aligned | missing {miss} | pkt_count mismatch {pk}")
        if a.dry_run:
            continue
        if not os.path.exists(pub):
            shutil.copyfile(lf, pub)
        with open(lf, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(out)
    print("\n".join(report))
    if not a.dry_run:
        open(os.path.join(ds_dir, "align_report.txt"), "w", encoding="utf-8").write("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
