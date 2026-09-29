# -*- coding: utf-8 -*-
"""Step 1 for CSE-CIC-IDS2018: place the official per-host captures under the names used in the article.

The official release (AWS, "Original Network Traffic and Log data/<Day>/pcap.zip") holds one capture per host and
day, e.g. <Day>/pcap/capDESKTOP-AN3U28N-172.31.64.17.pcap. rename_map.csv (4,006 rows, nine days) gives, for every
capture, the name used in our session lists: 01_pcap/<Day>/<Day>_<i>.pcap. This restores the exact file names.

usage:
    python 00_assets/datasets/_tools/cic18/apply_rename_map.py --src /downloads/cse-cic-ids2018 [--mode symlink] [--dry-run]
--src: the folder that contains the extracted day folders (Friday-02-03-2018/, Friday-16-02-2018/, ...)."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(RP.ROOT)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--src", required=True)
ap.add_argument("--root", default=str(RP.DATASETS))
ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
M = {(r["directory"], r["original_name"]): r["new_path"] for r in csv.DictReader(open(os.path.join(HERE, "rename_map.csv"), encoding="utf-8-sig"))}
days = {d for d, _ in M}
done, unknown = set(), []
for cur, _, files in os.walk(a.src):
    parts = cur.replace("\\", "/").split("/")
    day = next((p for p in reversed(parts) if p in days), None)
    for f in files:
        if not f.lower().endswith((".pcap", ".pcapng")):
            continue
        new = M.get((day, f))
        if new is None:
            unknown.append(os.path.join(cur, f))
            continue
        dst = os.path.join(a.root, "03_CIC-IDS-2018", new)
        done.add((day, f))
        if a.dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        {"copy": shutil.copyfile, "hardlink": os.link, "symlink": os.symlink}[a.mode](os.path.abspath(os.path.join(cur, f)), dst)
missing = sorted(set(M) - done)
print(f"cic18: placed {len(done)}/{len(M)} captures{' (dry run)' if a.dry_run else ''} | not in the map {len(unknown)} | missing {len(missing)}")
for x in missing[:30]:
    print("  missing:", x)
for x in unknown[:30]:
    print("  not in the map:", x)
