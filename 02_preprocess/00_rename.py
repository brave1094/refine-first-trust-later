# -*- coding: utf-8 -*-
"""00_rename.py: place the official captures under the exact file names of our 01_pcap (before 01_session_split.py).

    python 02_preprocess/00_rename.py --dataset tor16 --src /downloads/ISCX-Tor-2016 [--mode symlink] [--dry-run]

    -> <NM_DATASET_ROOT>/<dataset dir>/01_pcap/<class>/<our file name>

--src: the folder where the official archives were extracted (vpn16: NonVPN-PCAPs-01..03.zip, VPN-PCAPS-01.zip,
VPN-PCAPs-02.zip; tor16: Tor.zip, NonTor.tar.xz; cispec: the three cipher zips; cic17: the PCAPs folder; cic18: the day
folders with their pcap.zip extracted; ustc16: a clone of github.com/davidyslu/USTC-TFC2016 with every .7z extracted).
An official file is found by its file name and the folders named in official_file (e.g. Tor/AUDIO_tor_spotify.pcap,
Friday-02-03-2018/capDESKTOP-AN3U28N-172.31.64.17.pcap), wherever the archive was extracted; case is ignored.

lib_rename/<ds>.csv (.csv.gz for cispec) was built by comparing the official files with ours packet by packet (every placed file is
byte-identical to ours; checked on the full download). A row is one of:
  identical                   the official file under our name            (--mode: copy, hardlink, or symlink)
  same packets                an official pcapng converted to pcap        (editcap -F pcap)
  ... cut at 16 MiB           converted and cut to cut_to_bytes bytes     (tor16: our copy of Torrent01.pcapng)
  repaired copy  (01_pcap)    the first cut_to_bytes bytes of another file of ours: the complete packets before a
                              damaged end, kept in <class>/.salvaged/ as in our pipeline (tor16, 11 files)
  merged ... (mergecap)       several official parts merged in time order (ustc16 SMB, Weibo: mergecap -F pcap)
  name map ... placed         cic18: the name map made when we placed the captures (4,006 hosts x days)
  same name per day           cic17: <Day>-WorkingHours.pcap -> 0N_<Day>/0N_<Day>.pcap
  unused                      an official file not used in the article
Existing files are never overwritten. Every placed file is checked against our file (size, first-packet time, and the
number of packets where the map gives it)."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, gzip, os, shutil, struct, subprocess, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {"ustc16": "01_USTC-TFC_2016", "cic17": "02_CIC-IDS-2017", "cic18": "03_CIC-IDS-2018", "iot23": "04_CIC_IoT_Dataset_2023",
        "vpn16": "21_ISCX-VPN-2016", "tor16": "22_ISCX-TOR-2016", "tls1.3": "23_CSTNET_TLS1.3", "cispec": "24_CipherSpectrum"}
DATASETS = sorted(d for d in DIRS if any(os.path.exists(os.path.join(HERE, "lib_rename", d + e)) for e in (".csv", ".csv.gz")))


def count(path, first_only=False):
    """(complete packets, first-packet time in microseconds) of a pcap file; first_only: packets is None"""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(24)
        e = "<" if head[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
        nano = head[:4] in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        n, first, pos = 0, None, 24
        while True:
            h = fh.read(16)
            if len(h) < 16:
                return n, first
            sec, frac, incl, _ = struct.unpack(e + "IIII", h)
            if pos + 16 + incl > size:
                return n, first
            if first is None:
                first = sec * 1_000_000 + (frac // 1000 if nano else frac)
                if first_only:
                    return None, first
            fh.seek(incl, 1)
            pos += 16 + incl
            n += 1


def in_order(official, path):
    """the folders of official_file appear, in order, among the folders of the file on disk (case ignored)"""
    want, have = official.lower().split("/")[:-1], iter(path.lower().split("/")[:-1])
    return all(any(w == h for h in have) for w in want)


def copy_prefix(src, dst, nbytes):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        left = nbytes
        while left:
            b = fi.read(min(left, 1 << 22))
            if not b:
                raise IOError(f"{src} is shorter than {nbytes} bytes")
            fo.write(b)
            left -= len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--src", required=True)
    ap.add_argument("--root", default=str(RP.DATASETS), help="dataset root (default: NM_DATASET_ROOT or 00_assets/datasets)")
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    mp = os.path.join(HERE, "lib_rename", f"{a.dataset}.csv")
    rows = list(csv.DictReader(gzip.open(mp + ".gz", "rt", encoding="utf-8") if os.path.exists(mp + ".gz") else open(mp, encoding="utf-8")))
    pcap_dir = os.path.join(a.root, DIRS[a.dataset], "01_pcap")

    found, unknown = {}, []                                  # official path inside the archive -> file on disk
    by_name = defaultdict(list)                              # matched on the file name, then on the folders above it
    for r in rows:
        if r["archive"] != "(01_pcap)":
            by_name[os.path.basename(r["official_file"]).lower()].append(r["official_file"])
    for cur, _, files in os.walk(a.src):
        for f in files:
            p = os.path.join(cur, f).replace("\\", "/")
            ks = [k for k in by_name.get(f.lower(), []) if in_order(k, p)]
            for k in ks:
                found.setdefault(k, p)
            if not ks and f.lower().endswith((".pcap", ".pcapng")):
                unknown.append(p)

    placed, kept, unused, missing, bad = 0, 0, 0, [], []
    targets = defaultdict(list)                              # (class, our file) -> its row(s); several rows = parts to merge
    for r in rows:
        if r["our_file"]:
            targets[(r["class"], r["our_file"])].append(r)
        else:
            unused += r["official_file"] in found
    for (cls, our), rs in sorted(targets.items(), key=lambda x: x[1][0]["archive"] == "(01_pcap)"):  # repaired copies last
        r = rs[0]
        dst = os.path.join(pcap_dir, cls, our)
        if r["archive"] == "(01_pcap)":
            srcs = [os.path.join(pcap_dir, r["official_file"])]
            lost = [] if a.dry_run or os.path.exists(srcs[0]) else [r["official_file"]]
        else:
            srcs = [found.get(x["official_file"]) for x in rs]
            lost = [x["official_file"] for x, s in zip(rs, srcs) if s is None]
        if lost:
            missing += lost
            continue
        if os.path.lexists(dst):
            kept += 1
        elif a.dry_run:
            placed += 1                                   # would be placed
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cut = int(r["cut_to_bytes"]) if r["cut_to_bytes"] else None
            src = srcs[0]
            if len(srcs) > 1:                                # ustc16 SMB, Weibo: official parts merged in time order
                subprocess.run(["mergecap", "-F", "pcap", "-w", dst] + srcs, check=True)
            elif src.lower().endswith(".pcapng"):
                subprocess.run(["editcap", "-F", "pcap", src, dst], check=True)
                if cut:
                    os.truncate(dst, cut)
            elif cut:
                copy_prefix(src, dst, cut)
            else:
                {"copy": shutil.copyfile, "hardlink": os.link, "symlink": os.symlink}[a.mode](os.path.abspath(src), dst)
            placed += 1
        if not a.dry_run and os.path.exists(dst):
            got = (os.path.getsize(dst), *count(dst, first_only=not r["packets"]))
            want = (int(r["size"]), int(r["packets"]) if r["packets"] else None, int(r["first_packet_us"]))
            if got != want:
                bad.append(f"{cls}/{our}: size/packets/first-packet {got} != map {want}")
    n_t = len(targets)
    print(f"{a.dataset}: targets {n_t} | {'would place' if a.dry_run else 'placed'} {placed} | already there {kept} | "
          f"official files not used in the article {unused} | not in the map {len(unknown)} | missing {len(missing)} | "
          f"check failures {len(bad)}")
    for x in missing[:30]:
        print("  missing:", x)
    for x in unknown[:30]:
        print("  not in the map:", x)
    for x in bad[:30]:
        print("  check:", x)
    sys.exit(1 if missing or bad else 0)


if __name__ == "__main__":
    main()
