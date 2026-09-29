#!/usr/bin/env python3
"""
cispec: verify every class directory name (domain) against the actual SNI in
its pcaps. Flags any dir whose dominant ClientHello SNI does not match the
domain encoded in the directory name (none_<cipher>_<domain>).

Run from <dataset root>/24_CipherSpectrum/00_origin  (default ROOT = ../01_pcap),
or pass the pcap root as argv[1].

Requires: tshark, tqdm
"""

import sys
import subprocess
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../01_pcap")
SAMPLE = 20          # SNIs to sample per class dir
PROCS = 8
TSHARK = "tshark"
PCAP_EXTS = {".pcap", ".pcapng", ".cap"}
OUT_CSV = Path("cispec_sni_check.csv")


def parse_dir(name: str):
    """none_<cipher>_<domain> -> (cipher, domain). Domains have no '_'."""
    parts = name.split("_", 2)
    if len(parts) != 3 or parts[0] != "none":
        return None, None
    return parts[1], parts[2]


def get_sni(pcap: Path):
    try:
        out = subprocess.run(
            [TSHARK, "-r", str(pcap), "-Y", "tls.handshake.type==1",
             "-T", "fields", "-e", "tls.handshake.extensions_server_name"],
            capture_output=True, text=True, timeout=30,
        )
        for line in out.stdout.splitlines():
            line = line.strip()
            if line:
                return line.split(",")[0].strip().lower()
    except Exception:
        return None
    return None


def domain_match(sni: str, dom: str) -> bool:
    dom = dom.lower()
    return sni == dom or sni.endswith("." + dom)


def check_dir(d: Path):
    cipher, dom = parse_dir(d.name)
    if dom is None:
        return None
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in PCAP_EXTS)
    snis = Counter()
    got = 0
    for p in files:
        if got >= SAMPLE:
            break
        s = get_sni(p)
        if s:
            snis[s] += 1
            got += 1
    if not snis:
        return (d.name, cipher, dom, "(no SNI)", 0, got, "NO_SNI")
    top, topn = snis.most_common(1)[0]
    status = "OK" if domain_match(top, dom) else "MISMATCH"
    return (d.name, cipher, dom, top, topn, got, status)


def main():
    if not ROOT.exists():
        print(f"[ERR] root not found: {ROOT.resolve()}")
        sys.exit(1)
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("none_"))
    print(f"root  : {ROOT.resolve()}")
    print(f"dirs  : {len(dirs)}   sample/dir: {SAMPLE}   procs: {PROCS}\n")

    results = []
    with Pool(PROCS) as pool:
        for res in tqdm(pool.imap_unordered(check_dir, dirs),
                        total=len(dirs), desc="SNI check", unit="dir"):
            if res:
                results.append(res)

    results.sort(key=lambda r: r[0])
    mism = [r for r in results if r[6] == "MISMATCH"]
    nosni = [r for r in results if r[6] == "NO_SNI"]

    # write full CSV record
    with open(OUT_CSV, "w") as f:
        f.write("dir,cipher,dir_domain,top_sni,top_count,sampled,status\n")
        for r in results:
            f.write(",".join(str(x) for x in r) + "\n")

    print(f"\n==== RESULT : {len(results)} dirs ====")
    print(f"OK       : {sum(1 for r in results if r[6]=='OK')}")
    print(f"MISMATCH : {len(mism)}")
    print(f"NO_SNI   : {len(nosni)}")

    if mism:
        print("\n-- MISMATCH (dir_domain  !=  actual SNI) --")
        print(f"   {'dir':40s} {'dir_domain':22s} {'actual SNI':28s} (n/sampled)")
        for r in mism:
            print(f"   {r[0]:40s} {r[2]:22s} {r[3]:28s} ({r[4]}/{r[5]})")
    if nosni:
        print("\n-- NO_SNI (couldn't read SNI; check manually) --")
        for r in nosni:
            print(f"   {r[0]}")

    print(f"\nfull record -> {OUT_CSV}")
    if not mism:
        print("=> all directory names match their SNI. nothing to fix.")
    elif len(mism) == 1:
        print("=> exactly ONE mismatched class. fix only that one.")


if __name__ == "__main__":
    main()