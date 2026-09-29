# -*- coding: utf-8 -*-
"""Step 1: arrange the official captures of a public dataset into the layout the pipeline expects.

    <NM_DATASET_ROOT>/<dataset dir>/01_pcap/<class>/<class>_<i>.pcap        class = <task1>_<task2>_<task3>

usage:
    python 00_assets/datasets/_tools/arrange_pcaps.py --dataset ustc16 --src /downloads/USTC-TFC2016 [--mode symlink] [--dry-run]

--src is the folder where the official archive was extracted (see 99_documents/DATASETS.md). Each capture is assigned to one
class by the rules below, files of a class are numbered in natural-sort order of their source path, and a log
(arrange_<dataset>.csv: source -> target) is written next to 01_pcap. Files that match no class or more than one class
are NOT placed; they are listed so that they can be placed by hand.

Rule reliability:
  ustc16, tls1.3, cic17  folder / file names of the official release map one-to-one to our classes (verified)
  vpn16, tor16, iot23    keyword rules reconstructed from the official naming; confirm with verify_counts.py after step 3
  cic18, cispec          use cic18/apply_rename_map.py and cispec/relabel_cispec.py instead (see 99_documents/DATASETS.md)
Whatever the file names, align_session_lists.py matches the released session lists by content, not by file name."""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, json, os, re, shutil, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(RP.ROOT)
LAYOUT = json.load(open(os.path.join(HERE, "expected_layout.json"), encoding="utf-8"))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------- dataset rules: (relative path) -> class or None
def rule_ustc16(rel, classes):
    parts = rel.replace("\\", "/").split("/")
    kind = next((p.lower() for p in parts if p.lower() in ("benign", "malware")), None)
    if kind is None:
        return None
    stem = norm(os.path.splitext(parts[-1])[0])
    hits = [c for c in classes if c.startswith(kind + "_") and stem.startswith(norm(c.split("_", 2)[2]))]
    return max(hits, key=len) if hits else None


def rule_tls13(rel, classes):
    parts = rel.replace("\\", "/").split("/")
    for p in reversed(parts[:-1]):                        # the nearest folder named after a domain
        if f"none_none_{p}" in classes:
            return f"none_none_{p}"
    return None


VPN_TOKENS = {  # (category, app, sub) -> tokens that must all occur in the normalised file name
    ("Chat", "AIM", "none"): ["aim"], ("Chat", "Facebook", "none"): ["facebook", "chat"], ("Chat", "Gmail", "none"): ["gmail", "chat"],
    ("Chat", "Hangouts", "none"): ["hangout", "chat"], ("Chat", "ICQ", "none"): ["icq"], ("Chat", "Skype", "none"): ["skype", "chat"],
    ("Email", "Gmail", "none"): ["email"], ("FileTransfer", "FTPS", "none"): ["ftps"], ("FileTransfer", "SFTP", "none"): ["sftp"],
    ("FileTransfer", "SCP", "none"): ["scp"], ("FileTransfer", "Skype", "file"): ["skype", "file"],
    ("P2P", "BitTorrent", "none"): ["torrent"], ("Streaming", "Netflix", "none"): ["netflix"], ("Streaming", "Spotify", "none"): ["spotify"],
    ("Streaming", "Vimeo", "none"): ["vimeo"], ("Streaming", "Youtube", "none"): ["youtube"],
    ("VoIP", "Facebook", "audio"): ["facebook", "audio"], ("VoIP", "Facebook", "video"): ["facebook", "video"],
    ("VoIP", "Hangouts", "audio"): ["hangout", "audio"], ("VoIP", "Hangouts", "video"): ["hangout", "video"],
    ("VoIP", "Skype", "audio"): ["skype", "audio"], ("VoIP", "Skype", "video"): ["skype", "video"], ("VoIP", "VoIPBuster", "none"): ["voipbuster"],
}
TOR_TOKENS = {
    ("Audio-Streaming", "Spotify", "none"): ["spotify"], ("Browsing", "Browser", "none"): ["browsing"],
    ("Browsing", "Facebook", "none"): ["browsing", "facebook"], ("Browsing", "Google", "none"): ["browsing", "google"],
    ("Browsing", "Twitter", "none"): ["browsing", "twitter"], ("Chat", "AIM", "none"): ["aim"], ("Chat", "Facebook", "none"): ["facebook", "chat"],
    ("Chat", "Hangouts", "none"): ["hangout", "chat"], ("Chat", "ICQ", "none"): ["icq"], ("Chat", "Skype", "none"): ["skype", "chat"],
    ("Email", "Gmail", "none"): ["mail"], ("FileTransfer", "FTP", "none"): ["ftp"], ("FileTransfer", "SFTP", "none"): ["sftp"],
    ("FileTransfer", "Skype", "file"): ["skype", "transfer"], ("P2P", "BitTorrent", "none"): ["p2p"],
    ("Video-Streaming", "Vimeo", "none"): ["vimeo"], ("Video-Streaming", "Youtube", "none"): ["youtube"],
    ("VoIP", "Facebook", "audio"): ["facebook", "voi"], ("VoIP", "Hangouts", "audio"): ["hangout", "voi"], ("VoIP", "Skype", "audio"): ["skype", "voi"],
}
EXCLUDE = {"ftp": ["sftp", "ftps"], "browsing": [], "aim": ["gmail"], "email": [], "mail": ["gmail"]}


def keyword_rule(rel, classes, table, prefix_of):
    name = norm(os.path.basename(rel))
    pre = prefix_of(rel)
    if pre is None:
        return None
    hits = []
    for (cat, app, sub), toks in table.items():
        c = f"{pre}_{cat}_{app}_{sub}"
        if c not in classes:
            continue
        if all(t in name for t in toks) and not any(x in name for t in toks for x in EXCLUDE.get(t, [])):
            hits.append((len("".join(toks)), c))
    if not hits:
        return None
    hits.sort(reverse=True)                                # the most specific rule wins
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return ("AMBIGUOUS", [h[1] for h in hits])
    return hits[0][1]


def vpn_prefix(rel):
    return "VPN" if os.path.basename(rel).lower().startswith("vpn") else "nonVPN"


def tor_prefix(rel):
    parts = [p.lower() for p in rel.replace("\\", "/").split("/")[:-1]]
    if any(p.startswith("nontor") or p.startswith("non-tor") for p in parts):
        return "nonTor"
    return "Tor" if any("tor" in p for p in parts) else None


def rule_iot23(rel, classes):
    parts = rel.replace("\\", "/").split("/")
    cands = [norm(p) for p in parts[:-1]] + [norm(re.sub(r"\d+$", "", os.path.splitext(parts[-1])[0]))]
    for token in reversed(cands):
        if "benign" in token:
            return "benign_benign_benign"
        for c in classes:
            _, fam, tech = c.split("_", 2)
            if token in (norm(fam + tech), norm(tech)):
                return c
    return None


def rule_cic17(rel, classes):
    day = os.path.basename(rel).split("-")[0].lower()        # Monday-WorkingHours.pcap -> monday
    return next((c for c in classes if c.split("_", 1)[1].lower() == day), None)


RULES = {"ustc16": rule_ustc16, "tls1.3": rule_tls13, "iot23": rule_iot23, "cic17": rule_cic17,
         "vpn16": lambda r, c: keyword_rule(r, c, VPN_TOKENS, vpn_prefix),
         "tor16": lambda r, c: keyword_rule(r, c, TOR_TOKENS, tor_prefix)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(RULES))
    ap.add_argument("--src", required=True, help="folder with the extracted official release")
    ap.add_argument("--root", default=str(RP.DATASETS))
    ap.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    info = LAYOUT[a.dataset]
    classes = set(info["classes"])
    found, unmatched, ambiguous = defaultdict(list), [], []
    for cur, _, files in os.walk(a.src):
        for f in files:
            if not f.lower().endswith((".pcap", ".pcapng", ".cap")):
                continue
            rel = os.path.relpath(os.path.join(cur, f), a.src)
            c = RULES[a.dataset](rel, classes)
            if c is None:
                unmatched.append(rel)
            elif isinstance(c, tuple):
                ambiguous.append((rel, c[1]))
            else:
                found[c].append(rel)
    dst_root = os.path.join(a.root, info["dir"], "01_pcap")
    log = []
    for c in sorted(found):
        for i, rel in enumerate(sorted(found[c], key=natkey), 1):
            name = f"{c}.pcap" if a.dataset == "cic17" else f"{c}_{i}.pcap"   # cic17: one capture per day, e.g. 01_Monday.pcap
            dst = os.path.join(dst_root, c, name)
            log.append((rel, os.path.relpath(dst, a.root)))
            if a.dry_run:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            src = os.path.join(a.src, rel)
            if os.path.lexists(dst):
                os.remove(dst)
            {"copy": shutil.copyfile, "hardlink": os.link, "symlink": os.symlink}[a.mode](os.path.abspath(src), dst)
    if not a.dry_run:
        os.makedirs(os.path.join(a.root, info["dir"]), exist_ok=True)
        with open(os.path.join(a.root, info["dir"], f"arrange_{a.dataset}.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source", "target"])
            w.writerows(log)
    print(f"{a.dataset}: {sum(len(v) for v in found.values())} files -> {len(found)}/{len(classes)} classes"
          f"{' (dry run)' if a.dry_run else ''} | unmatched {len(unmatched)} | ambiguous {len(ambiguous)}")
    print(f"{'class':45s} {'placed':>7s} {'reference':>9s}")
    for c in sorted(classes):
        n, ref = len(found.get(c, [])), info["classes"][c]
        print(f"{c:45s} {n:7d} {ref:9d}{'' if n == ref else '   <-- differs (see 99_documents/DATASETS.md)'}")
    for rel in unmatched[:50]:
        print("  unmatched:", rel)
    for rel, cs in ambiguous[:50]:
        print("  ambiguous:", rel, "->", cs)
    if unmatched or ambiguous:
        print("Place the files above by hand into <root>/<dataset dir>/01_pcap/<class>/ (any file name), then run step 3.")


if __name__ == "__main__":
    main()
