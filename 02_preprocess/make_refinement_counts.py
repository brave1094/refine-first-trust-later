#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_refinement_counts.py  ─  exact per-class and per-rule refinement counts of each dataset (streaming; Tables 8-16)
────────────────────────────────────────────────────────────────────────
Input : <dataset root>/{dir}/04_session_noisy_labeled/*.csv   (marked session CSVs; Aa_1..Df_1 = "0"/"1")
        02_preprocess/noise_rule/NN_noise_rule_{ds}.py  (default_clean_rules)
Output: 99_documents/results/refinement_counts/
       - perclass_{tag}.csv    : group, class, before, after, removed, vanished
       - summary.csv           : tag, is_app, n_class, before, after, refinement_rate(%), n_vanished
       - perrule_app.csv       : rule, desc, vpn16, tor16, tls1.3, cispec  (sessions removed)
       - perrule_attack.csv    : rule, desc, ustc16, cic17, cic18, iot23

  class = task3.  benign/attack group = (task1.lower()=="benign" -> normal, else attack).  application datasets: group=app.
  removed = sessions with 1 in any of default_clean_rules (= the sessions dropped from the refined data).
  per-rule count = sessions with 1 in that rule column (rules may overlap; default rules only).

Note: iot23 step-4 output is 341 GB (1,156 files); run it on a server. Chunked streaming keeps memory low.
   usage:  python3 make_refinement_counts.py             # all eight datasets
           python3 make_refinement_counts.py vpn16 cic17 # a subset
           (the heavy iot23/cic18 separately:  python3 make_refinement_counts.py iot23)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os, sys, re, ast, glob, csv, time
from collections import defaultdict
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────
DATA8T = str(RP.DATASETS)
ROOT   = str(RP.ROOT)
NRDIR  = os.path.join(ROOT, "02_preprocess", "noise_rule")
OUTDIR = str(RP.REFINEMENT_COUNTS)
os.makedirs(OUTDIR, exist_ok=True)
CHUNK  = 500_000

# ── datasets: tag -> (dataset dir, noise_rule file, is_app) ───────────────
DS = [
    # application datasets
    ("vpn16",  "21_ISCX-VPN-2016",        "21_noise_rule_vpn16.py",  True),
    ("tor16",  "22_ISCX-TOR-2016",        "22_noise_rule_tor16.py",  True),
    ("tls1.3", "23_CSTNET_TLS1.3",        "23_noise_rule_tls1.3.py", True),
    ("cispec", "24_CipherSpectrum",       "24_noise_rule_cispec.py", True),
    # attack datasets
    ("ustc16", "01_USTC-TFC_2016",        "01_noise_rule_ustc16.py", False),
    ("cic17",  "02_CIC-IDS-2017",         "02_noise_rule_cic17.py",  False),
    ("cic18",  "03_CIC-IDS-2018",         "03_noise_rule_cic18.py",  False),
    ("iot23",  "04_CIC_IoT_Dataset_2023", "04_noise_rule_iot23.py",  False),
]
APP_ORDER    = ["vpn16","tor16","tls1.3","cispec"]
ATTACK_ORDER = ["ustc16","cic17","cic18","iot23"]

# ── rule names (the 'desc' column), as in Supplementary Material S1 ─────────
RULE_DESC = {
    "Aa_1": "Incomplete TCP 3-way handshake",
    "Aa_2": "Capture damage or truncation",
    "Aa_3": "No FIN in either direction",
    "Aa_4": "Malformed packets",
    "Aa_5": "Missing TLS Hello",
    "Aa_6": "Unlabeled or doubly labeled session",
    "Aa_7": "IPv6 session",
    "Bb_1": "Infrastructure management protocols (common, class-aware)",
    "Bb_2": "P2P background (common, class-aware)",
    "Bb_3": "Class-protocol mismatch (common)",
    "Bb_4": "SNI-domain label mismatch (common)",
    "Bb_5": "Attack signature under a benign label (common)",
    "Bb_6": "Gateway VPN beacon in an attack label (common)",
    "Bb_7": "Operating-system traffic of the infected VM (ustc16)",
    "Bb_8": "Infiltration class (cic17)",
    "Bb_9": "FTP-Patator fingerprint under a benign label (cic17)",
    "Bb_10": "SSH-Patator tool fingerprint under a benign label (cic17)",
    "Bb_11": "HULK URI under a benign label (cic17)",
    "Bb_12": "GoldenEye URI under a benign label (cic17)",
    "Bb_13": "Attacker address under a benign label (cic18)",
    "Bb_14": "Reconnaissance and scanner traffic under a benign label (cic18)",
    "Bb_15": "FTP-Patator without an attack payload (cic18)",
    "Bb_16": "Slowhttptest without an attack payload (cic18)",
    "Bb_17": "SSH-Patator misfire to port 21 (cic18)",
    "Bb_18": "LOIC-UDP processed as ICMP (cic18)",
    "Bb_19": "Empty Slowloris flow (cic18)",
    "Bb_20": "Early-reset GoldenEye attempt (cic18)",
    "Bb_21": "Page resources in web attacks (cic18)",
    "Bb_22": "Empty Bot flow (cic18)",
    "Bb_23": "Internal NMAP scan under a benign label (cic18)",
    "Bb_24": "Session with an Internet endpoint in an attack capture (iot23)",
    "Bb_25": "Internal infrastructure protocols in an attack capture (iot23)",
    "Bb_26": "Broadcast, multicast, or link-local destination in an attack capture (iot23)",
    "Bb_27": "nmap UDP probe under a benign label",
    "Bb_28": "SYN-only scan in a configured benign class",
    "Bb_29": "SYN-only sweep in a configured benign class",
    "Cc_1": "Layer-3 control and IPv6",
    "Cc_2": "Name resolution and address assignment",
    "Cc_3": "Windows network services",
    "Cc_4": "Infrastructure management",
    "Cc_5": "Broadcast, multicast, and link-local destinations",
    "Cd_1": "P2P DHT and tracker maintenance",
    "Cd_2": "Windows background traffic",
    "Cd_3": "Microsoft update delivery",
    "Cd_4": "Package-manager updates",
    "De_1": "Google shared infrastructure and SSO",
    "De_2": "Mail SSO and CDN",
    "De_3": "Advertising and tracking",
    "De_4": "Browser infrastructure",
    "Df_1": "Browser start page",
}
ALL_CODES = ["Aa_1","Aa_2","Aa_3","Aa_4","Aa_5","Aa_6","Aa_7"] + \
            ["Bb_%d"%i for i in range(1,30)] + ["Cc_1","Cc_2","Cc_3","Cc_4","Cc_5"] + \
            ["Cd_1","Cd_2","Cd_3","Cd_4"] + ["De_1","De_2","De_3","De_4","Df_1"]

def get_default_rules(nrfile):
    s = open(os.path.join(NRDIR, nrfile), encoding="utf-8").read()
    s = re.sub(r"#.*", "", s)                    # drop comments first (brackets inside comments)
    m = re.search(r"default_clean_rules\s*=\s*\[", s)   # the assignment, not a mention in a docstring
    j = m.end() - 1; depth = 0
    for k in range(j, len(s)):
        if s[k] == "[": depth += 1
        elif s[k] == "]":
            depth -= 1
            if depth == 0: end = k+1; break
    return ast.literal_eval(s[j:end])

TRUE = {"1","True","TRUE","true"}

def process(tag, subdir, nrfile, is_app):
    rules = get_default_rules(nrfile)
    rules = [r for r in rules if r in ALL_CODES]
    csvs = sorted(glob.glob(os.path.join(DATA8T, subdir, "04_session_noisy_labeled", "*.csv")))
    if not csvs:
        print(f"[!] {tag}: no CSV ({subdir})"); return None
    cols = ["task1","task3"] + rules
    before = defaultdict(int); after = defaultdict(int)     # key=(group,cls)
    perrule = defaultdict(int)                              # rule -> removed count
    t0 = time.time(); nrow = 0
    for fi, f in enumerate(csvs, 1):
        for ch in pd.read_csv(f, usecols=cols, dtype=str, keep_default_na=False, chunksize=CHUNK):
            nrow += len(ch)
            mark = ch[rules].isin(TRUE)                     # bool DataFrame
            removed = mark.any(axis=1)
            if is_app:
                grp = "app"
                g = pd.Series(grp, index=ch.index)
            else:
                g = (ch["task1"].str.lower() == "benign").map({True:"normal", False:"attack"})
            key = g + "\t" + ch["task3"]
            for k, c in key.value_counts().items():   before[k] += int(c)
            for k, c in key[~removed].value_counts().items(): after[k] += int(c)
            for r in rules: perrule[r] += int(mark[r].sum())
        print(f"    {tag}: {fi}/{len(csvs)} files, {nrow:,} rows, {time.time()-t0:.0f}s", flush=True)
    # per-class counts
    rows = []
    for k in before:
        grp, cls = k.split("\t", 1)
        bf = before[k]; af = after.get(k, 0)
        rows.append([grp, cls, bf, af, bf-af, int(af == 0 and bf > 0)])
    # order: app = class a-z / attack = normal (a-z) -> attack (a-z)
    if is_app:
        rows.sort(key=lambda r: r[1].lower())
    else:
        rows.sort(key=lambda r: (0 if r[0]=="normal" else 1, r[1].lower()))
    with open(os.path.join(OUTDIR, f"perclass_{tag}.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp); w.writerow(["group","class","before","after","removed","vanished"]); w.writerows(rows)
    bt = sum(before.values()); at = sum(after.values())
    nvan = sum(r[5] for r in rows)
    print(f"[✓] {tag}: classes {len(rows)} · before {bt:,} · after {at:,} · refinement {100*(bt-at)/bt:.1f}% · vanished {nvan}")
    return dict(tag=tag, is_app=is_app, n_class=len(rows), before=bt, after=at,
                rate=100*(bt-at)/bt if bt else 0, nvan=nvan, rules=rules, perrule=dict(perrule))

def main():
    sel = [a for a in sys.argv[1:]]
    todo = [d for d in DS if (not sel or d[0] in sel)]
    results = {}
    for tag, subdir, nrfile, is_app in todo:
        print(f"── {tag} ({subdir}) ──", flush=True)
        r = process(tag, subdir, nrfile, is_app)
        if r: results[tag] = r
    # summary
    sp = os.path.join(OUTDIR, "summary.csv"); head = not os.path.exists(sp)
    with open(sp, "a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if head: w.writerow(["tag","is_app","n_class","before","after","refinement_rate(%)","n_vanished","rules"])
        for tag in results:
            r = results[tag]
            w.writerow([tag, r["is_app"], r["n_class"], r["before"], r["after"],
                        f"{r['rate']:.2f}", r["nvan"], " ".join(r["rules"])])
    # per-rule table (repeated runs keep earlier values and update only the datasets of this run)
    def write_perrule(order, fname):
        path = os.path.join(OUTDIR, fname)
        cell = defaultdict(dict)                       # cell[rule][tag] = value
        if os.path.exists(path):                       # load earlier values
            with open(path, encoding="utf-8-sig") as fp:
                for row in csv.DictReader(fp):
                    for tag in order:
                        if row.get(tag, "—") not in ("", "—"): cell[row["rule"]][tag] = row[tag]
        for tag in order:                              # update the datasets of this run
            if tag not in results: continue
            r = results[tag]
            for rc in r["rules"]:
                cell[rc][tag] = r["perrule"].get(rc, 0)
        rules_sorted = [c for c in ALL_CODES if c in cell]
        with open(path, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.writer(fp); w.writerow(["rule","desc"]+order)
            for rc in rules_sorted:
                w.writerow([rc, RULE_DESC.get(rc,"")] + [cell[rc].get(tag,"—") for tag in order])
        print(f"[✓] {fname} updated")
    write_perrule(APP_ORDER, "perrule_app.csv")
    write_perrule(ATTACK_ORDER, "perrule_attack.csv")
    print("\ndone ->", OUTDIR)

if __name__ == "__main__":
    main()
