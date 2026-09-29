#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a7_field_agg.py — Illusion 5-B: aggregate DL attributions (a6_dlshap*) from byte→field, compare exp1 vs exp3 from 3 perspectives.

Input:
  shap_dl/{variant}/{model}/{ds}/exp{1,3}_attr.npy          (a6 output, (n,L) |IG|)
  01_dataset/{ds}/{variant}/{model}/test_denoised/field_id.npy · pkt_id.npy
                                                             (08_make_field_map output, files.csv row order)
  01_dataset/{ds}/{variant}/field_vocab.csv

3 perspectives (all exp1 (refined training) vs exp3 (noisy training), same test_denoised input):
  ① field   : per-field attribution mass (normalized within session, then averaged) → top-25 table + Spearman rankρ
              + artifact mass (ip.id·checksum·ttl·dsfield·flags·frag — same definition as a5 TreeSHAP)
  ② packet  : mass per pkt_id (1..5, padding 0) → which packets noisy training concentrates on
  ③ layer   : L3/L4 header vs application (tls/dns/http/…) vs masked (#masked) vs padding — mass distribution

Output: 99_documents/results/analysis/06_dl_shap/{variant}/
  {model}_{ds}_field.csv / _packet.csv / _layer.csv   (exp1·exp3·Δ per perspective)
  _summary.csv  (per model·ds: rankρ, artifact mass e1/e3/Δ, top1 field, layer distribution)

Run (on the host — no GPU·container needed, numpy only):
  python3 a7_field_agg.py                       # everything in shap_dl
  python3 a7_field_agg.py --models 2dcnn,yatc --datasets vpn16,tor16 --variant sizectrl
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import re
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
DATA = os.path.join(WORK, "01_dataset")
MODELS = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
DSS = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]

# artifact (capture-dependent) fields — consistent with the a5_treeshap ARTIFACT definition (raw header identifiers)
ART = re.compile(r"(ip\.id|checksum|ip\.ttl|dsfield|ip\.flags|frag|"
                 r"\.srcport|\.dstport|ip\.src|ip\.dst|ipv6\.src|ipv6\.dst)", re.I)
L34 = re.compile(r"^(ip|ipv6|tcp|udp|icmp|igmp|arp)(\.|$)", re.I)
SPECIAL = {"<pad>", "<sep>"}


def load_vocab(ds, variant):
    p = os.path.join(DATA, ds, variant, "field_vocab.csv")
    id2name = {}
    with open(p, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) >= 2:
                id2name[int(row[0])] = row[1]
    return id2name


def layer_of(name):
    if name in SPECIAL:
        return "pad/sep"
    if name.endswith("#masked"):
        return "masked(ip/port)"
    if L34.match(name):
        return "L3/L4 header"
    return "app/payload"


def field_mass(attr, fid, id2name):
    """Normalize per session (sum=1), then average field mass → {field_name: mass}."""
    tot = attr.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    w = attr / tot
    mass = defaultdict(float)
    for f in np.unique(fid):
        mass[id2name.get(int(f), f"?{f}")] += float(w[fid == f].sum())
    n = attr.shape[0]
    return {k: v / n for k, v in mass.items()}


def packet_mass(attr, pkt):
    tot = attr.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    w = attr / tot
    out = {}
    for p in range(0, int(pkt.max()) + 1):
        out[p] = float(w[pkt == p].sum()) / attr.shape[0]
    return out


def spearman(a, b):
    """Rank correlation over the union of keys (filled with 0) — pure numpy."""
    keys = sorted(set(a) | set(b))
    x = np.array([a.get(k, 0.0) for k in keys])
    y = np.array([b.get(k, 0.0) for k in keys])
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def process(model, ds, variant, outd):
    sd = os.path.join(str(RP.SHAP_DL), variant, model, ds)
    a1p, a3p = os.path.join(sd, "exp1_attr.npy"), os.path.join(sd, "exp3_attr.npy")
    fdir = os.path.join(DATA, ds, variant, model, "test_denoised")
    fidp, pktp = os.path.join(fdir, "field_id.npy"), os.path.join(fdir, "pkt_id.npy")
    vocp = os.path.join(DATA, ds, variant, "field_vocab.csv")
    for p, what in ((a1p, "exp1_attr"), (a3p, "exp3_attr"),
                    (fidp, "field_id"), (pktp, "pkt_id"),
                    (vocp, "field_vocab(08 incomplete)")):
        if not os.path.exists(p):
            print(f"  [skip] {model}/{ds} — {what} missing")
            return None
    a1, a3 = np.load(a1p), np.load(a3p)
    # field_id/pkt_id cover the whole split (thousands to tens of thousands of rows), but only the 500 rows sampled by a6 are used.
    # Open with mmap and read only the needed rows (tens of times faster on a network mount).
    fid = np.load(fidp, mmap_mode="r")
    pkt = np.load(pktp, mmap_mode="r")
    # 08 stores 2dcnn(n,28,28)/yatc·netmamba(n,40,40) as images → flatten the position axes
    fid = fid.reshape(len(fid), -1)
    pkt = pkt.reshape(len(pkt), -1)
    # a6 stratified sample → align field_id (files.csv row order) to the same rows via rows_idx.npy
    rip = os.path.join(sd, "rows_idx.npy")
    if os.path.exists(rip):
        ridx = np.load(rip)
        if ridx.max() >= len(fid):
            print(f"  [ERROR] {model}/{ds} rows_idx out of range "
                  f"(max {ridx.max()} ≥ field_id {len(fid)} rows)")
            return None
        fid, pkt = np.asarray(fid[ridx]), np.asarray(pkt[ridx])
    else:
        fid, pkt = np.asarray(fid[:len(a1)]), np.asarray(pkt[:len(a1)])
    # UER models: tokenization prepends [CLS] and truncates to seq → attr[0]=[CLS],
    # attr[i]=bigram i−1. field_id starts at bigram 0.., so align with a one-position shift.
    if model in ("etbert", "trafficformer"):
        a1, a3 = a1[:, 1:], a3[:, 1:]
        fid, pkt = fid[:, :a1.shape[1]], pkt[:, :a1.shape[1]]
    if a1.shape[1] != fid.shape[1]:
        print(f"  [ERROR] {model}/{ds} length mismatch attr L={a1.shape[1]} vs "
              f"field_id L={fid.shape[1]} — check the 08 layout")
        return None
    n = min(len(a1), len(a3), len(fid))
    a1, a3, fid, pkt = a1[:n], a3[:n], fid[:n], pkt[:n]
    # sessions unmatched in 08 (field_id all 0=<pad>) have no labels and cannot be aggregated → excluded
    keep = fid.max(axis=1) > 0
    drop = int((~keep).sum())
    if drop:
        a1, a3, fid, pkt = a1[keep], a3[keep], fid[keep], pkt[keep]
        n = len(fid)
        print(f"  [warn] {model}/{ds} excluded {drop} unmatched sessions (truncated pcaps etc.)")
    id2name = load_vocab(ds, variant)

    # ① field
    m1, m3 = field_mass(a1, fid, id2name), field_mass(a3, fid, id2name)
    rho = spearman(m1, m3)
    keys = sorted(set(m1) | set(m3), key=lambda k: -m3.get(k, 0.0))
    with open(os.path.join(outd, f"{model}_{ds}_field.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["field", "layer", "artifact", "exp1_refined", "exp3_noisy", "delta"])
        for k in keys:
            w.writerow([k, layer_of(k), int(bool(ART.search(k))),
                        f"{m1.get(k,0):.5f}", f"{m3.get(k,0):.5f}",
                        f"{m3.get(k,0)-m1.get(k,0):+.5f}"])
    art1 = sum(v for k, v in m1.items() if ART.search(k))
    art3 = sum(v for k, v in m3.items() if ART.search(k))

    # ② packet
    p1, p3 = packet_mass(a1, pkt), packet_mass(a3, pkt)
    with open(os.path.join(outd, f"{model}_{ds}_packet.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["packet", "exp1_refined", "exp3_noisy", "delta"])
        for p in sorted(set(p1) | set(p3)):
            lab = "pad" if p == 0 else f"pkt{p}"
            w.writerow([lab, f"{p1.get(p,0):.5f}", f"{p3.get(p,0):.5f}",
                        f"{p3.get(p,0)-p1.get(p,0):+.5f}"])

    # ③ layer
    l1, l3 = defaultdict(float), defaultdict(float)
    for k, v in m1.items():
        l1[layer_of(k)] += v
    for k, v in m3.items():
        l3[layer_of(k)] += v
    with open(os.path.join(outd, f"{model}_{ds}_layer.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["layer", "exp1_refined", "exp3_noisy", "delta"])
        for k in ["L3/L4 header", "app/payload", "masked(ip/port)", "pad/sep"]:
            w.writerow([k, f"{l1.get(k,0):.5f}", f"{l3.get(k,0):.5f}",
                        f"{l3.get(k,0)-l1.get(k,0):+.5f}"])

    top1_1 = max(m1, key=m1.get) if m1 else ""
    top1_3 = max(m3, key=m3.get) if m3 else ""
    print(f"  [OK] {model}/{ds}  n={n}  rankρ={rho:.3f}  "
          f"artifact mass {art1*100:.1f}%→{art3*100:.1f}% ({(art3-art1)*100:+.1f}%p)")
    return [model, ds, n, f"{rho:.3f}",
            f"{art1*100:.1f}", f"{art3*100:.1f}", f"{(art3-art1)*100:+.1f}",
            top1_1, top1_3,
            f"{l1.get('L3/L4 header',0)*100:.1f}", f"{l3.get('L3/L4 header',0)*100:.1f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DSS))
    ap.add_argument("--variant", default="sizectrl",
                    choices=["full", "sizectrl", "strat"])
    args = ap.parse_args()
    outd = os.path.join(str(RP.ANALYSIS_OUT), "06_dl_shap", args.variant)
    os.makedirs(outd, exist_ok=True)
    summ = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            r = process(m, ds, args.variant, outd)
            if r:
                summ.append(r)
    if summ:
        with open(os.path.join(outd, "_summary.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["model", "dataset", "n", "rank_rho",
                        "artifact_mass_exp1_%", "artifact_mass_exp3_%",
                        "artifact_delta_%p", "top1_field_exp1", "top1_field_exp3",
                        "L34_mass_exp1_%", "L34_mass_exp3_%"])
            w.writerows(summ)
        print(f"\n[saved] {outd}/_summary.csv  ({len(summ)} rows)")


if __name__ == "__main__":
    main()
