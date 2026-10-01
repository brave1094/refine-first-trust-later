#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a4_exp2_noise.py — Illusion 4 (evaluation objectivity collapse) key evidence.
  Noise definition = filelist difference (list_test_noisy − list_test_denoised). No label CSV needed.
  Computes the share of noise sessions among exp2 (refined training→noisy test) errors + Lift over the base rate.
  Run: python3 a4_exp2_noise.py  →  a4_exp2_noise.csv + console
  ※ Uses relative paths, so it works from any location of the repository.
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import re
import csv
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
DSROOT = str(RP.DATASET)
VARIANT = os.environ.get("SCIE_VARIANT", "sizectrl")  # the article uses sizectrl
WRONG  = str(RP.WRONG_LIST / VARIANT)
RT     = str(RP.RESULT_TABLES / VARIANT)
LISTS  = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}[VARIANT]
MODELS = ["rf", "xgboost", "2dcnn", "etbert", "netmamba", "trafficformer", "yatc"]
DSS = ["vpn16", "tor16", "cispec", "tls1.3",
       "ustc16", "cic17", "cic18", "iot23"]


def _num(s):
    m = re.search(r"\d+", str(s))
    return m.group(0) if m else ""


def noise_keys(ds):
    """noise = test_noisy − test_denoised (zero-leakage split guarantees denoised ⊆ noisy)."""
    fl = os.path.join(DSROOT, ds, LISTS)
    fn_n = os.path.join(fl, "list_test_noisy.csv")
    fn_d = os.path.join(fl, "list_test_denoised.csv")
    if not (os.path.exists(fn_n) and os.path.exists(fn_d)):
        return None
    n = pd.read_csv(fn_n, dtype=str, na_filter=False, encoding="utf-8-sig")
    d = pd.read_csv(fn_d, dtype=str, na_filter=False, encoding="utf-8-sig")
    n.columns = [c.strip().lstrip("﻿") for c in n.columns]
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    nk = set(zip(n["filename"], n["session_id"]))
    dk = set(zip(d["filename"], d["session_id"]))
    noise = nk - dk
    return dict(noise=noise, total=nk,
                noise_num=set((f, _num(s)) for f, s in noise),
                total_num=set((f, _num(s)) for f, s in nk),
                n_total=len(nk), n_noise=len(noise))


def load_acc():
    acc = {}
    for m in MODELS:
        p = os.path.join(RT, f"{m}.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            acc[(m, r["dataset"])] = (float(r["exp1"]), float(r["exp2"]))
    return acc


def main():
    acc = load_acc()
    rows, summ = [], []
    for ds in DSS:
        k = noise_keys(ds)
        if k is None:
            print(f"[skip] {ds}: filelist missing"); continue
        base = k["n_noise"] / max(k["n_total"], 1)
        print(f"[{ds}] test {k['n_total']} / noise {k['n_noise']} (base {base:.1%})")
        ratios = []
        for m in MODELS:
            p = os.path.join(WRONG, m, ds, "exp2_wrong.csv")
            if not os.path.exists(p):
                continue
            w = pd.read_csv(p, dtype=str, na_filter=False)
            nn = len(w)
            if nn == 0:
                continue
            hit = matched = 0
            for fn, sid in zip(w["filename"], w["session_id"]):
                if re.search("[a-zA-Z]", sid):        # DL: string as-is
                    matched += (fn, sid) in k["total"]
                    hit += (fn, sid) in k["noise"]
                else:                                  # tree: (filename, number)
                    matched += (fn, _num(sid)) in k["total_num"]
                    hit += (fn, _num(sid)) in k["noise_num"]
            ratio = hit / nn
            ratios.append(ratio)
            e = acc.get((m, ds))
            drop = (e[0] - e[1]) * 100 if e else float("nan")
            rows.append([m, ds, nn, hit, round(ratio, 4), round(base, 4), round(drop, 2)])
            flag = "" if matched == nn else f"  !key mismatch {nn-matched}/{nn}"
            print(f"  {m:<14} {ds:<9} wrong {nn:<6} noise {hit:<6} = {ratio:5.1%}{flag}")
        if ratios:
            mr = sum(ratios) / len(ratios)
            lift = mr / base if base > 0 else float("nan")
            summ.append([ds, k["n_total"], k["n_noise"], round(base, 4),
                         round(mr, 4), round(lift, 2) if base > 0 else "-"])
    with open(os.path.join(HERE, "a4_exp2_noise.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "dataset", "exp2_wrong", "noise_in_wrong",
                     "noise_ratio", "base_rate", "e1_e2_drop(%p)"])
        wr.writerows(rows)
    print("\n=== per-dataset summary (base / noise-in-errors / Lift) ===")
    for s in summ:
        print(f"  {s[0]:<9} base {s[3]:.1%}  noise-in-errors {s[4]:.1%}  Lift {s[5]}")
    print(f"\n[a4] → a4_exp2_noise.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
