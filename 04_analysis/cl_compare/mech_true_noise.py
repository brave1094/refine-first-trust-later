#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mech_true_noise.py — recomputes the error composition (Fig. 7, Table D.5) with the 'actual noise decision'.

Background: 04_analysis/a4_exp2_noise.py defines the noise sessions as the set difference
list_test_noisy − list_test_denoised. The two lists are drawn independently, so the difference greatly overcounts
the noise (ustc16: 26 actual vs. 2,495 in the difference).
Here, with the same method as cl_compare (default_clean_rules − Aa_7 marking in 04_session_noisy_labeled), every
session of the sizectrl noisy test set (xgboost test_noisy feature file = the set actually evaluated) is marked as
noise or not, joined to the Exp2/Exp4 wrong lists of the seven models (wrong_lists/sizectrl, seed 42) and counted again.

Outputs (99_documents/results/analysis/09_cl_compare/)
  keys/{ds}_test_noise.csv        per-session decision (filename, session_id, rules)
  mech_true_noise_long.csv        dataset × model: n_test, w2, n2, w4, n4, gain, gain_noise, ambiguous
  mech_true_noise.csv             mean of the seven models per dataset (Table D.5 layout) + actual / set-difference noise counts
Run: python3 mech_true_noise.py --procs 16
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, csv, glob, os, re, statistics as st
from multiprocessing import Pool
import pandas as pd
import cl_compare as C

WL = str(RP.WRONG_LIST / "sizectrl")
MODELS = ["rf", "xgboost", "2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
ORDER = ["vpn16", "tor16", "iot23", "cic18", "tls1.3", "cispec", "ustc16", "cic17"]


def wrong_keys(p):
    df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return list(zip(df["filename"], df["session_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ORDER)
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.join(C.OUT, "keys"), exist_ok=True)
    longp, sump = os.path.join(C.OUT, "mech_true_noise_long.csv"), os.path.join(C.OUT, "mech_true_noise.csv")
    with open(longp, "w", newline="", encoding="utf-8") as fl, open(sump, "w", newline="", encoding="utf-8") as fs:
        wl, ws = csv.writer(fl), csv.writer(fs)
        wl.writerow(["dataset", "model", "n_test", "w2", "n2", "w4", "n4", "gain", "gain_noise", "ambiguous"])
        ws.writerow(["dataset", "n_test", "noise_true", "noise_setdiff", "models",
                     "acc2", "w2", "real2", "real2_pct", "n2", "n2_pct",
                     "acc4", "w4", "real4", "real4_pct", "n4", "n4_pct", "gain", "gain_noise_pct"])
        for ds in a.datasets:
            keys = C.feature_keys(ds, "test_noisy")
            need = set(keys)
            R = C.rules_of(ds)
            files = sorted(glob.glob(os.path.join(C.D8, C.MAP[ds][0], "04_session_noisy_labeled", "*.csv")))
            C.log(f"[mech] {ds}: test sessions {len(need):,} · rules {len(R)} · 04 CSVs {len(files)}")
            hit, seen = {}, set()
            with Pool(min(a.procs, max(1, len(files))), initializer=C._kinit, initargs=(R, need)) as pool:
                for i, (h, s) in enumerate(pool.imap_unordered(C._kwork, files), 1):
                    hit.update(h); seen.update(s)
                    if i % 50 == 0 or i == len(files):
                        C.log(f"   {i}/{len(files)}  noise {len(hit):,}")
            with open(os.path.join(C.OUT, "keys", f"{ds}_test_noise.csv"), "w", newline="", encoding="utf-8") as fk:
                w = csv.writer(fk); w.writerow(["filename", "session_id", "rules"])
                for k, r in sorted(hit.items()):
                    w.writerow([k[0], k[1], r])
            # lookup: exact match on (filename, 'tcp_12'); numeric-only session_id (RF, XGB) as (filename, 12)
            exact = {k: (k in hit) for k in keys}
            bynum = {}
            for fn, sid in keys:
                bynum.setdefault((fn, re.sub(r"\D", "", sid)), set()).add((fn, sid) in hit)

            def flag(fn, sid):
                if (fn, sid) in exact:
                    return exact[(fn, sid)], False
                s = bynum.get((fn, re.sub(r"\D", "", sid)))
                if not s:
                    return False, True
                return any(s), len(s) > 1
            # set-difference noise count (for comparison)
            fld = os.path.join(C.DSROOT, ds, "00_filelist_sm")
            sd = ""
            try:
                ln = pd.read_csv(os.path.join(fld, "list_test_noisy.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
                ld = pd.read_csv(os.path.join(fld, "list_test_denoised.csv"), dtype=str, keep_default_na=False, encoding="utf-8-sig")
                sd = len(set(zip(ln["filename"], ln["session_id"])) - set(zip(ld["filename"], ld["session_id"])))
            except Exception as e:
                C.log(f"   set difference failed: {e}")
            rows = []
            for m in MODELS:
                p2, p4 = os.path.join(WL, m, ds, "exp2_wrong.csv"), os.path.join(WL, m, ds, "exp4_wrong.csv")
                if not (os.path.exists(p2) and os.path.exists(p4)):
                    C.log(f"   {m}: no wrong lists"); continue
                w2, w4 = wrong_keys(p2), wrong_keys(p4)
                f2 = [flag(*k) for k in w2]; f4 = [flag(*k) for k in w4]
                s2, s4 = set(w2), set(w4)
                gain = [k for k in w2 if k not in s4]
                gn = sum(flag(*k)[0] for k in gain)
                amb = sum(x[1] for x in f2 + f4)
                r = dict(m=m, w2=len(w2), n2=sum(x[0] for x in f2), w4=len(w4), n4=sum(x[0] for x in f4),
                         gain=len(gain), gn=gn, amb=amb)
                rows.append(r)
                wl.writerow([ds, m, len(keys), r["w2"], r["n2"], r["w4"], r["n4"], r["gain"], gn, amb]); fl.flush()
            if not rows:
                continue
            n = len(keys)
            mw2, mn2 = st.mean(r["w2"] for r in rows), st.mean(r["n2"] for r in rows)
            mw4, mn4 = st.mean(r["w4"] for r in rows), st.mean(r["n4"] for r in rows)
            acc2 = st.mean(1 - r["w2"] / n for r in rows) * 100
            acc4 = st.mean(1 - r["w4"] / n for r in rows) * 100
            g = sum(r["gain"] for r in rows); gn = sum(r["gn"] for r in rows)
            ws.writerow([ds, n, len(hit), sd, len(rows),
                         round(acc2, 2), round(mw2, 1), round(mw2 - mn2, 1), round(100 * (mw2 - mn2) / mw2, 2) if mw2 else "",
                         round(mn2, 1), round(100 * mn2 / mw2, 2) if mw2 else "",
                         round(acc4, 2), round(mw4, 1), round(mw4 - mn4, 1), round(100 * (mw4 - mn4) / mw4, 2) if mw4 else "",
                         round(mn4, 1), round(100 * mn4 / mw4, 2) if mw4 else "",
                         g, round(100 * gn / g, 2) if g else ""]); fs.flush()
            C.log(f"[mech] {ds}: actual noise {len(hit)} / set difference {sd} · E2 errors {mw2:.1f} (noise {mn2:.1f}) · E4 {mw4:.1f} ({mn4:.1f})")
    C.log(f"[mech] → {sump}")


if __name__ == "__main__":
    main()
