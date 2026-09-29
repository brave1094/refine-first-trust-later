# -*- coding: utf-8 -*-
"""Illusion 4 (evaluation objectivity collapse) output generation — VPN16/TOR16/CISPEC/TLS1.3.
   Noise definition = filelist difference (test_noisy − test_denoised). No label CSV needed.
   Output: 99_documents/results/analysis/04_fail_to_eval/a4_eval_sizectrl/{csv/{ds}.csv, _summary.csv}
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os, re, csv
import pandas as pd

WORK   = os.environ.get("SCIE_WORK", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))   # repository root
DSROOT = os.path.join(WORK, "01_dataset")
WRONG  = str(RP.WRONG_LIST)
RT     = str(RP.RESULT_TABLES)
OUT    = os.path.join(str(RP.ANALYSIS_OUT), "04_fail_to_eval", "a4_eval_sizectrl")
MODELS = ["rf", "xgboost", "2dcnn", "etbert", "netmamba", "trafficformer", "yatc"]
DSS = ["vpn16", "tor16", "cispec", "tls1.3"]

def _num(s):
    m = re.search(r"\d+", str(s)); return m.group(0) if m else ""

def noise_keys(ds):
    fl = os.path.join(DSROOT, ds, "00_filelist")
    n = pd.read_csv(os.path.join(fl, "list_test_noisy.csv"), dtype=str, na_filter=False, encoding="utf-8-sig")
    d = pd.read_csv(os.path.join(fl, "list_test_denoised.csv"), dtype=str, na_filter=False, encoding="utf-8-sig")
    n.columns = [c.strip().lstrip("\ufeff") for c in n.columns]
    d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]
    nk = set(zip(n["filename"], n["session_id"]))
    dk = set(zip(d["filename"], d["session_id"]))
    noise = nk - dk
    return dict(noise=noise, total=nk,
                noise_num=set((f, _num(s)) for f, s in noise),
                total_num=set((f, _num(s)) for f, s in nk),
                n_total=len(nk), n_noise=len(noise))

# load result_table → acc[(model,ds)] = (exp1,exp2)
acc = {}
for m in MODELS:
    p = os.path.join(RT, f"{m}.csv")
    if not os.path.exists(p): continue
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        acc[(m, r["dataset"])] = (float(r["exp1"]), float(r["exp2"]))

os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
summary = []
for ds in DSS:
    k = noise_keys(ds)
    base = k["n_noise"] / max(k["n_total"], 1)
    rows = []
    ratios, drops = [], []
    for m in MODELS:
        p = os.path.join(WRONG, m, ds, "exp2_wrong.csv")
        if not os.path.exists(p) or (m, ds) not in acc:
            continue
        w = pd.read_csv(p, dtype=str, na_filter=False)
        nn = len(w)
        hit = 0
        for fn, sid in zip(w["filename"], w["session_id"]):
            if re.search("[a-zA-Z]", sid):
                hit += (fn, sid) in k["noise"]
            else:
                hit += (fn, _num(sid)) in k["noise_num"]
        ratio = hit / nn if nn else 0.0
        e1, e2 = acc[(m, ds)]
        drop = (e1 - e2) * 100
        rows.append([m, f"{e1*100:.2f}", f"{e2*100:.2f}", f"{drop:+.2f}",
                     nn, hit, f"{ratio*100:.1f}"])
        ratios.append(ratio); drops.append(drop)
    # save the dataset CSV
    with open(os.path.join(OUT, "csv", f"{ds}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "exp1_refined_test(%)", "exp2_noisy_test(%)",
                     "e1-e2(%p,eval_distortion)", "exp2_n_wrong", "n_noise_in_wrong", "noise_rate_in_wrong(%)"])
        wr.writerows(rows)
    mean_ratio = sum(ratios)/len(ratios) if ratios else 0
    lift = mean_ratio/base if base > 0 else float("nan")
    mean_drop = sum(drops)/len(drops) if drops else 0
    if base == 0:
        verdict = "control(noise0)"
    elif lift >= 1.3:
        verdict = "strong(overrepresented)"
    elif lift >= 1.05:
        verdict = "weak(base_effect)"
    else:
        verdict = "neutral"
    summary.append([ds, k["n_total"], k["n_noise"], f"{base*100:.1f}",
                    f"{mean_ratio*100:.1f}", f"{lift:.2f}" if base>0 else "-",
                    f"{mean_drop:+.2f}", verdict])
    print(f"{ds:<8} base {base*100:4.1f}%  wrong_noise {mean_ratio*100:4.1f}%  Lift {lift:.2f}  e1-e2 {mean_drop:+.1f}%p  → {verdict}")

with open(os.path.join(OUT, "_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.writer(f)
    wr.writerow(["dataset", "n_test_sessions", "n_noise_sessions", "base_noise_rate(%)",
                 "noise_rate_in_wrong_mean(%)", "Lift_overrep_ratio", "e1-e2_mean(%p)", "verdict"])
    wr.writerows(summary)
print(f"\n[done] → {OUT}")
