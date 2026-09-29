#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_stats.py — multi-seed (5 seeds) result aggregation: quantify the error range of Δ=exp1−exp3.
  ★ variant support: aggregates sizectrl·strat·full separately. (default: all 3)

Reads: result_table/{variant}/{model}.csv                       (seed 42)
       result_table/{variant}_seed{1,7,2024,31337}/{model}.csv  (extra seeds)

Outputs (99_documents/results/analysis/08_seed_stats/{variant}/):
  seed_delta_long.csv   : model, dataset, seed, exp1, exp3, delta_pp  (raw data)
  seed_stats_cell.csv   : per (model, dataset) cell  n_seed, mean Δ, seed SD, min, max
  seed_stats_ds.csv     : per dataset  grand mean Δ, within-seed SD (model average), between-model SD (SD of seed means),
                          95%CI, sign test (per model +/−), verdict
Console: the summary above + missing audit (which cells·seeds are still missing).

Verdict logic (pre-declared):
  within-seed SD < between-model SD → differences between models exceed seed variation = structural sensitivity differences are real
  |mean Δ| ≤ 2×within-seed SD → Δ of that cell is within run-to-run error (no effect claimed)

Usage: python3 seed_stats.py                 # all of sizectrl·strat·full
       python3 seed_stats.py --variant strat # just one
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import csv
import math
import argparse
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RT = str(RP.RESULT_TABLES)
OUT_BASE = str(RP.ANALYSIS_OUT / "08_seed_stats")

VARIANTS = ["sizectrl", "strat", "full"]
SEEDS = [42, 1, 7, 2024, 31337]
MODELS = ["rf", "xgboost", "2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
DS = ["vpn16", "tor16", "tls1.3", "cispec", 
      "ustc16", "cic17", "cic18", "iot23"]
DL = {"2dcnn", "etbert", "yatc", "netmamba", "trafficformer"}
# tls1.3·cispec DL training is now done for all seeds → all datasets are multi-seed targets.
DL_DS = set(DS)
# two-sided 97.5% quantile of the t distribution (df=1..9) — for small-sample CI
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def vdir(variant, seed):
    return variant if seed == 42 else f"{variant}_seed{seed}"


def load(variant):
    """{(model, ds, seed): (exp1, exp3)}"""
    D = {}
    for s in SEEDS:
        for m in MODELS:
            p = os.path.join(RT, vdir(variant, s), f"{m}.csv")
            if not os.path.exists(p):
                continue
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                try:
                    D[(m, r["dataset"], s)] = (float(r["exp1"]), float(r["exp3"]))
                except (ValueError, KeyError):
                    pass
    return D


def expected(m, ds):
    """Whether this cell is a multi-seed target (seed 42 covers all cells)."""
    if m in ("rf", "xgboost"):
        return True
    return m in DL and ds in DL_DS


def main(variant):
    OUT = os.path.join(OUT_BASE, variant)
    os.makedirs(OUT, exist_ok=True)
    print(f"\n{'='*72}\n [{variant}] multi-seed Δ=exp1−exp3 aggregation\n{'='*72}")
    D = load(variant)

    # ── raw data (long) ──
    rows_long = []
    for (m, ds, s), (e1, e3) in sorted(D.items()):
        rows_long.append([m, ds, s, f"{e1:.4f}", f"{e3:.4f}", f"{(e1-e3)*100:.2f}"])
    with open(os.path.join(OUT, "seed_delta_long.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "seed", "exp1", "exp3", "delta_pp"])
        w.writerows(rows_long)

    # ── per-cell statistics ──
    cell = {}
    rows_cell = []
    for m in MODELS:
        for ds in DS:
            v = [(D[(m, ds, s)][0] - D[(m, ds, s)][1]) * 100
                 for s in SEEDS if (m, ds, s) in D]
            if not v:
                continue
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            cell[(m, ds)] = (len(v), st.mean(v), sd, min(v), max(v))
            rows_cell.append([m, ds, len(v), f"{st.mean(v):.2f}",
                              f"{sd:.2f}" if len(v) > 1 else "",
                              f"{min(v):.2f}", f"{max(v):.2f}"])
    with open(os.path.join(OUT, "seed_stats_cell.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "n_seed", "delta_mean_pp",
                    "seed_sd_pp", "min_pp", "max_pp"])
        w.writerows(rows_cell)

    # ── per-dataset statistics ──
    print(f"{'ds':9}{'n':>4}{'meanΔ':>8}{'seedSD':>9}{'modelSD':>9}"
          f"{'95%CI':>16}{'sign(+/−)':>10}   verdict")
    rows_ds = []
    for ds in DS:
        means, wsds, alldelta = [], [], []
        pos = neg = 0
        for m in MODELS:
            c = cell.get((m, ds))
            if not c:
                continue
            n, mu, sd, lo, hi = c
            means.append(mu)
            alldelta += [(D[(m, ds, s)][0] - D[(m, ds, s)][1]) * 100
                         for s in SEEDS if (m, ds, s) in D]
            if n > 1 and not math.isnan(sd):
                wsds.append(sd)
            if mu > 0:
                pos += 1
            elif mu < 0:
                neg += 1
        if not means:
            continue
        gmu = st.mean(alldelta)
        wsd = st.mean(wsds) if wsds else float("nan")          # within-seed (run) variation
        bsd = st.stdev(means) if len(means) > 1 else float("nan")  # between-model variation
        n = len(alldelta)
        se = st.stdev(alldelta) / math.sqrt(n) if n > 1 else float("nan")
        t = T975.get(n - 1, 1.96)
        ci = (gmu - t * se, gmu + t * se)
        if not math.isnan(wsd) and abs(gmu) <= 2 * wsd:
            j = "within error (≈0)"
        elif ci[0] > 0:
            j = "significant effect (+)"
        elif ci[1] < 0:
            j = "significant effect (−)"
        else:
            j = "inconclusive (CI includes 0)"
        print(f"{ds:9}{n:4d}{gmu:8.2f}{wsd:9.2f}{bsd:9.2f}"
              f"  [{ci[0]:6.2f},{ci[1]:6.2f}]{pos:>6}/{neg:<3}   {j}")
        rows_ds.append([ds, n, f"{gmu:.2f}", f"{wsd:.2f}", f"{bsd:.2f}",
                        f"{ci[0]:.2f}", f"{ci[1]:.2f}", pos, neg, j])
    with open(os.path.join(OUT, "seed_stats_ds.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "delta_mean_pp", "within_seed_sd_pp",
                    "between_model_sd_pp", "ci_lo", "ci_hi",
                    "n_model_pos", "n_model_neg", "verdict"])
        w.writerows(rows_ds)

    # ── missing audit ──
    miss = []
    for m in MODELS:
        for ds in DS:
            for s in SEEDS:
                if expected(m, ds) and (m, ds, s) not in D:
                    miss.append(f"{m}/{ds}/seed{s}")
    got = sum(1 for m in MODELS for ds in DS for s in SEEDS
              if expected(m, ds) and (m, ds, s) in D)
    tot = sum(1 for m in MODELS for ds in DS for s in SEEDS if expected(m, ds))
    print(f"\n[progress] multi-seed cells {got}/{tot} filled"
          + (f"  — {len(miss)} missing" if miss else "  — complete ✅"))
    if miss:
        print("  " + ", ".join(miss[:40]) + (" …" if len(miss) > 40 else ""))
    print(f"[saved] {OUT}/seed_delta_long.csv · seed_stats_cell.csv · seed_stats_ds.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    help="sizectrl|strat|full|all (comma-separated allowed). default all")
    a = ap.parse_args()
    vs = VARIANTS if a.variant == "all" else [x.strip() for x in a.variant.split(",")]
    for v in vs:
        main(v)
