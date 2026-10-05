#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cl_dl.py — builds the DL training sets of the CL comparison (subsets of the sizectrl training sample).

Conditions (variant name = 01_dataset/{ds}/{variant}/{model}/)
  cl_rule          : the m rule-noise sessions removed                        (vpn16, tor16, iot23)
  cl_budget        : the m sessions with the lowest CL quality score removed (same number)   (vpn16, tor16, iot23)
  cl_random_s{N}   : m sessions removed at random, a different set for every seed N          (vpn16, tor16, iot23)
  cl_default       : the set CL chooses itself removed                       (cic17: 0 rule-noise sessions → control)
  The removal sets are the same as in cl_compare.py (ML), {ds}_cl_flags.csv (the random ones too: same seeds, same order).
Only the training set is subset; test_denoised and test_noisy are shared with sizectrl through relative symbolic links.
Finished directories get a .done marker → skipped on a re-run.
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import json, os, shutil, sys
import numpy as np
import pandas as pd

DS = str(RP.DATASET)
OUT = str(RP.ANALYSIS_OUT / "09_cl_compare")
MODELS = ["2dcnn", "etbert", "yatc", "netmamba", "trafficformer"]
SEEDS = [42, 1, 7, 2024, 31337]
PLAN = {"vpn16": "noise", "tor16": "noise", "iot23": "noise", "cic17": "control"}


def removal_sets(ds):
    fl = pd.read_csv(os.path.join(OUT, f"{ds}_cl_flags.csv"), keep_default_na=False)
    keys = fl["key"].to_numpy()
    R, B, C = fl["rule_noise"].to_numpy() == 1, fl["cl_budget"].to_numpy() == 1, fl["cl_issue"].to_numpy() == 1
    m = int(R.sum())
    sets = {}
    if PLAN[ds] == "noise":
        sets["cl_rule"], sets["cl_budget"] = set(keys[R]), set(keys[B])
        for s in SEEDS:   # same procedure as stage_run of cl_compare.py
            idx = np.random.default_rng(s).choice(len(keys), m, replace=False)
            sets[f"cl_random_s{s}"] = set(keys[idx])
    else:
        sets["cl_default"] = set(keys[C])
    return set(keys), sets


def subset(src, dst, drop, allkeys):
    f = pd.read_csv(os.path.join(src, "files.csv"), dtype=str, keep_default_na=False)
    k = (f["filename"] + "\t" + f["session_id"]).to_numpy()
    cover = np.mean([x in allkeys for x in k])
    keep = np.array([x not in drop for x in k])
    os.makedirs(dst, exist_ok=True)
    f[keep].to_csv(os.path.join(dst, "files.csv"), index=False)
    if os.path.exists(os.path.join(src, "x_data.npy")):
        for n in ("x_data.npy", "y_data.npy"):
            a = np.load(os.path.join(src, n))
            assert len(a) == len(f), (src, n, len(a), len(f))
            np.save(os.path.join(dst, n), a[keep])
    else:
        lines = open(os.path.join(src, "data.tsv"), encoding="utf-8").read().split("\n")
        head, body = lines[0], [l for l in lines[1:] if l != ""]
        assert len(body) == len(f), (src, len(body), len(f))
        with open(os.path.join(dst, "data.tsv"), "w", encoding="utf-8") as w:
            w.write("\n".join([head] + [l for l, kk in zip(body, keep) if kk]) + "\n")
    return len(f), int(keep.sum()), cover


def main():
    for ds in PLAN:
        allkeys, sets = removal_sets(ds)
        for v, drop in sets.items():
            for m in MODELS:
                base = os.path.join(DS, ds, v, m)
                if os.path.exists(os.path.join(base, ".done")):
                    continue
                src = os.path.join(DS, ds, "sizectrl", m)
                shutil.rmtree(base, ignore_errors=True)
                os.makedirs(base)
                n0, n1, cover = subset(os.path.join(src, "train_noisy"), os.path.join(base, "train_noisy"), drop, allkeys)
                for sp in ("test_denoised", "test_noisy"):
                    os.symlink(os.path.join("..", "..", "sizectrl", m, sp), os.path.join(base, sp))
                if os.path.exists(os.path.join(src, "dataset_config.json")):
                    shutil.copy(os.path.join(src, "dataset_config.json"), base)
                open(os.path.join(base, ".done"), "w").write(f"{n0}->{n1}\n")
                print(f"[build] {ds:6} {v:16} {m:13} train {n0:,} → {n1:,} (removed {n0 - n1:,}, target {len(drop):,}) "
                      f"· share of ML sample keys {100 * cover:.1f}%", flush=True)
                if cover < 0.98:
                    sys.exit(f"[ERROR] {ds} {m}: the DL sample differs from the ML sample (share {cover:.3f}) → stopped")
    print("[build] done", flush=True)


if __name__ == "__main__":
    main()
