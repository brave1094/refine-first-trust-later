#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illusion 5 — TreeSHAP feature importance for RF / XGBoost (multiprocessing).

   Output: 03_model/param/{variant}/{model}/{ds}/exp{e}/treeshap.csv  (raw)
         99_documents/results/analysis/05_feat_importance/by_dataset/{ds}/{model}_treeshap.csv
         (feature, exp1_refined, exp3_noise, delta)

   Run (recommended — use 24 of 32 threads, leave 8 for GPU training):
     python3 04_analysis/05b_treeshap.py --workers 12 --threads 2
   Options:
     --models rf xgboost  --datasets vpn16 …  --cap 500  --batch 100
     --max-trees 0        RF tree cap (0 = unlimited, when memory allows)
     --workers 1          sequential run (for debugging)

   Design notes
     · Unit of work = (model, dataset). One worker handles exp1·exp3 together so
       features.csv is read only once. 20 jobs = 2 models × 10 datasets.
     · Jobs are assigned largest model file first (LPT) — if large RF models (~1GB)
       start late, the total wait time grows.
     · The shap TreeExplainer uses OpenMP internally, so processes × threads multiply.
       OMP_NUM_THREADS is bound to --threads per worker to avoid core contention.
     · On out-of-memory (MemoryError), the tree count is halved and retried automatically.

   Caution — lessons from earlier failures
     · features.csv still contains columns excluded from training (srcip·dstip·srcport·dstport·
       protocol). Must be aligned to the feature list and order used by the model.
     · SHAP output axis order differs by version and model. Find the axis matching the feature count
       and keep only that axis (do not mistake the class count for features).
     · NaN/Inf in the input turns all results into NaN → replace with 0 beforehand.
     · If results look abnormal, do not write the file (bad values left behind get skipped, which is more dangerous).
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import os
import sys

# thread counts must be fixed before importing numpy/shap to actually take effect.
_T = "2"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _T = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import csv                       # noqa: E402
import gc                        # noqa: E402
import time                      # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = str(RP.ROOT)
DSR = os.path.join(WORK, "01_dataset")
PARAM = str(RP.PARAM)
BYDS = os.path.join(str(RP.ANALYSIS_OUT), "05_feat_importance", "by_dataset")
DSS = ["vpn16", "tor16", "iot23", "cic18", 
       "tls1.3", "cispec", "ustc16", "cic17"]
META = {"Label", "filename", "Stream_num", "label", "session_id"}


def load_X(ds, model, variant, split, cap, seed=42):
    p = os.path.join(DSR, ds, variant, model, split, "features.csv")
    if not os.path.exists(p):
        return None, None
    with open(p, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        hdr = next(rd)
        keep = [i for i, c in enumerate(hdr) if c not in META]
        names = [hdr[i] for i in keep]
        rows = []
        for r in rd:
            if len(r) <= keep[-1]:
                continue
            vals = []
            for i in keep:
                t = r[i].strip()
                try:
                    vals.append(float(t) if t else 0.0)
                except ValueError:
                    vals.append(0.0)
            rows.append(vals)
    if not rows:
        return None, None
    X = np.nan_to_num(np.asarray(rows, dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if cap and len(X) > cap:
        idx = np.random.default_rng(seed).permutation(len(X))[:cap]
        X = X[idx]
    return X, names


def model_features(ck, model):
    """Feature list and order actually used by the model in training."""
    if model == "rf":
        fi = os.path.join(ck, "feature_importance.csv")
        if os.path.exists(fi):
            return [r["feature"] for r in csv.DictReader(open(fi, encoding="utf-8-sig"))]
        return None
    mj = os.path.join(ck, "model.json")
    if os.path.exists(mj):
        import json
        return json.load(open(mj, encoding="utf-8"))["learner"].get("feature_names")
    return None


def align(X, names, mfeat):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in mfeat if f not in pos]
    if miss:
        raise ValueError(f"{len(miss)} model features missing from features.csv: {miss[:5]}")
    return X[:, [pos[f] for f in mfeat]], list(mfeat)


def reduce_sv(sv, nfeat):
    a = np.abs(np.asarray(sv, dtype=np.float64))
    cand = [i for i, s in enumerate(a.shape) if s == nfeat]
    if not cand:
        raise ValueError(f"feature axis not found shape={a.shape} nfeat={nfeat}")
    fax = cand[-1]
    other = tuple(i for i in range(a.ndim) if i != fax)
    return a.mean(axis=other) if other else a


def run_rf(ck, X, names, batch, max_trees):
    import joblib
    import shap
    m = joblib.load(os.path.join(ck, "best_model.joblib"))
    n0 = len(getattr(m, "estimators_", []))
    used = n0
    if max_trees and n0 > max_trees:
        m.estimators_ = m.estimators_[:max_trees]
        m.n_estimators = used = max_trees
    ex = shap.TreeExplainer(m, feature_perturbation="tree_path_dependent")
    acc, n = np.zeros(len(names)), 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        acc += reduce_sv(ex.shap_values(xb, check_additivity=False), len(names)) * len(xb)
        n += len(xb)
        gc.collect()
    del m, ex
    gc.collect()
    return acc / max(n, 1), f"trees {used}/{n0}"


def run_xgb(ck, X, names, batch):
    """XGBoost native TreeSHAP (pred_contribs=True). Does not use the shap library —
       because shap 0.49 cannot parse XGBoost 3.x multiclass vector-leaf trees.
       pred_contribs is the exact TreeSHAP computed directly by XGBoost C++."""
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(os.path.join(ck, "model.json"))
    acc, n = np.zeros(len(names)), 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        d = xgb.DMatrix(xb, feature_names=names)
        c = np.abs(b.predict(d, pred_contribs=True))
        # binary: (B, F+1) · multiclass: (B, C, F+1). The last column is bias → dropped.
        if c.ndim == 3:
            c = c.mean(axis=1)                     # class mean → (B, F+1)
        acc += c[:, :len(names)].sum(axis=0)       # exclude bias column
        n += len(xb)
        gc.collect()
    del b
    gc.collect()
    return acc / max(n, 1), "native TreeSHAP"


def valid(path, nfeat):
    if not os.path.exists(path):
        return False
    try:
        rs = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        vs = [float(r["mean_abs_shap"]) for r in rs]
    except (OSError, ValueError, KeyError):
        return False
    # feature count must match exactly — old unaligned results (1210 rows)
    # passed a loose criterion (0.9x) and were skipped, leaving invalid results.
    return len(rs) == nfeat and not any(np.isnan(vs)) and sum(vs) > 0


def merge_by_dataset(model, ds, variant, exps):
    got = {}
    for e in exps:
        p = os.path.join(PARAM, variant, model, ds, f"exp{e}", "treeshap.csv")
        if not os.path.exists(p):
            return None
        d = {}
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            try:
                d[r["feature"]] = float(r["mean_abs_shap"])
            except (ValueError, KeyError):
                pass
        s = sum(d.values())
        got[e] = {k: v / s for k, v in d.items()} if s > 0 else d
    if len(got) < 2:
        return None
    od = os.path.join(BYDS, ds)
    os.makedirs(od, exist_ok=True)
    a, b = got[exps[0]], got[exps[1]]
    op = os.path.join(od, f"{model}_treeshap.csv")
    with open(op, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", "exp1_refined", "exp3_noisy", "delta"])
        for k in sorted(set(a) | set(b), key=lambda k: -b.get(k, 0)):
            x, y = a.get(k, 0.0), b.get(k, 0.0)
            w.writerow([k, f"{x:.8g}", f"{y:.8g}", f"{y - x:+.8g}"])
    return op


CFG = {}


def job(task):
    """Unit of work = (model, dataset). Handles exp1·exp3 together."""
    m, ds = task
    t00 = time.time()
    log = []
    X, names = load_X(ds, m, CFG["variant"], CFG["split"], CFG["cap"])
    if X is None:
        return f"[missing] {m}/{ds} features.csv"
    for e in CFG["exps"]:
        ck = os.path.join(PARAM, CFG["variant"], m, ds, f"exp{e}")
        out = os.path.join(ck, "treeshap.csv")
        mfeat = model_features(ck, m)
        if not mfeat:
            log.append(f"  {m}/{ds}/exp{e} [missing] model feature list")
            continue
        try:
            Xe, names_e = align(X, names, mfeat)
        except ValueError as ve:
            log.append(f"  {m}/{ds}/exp{e} [failed] {ve}")
            continue
        if valid(out, len(names_e)):
            log.append(f"  {m}/{ds}/exp{e} [skip] valid result exists")
            continue
        if os.path.exists(out):
            os.remove(out)
        t0 = time.time()
        mt = CFG["max_trees"]
        for attempt in range(3):                    # on out-of-memory, retry with fewer trees
            try:
                v, note = (run_rf(ck, Xe, names_e, CFG["batch"], mt) if m == "rf"
                           else run_xgb(ck, Xe, names_e, CFG["batch"]))
                break
            except MemoryError:
                mt = (mt // 2) if mt else 200
                log.append(f"  {m}/{ds}/exp{e} [retry] out of memory → trees {mt}")
                gc.collect()
            except Exception as ex:                 # noqa: BLE001
                log.append(f"  {m}/{ds}/exp{e} [failed] {type(ex).__name__}: {ex}")
                v = None
                break
        else:
            v = None
        if v is None:
            continue
        if len(v) != len(names_e) or np.isnan(v).any() or v.sum() <= 0:
            log.append(f"  {m}/{ds}/exp{e} [failed] abnormal result len={len(v)}")
            continue
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["feature", "mean_abs_shap"])
            for n_, x in zip(names_e, v):
                w.writerow([n_, f"{float(x):.8g}"])
        log.append(f"  {m}/{ds}/exp{e} [saved] {time.time()-t0:5.0f}s  n={len(Xe)} "
                   f"features={len(names_e)}  {note}  top={names_e[int(np.argmax(v))]}")
    op = merge_by_dataset(m, ds, CFG["variant"], CFG["exps"])
    head = f"══ {m}/{ds}  ({time.time()-t00:.0f}s)"
    if op:
        log.append(f"  └ per-dataset save complete")
    return head + "\n" + "\n".join(log)


def init(cfg):
    CFG.update(cfg)


def size_of(m, ds, variant):
    tot = 0
    for e in (1, 3):
        d = os.path.join(PARAM, variant, m, ds, f"exp{e}")
        for fn in ("best_model.joblib", "model.json"):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                tot += os.path.getsize(p)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgboost"])
    ap.add_argument("--datasets", nargs="+", default=DSS)
    ap.add_argument("--exps", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--variant", default="sizectrl")
    ap.add_argument("--split", default="test_denoised")
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--max-trees", type=int, default=0, dest="max_trees",
                    help="RF tree cap (0 = unlimited)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2,
                    help="OpenMP threads per worker (workers × threads ≈ available cores)")
    args = ap.parse_args()

    tasks = [(m, ds) for m in args.models for ds in args.datasets]
    tasks.sort(key=lambda t: -size_of(t[0], t[1], args.variant))   # largest first
    cfg = dict(variant=args.variant, split=args.split, cap=args.cap,
               batch=args.batch, max_trees=args.max_trees, exps=args.exps)
    print(f"jobs {len(tasks)} · workers {args.workers} × threads {args.threads} "
          f"= {args.workers * args.threads} cores · cap {args.cap} · "
          f"max_trees {args.max_trees or 'unlimited'}", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        init(cfg)
        for t in tasks:
            print(job(t), flush=True)
    else:
        with Pool(args.workers, initializer=init, initargs=(cfg,)) as pool:
            for res in pool.imap_unordered(job, tasks):
                print(res, flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
