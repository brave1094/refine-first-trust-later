#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cl_compare.py — rule refinement (50 rules) vs. Confident Learning (CL, cleanlab) (lightweight version, CPU only).

Question: with an automatic label-noise detector (CL), are different sessions removed than by the rules, and does the
conclusion change?
Target: the sizectrl training sample (list_train_noisy, the Exp3 training set of Section 4) — the eight public datasets.

Stages
  keys : scan 04_session_noisy_labeled and record the rules that fire on each session of the sizectrl sample
         (rules = default_clean_rules − Aa_7, the same rule set used to compute the sampled noise rates of the article)
  cl   : out-of-sample probabilities on the training sample from 5-fold cross-validated XGBoost → cleanlab
         · cl_default = find_label_issues(filter_by='prune_by_noise_rate')   (the number CL chooses itself)
         · rank by quality score (self_confidence) → select as many as the rules remove, m (budget)
  run  : 5 training conditions × {xgboost, rf} × 5 seeds → accuracy on the refined test set and on the noisy test set
         raw (= Exp3 training set) / rule (rule noise removed) / cl_budget (CL top m removed)
         / random (m removed at random) / cl_default (CL default removal)
         rule, cl_budget and random remove the same number from the same sample, so only 'what was removed' differs
         (paired comparison).
  sum  : overlap (precision, recall, per-category recall) and accuracy summary

Leakage control: early stopping uses only a 10% validation split inside the training set (the test sets are never
used). RF uses its defaults (300 trees).
Parallelism: --workers × --threads ≤ number of CPUs (default 4 × 4 = 16 cores). Results are appended after every job →
a re-run after an interruption resumes where it stopped.

Examples:
  python3 cl_compare.py keys --procs 16                 # 1) collect rule decisions (iot23 takes longest: 1,156 CSVs in 04)
  python3 cl_compare.py cl  run sum --workers 4 --threads 4
  python3 cl_compare.py cl run sum --datasets vpn16 tor16   # a subset only
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse, ast, csv, glob, os, re, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = str(RP.ROOT)
D8 = str(RP.DATASETS)
DSROOT = str(RP.DATASET)
NR = str(RP.PREPROCESS / "noise_rule")
OUT = str(RP.ANALYSIS_OUT / "09_cl_compare")
KEYS = os.path.join(OUT, "keys")

MAP = {  # alias: (dataset directory under NM_DATASET_ROOT, rule file)
    "vpn16": ("21_ISCX-VPN-2016", "21_noise_rule_vpn16.py"),
    "tor16": ("22_ISCX-TOR-2016", "22_noise_rule_tor16.py"),
    "tls1.3": ("23_CSTNET_TLS1.3", "23_noise_rule_tls1.3.py"),
    "cispec": ("24_CipherSpectrum", "24_noise_rule_cispec.py"),
    "ustc16": ("01_USTC-TFC_2016", "01_noise_rule_ustc16.py"),
    "cic17": ("02_CIC-IDS-2017", "02_noise_rule_cic17.py"),
    "cic18": ("03_CIC-IDS-2018", "03_noise_rule_cic18.py"),
    "iot23": ("04_CIC_IoT_Dataset_2023", "04_noise_rule_iot23.py"),
}
SEEDS = [42, 1, 7, 2024, 31337]
MODELS = ["xgboost", "rf"]
CONDS = ["raw", "rule", "cl_budget", "random", "cl_default"]
CATS = ["Aa", "Bb", "Cc", "Cd", "De", "Df"]
TRUE = {"1", "True", "TRUE", "true"}
COMMON_COLS = ["Label", "filename", "Stream_num", "protocol", "srcip", "srcport", "dstip", "dstport"]
PROTO = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmpv6"}
XGB = dict(eta=0.1, max_depth=8, subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
           gamma=0.0, alpha=0.0, **{"lambda": 1.0})  # same values as DEFAULTS in 03_model/model/train_xgboost.py
ROUNDS, PATIENCE = 500, 50


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ───────────────────────────── keys ─────────────────────────────
def rules_of(ds):
    """default_clean_rules − Aa_7 (the rule set used for the sampled noise rates of the article)."""
    s = re.sub(r"#.*", "", open(os.path.join(NR, MAP[ds][1]), encoding="utf-8").read())
    m = re.search(r"default_clean_rules\s*=\s*\[", s)
    j, d = m.end() - 1, 0
    for k in range(j, len(s)):
        if s[k] == "[":
            d += 1
        elif s[k] == "]":
            d -= 1
            if d == 0:
                end = k + 1
                break
    return [r for r in ast.literal_eval(s[j:end]) if r != "Aa_7"]


def train_list(ds):
    return pd.read_csv(os.path.join(DSROOT, ds, "00_filelist_sm", "list_train_noisy.csv"),
                       dtype=str, keep_default_na=False, encoding="utf-8-sig")


_K = {}


def _kinit(R, need):
    _K["R"], _K["need"] = R, need


def _kwork(fp):
    R, need, hit, seen = _K["R"], _K["need"], {}, set()
    for ch in pd.read_csv(fp, usecols=["filename", "session_id"] + R, dtype=str,
                          keep_default_na=False, chunksize=400_000):
        seen.update(k for k in zip(ch["filename"], ch["session_id"]) if k in need)
        mk = ch[R].isin(TRUE)
        rows = mk.any(axis=1)
        if not rows.any():
            continue
        sub, mks = ch.loc[rows], mk.loc[rows]
        for (fn, sid), flags in zip(zip(sub["filename"], sub["session_id"]), mks.to_numpy()):
            if (fn, sid) in need:
                hit[(fn, sid)] = ";".join(r for r, f in zip(R, flags) if f)
    return hit, seen


SPLITS = ["train_noisy", "train_denoised", "test_noisy", "test_denoised"]


def feature_keys(ds, split):
    """Session keys (filename, 'proto_stream') of the feature CSV (sizectrl/xgboost) actually used for training and evaluation."""
    p = os.path.join(DSROOT, ds, "sizectrl", "xgboost", split, "features.csv")
    df = pd.read_csv(p, usecols=["filename", "Stream_num", "protocol"], low_memory=False)
    pr = pd.to_numeric(df["protocol"], errors="coerce").fillna(-1).astype(int)
    st = pd.to_numeric(df["Stream_num"], errors="coerce").fillna(-1).astype(int)
    return [(fn, f"{PROTO.get(q, str(q))}_{s}") for fn, q, s in zip(df["filename"].astype(str), pr, st)]


def stage_keys(dss, procs):
    """Record the rules that fire on every session of the four feature-CSV splits and check the noise rate of the actual sample."""
    os.makedirs(KEYS, exist_ok=True)
    for ds in dss:
        outp = os.path.join(KEYS, f"{ds}_train_noise.csv")
        if os.path.exists(outp):
            log(f"[keys] {ds}: already exists → skipped ({outp})")
            continue
        R = rules_of(ds)
        fk = {sp: feature_keys(ds, sp) for sp in SPLITS}
        need = set(k for sp in SPLITS for k in fk[sp])
        files = sorted(glob.glob(os.path.join(D8, MAP[ds][0], "04_session_noisy_labeled", "*.csv")))
        log(f"[keys] {ds}: sample {len(need):,} · rules {len(R)} · 04 CSVs {len(files)}")
        hit, seen = {}, set()
        with Pool(min(procs, max(1, len(files))), initializer=_kinit, initargs=(R, need)) as pool:
            for i, (h, s) in enumerate(pool.imap_unordered(_kwork, files), 1):
                hit.update(h)
                seen |= s
                if i % 50 == 0 or i == len(files):
                    log(f"   {i}/{len(files)}  noise {len(hit):,}")
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "session_id", "rules"])
            for (fn, sid), r in sorted(hit.items()):
                w.writerow([fn, sid, r])
        # noise rate of the actual sample: feature CSV (the sample used for training) vs. the current file list
        # (00_filelist_sm, the basis of the article's noise rates)
        chk = os.path.join(OUT, "sample_noise_check.csv")
        new = not os.path.exists(chk)
        with open(chk, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["dataset", "split", "n_features", "found_in_04", "noise_n", "noise_pct", "n_filelist",
                            "overlap_with_filelist"])
            for sp in SPLITS:
                ks = fk[sp]
                nn = sum(k in hit for k in ks)
                lp = os.path.join(DSROOT, ds, "00_filelist_sm", f"list_{sp}.csv")
                lk = set()
                if os.path.exists(lp):
                    l = pd.read_csv(lp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                    lk = set(zip(l["filename"], l["session_id"]))
                ov = sum(k in lk for k in ks) / max(1, len(ks))
                fd = sum(k in seen for k in ks)
                w.writerow([ds, sp, len(ks), fd, nn, round(100 * nn / max(1, len(ks)), 2), len(lk), round(ov, 4)])
                log(f"   {ds:8} {sp:15} features {len(ks):>7,} (found in 04 {fd:,}) · noise {nn:>6,} ({100 * nn / max(1, len(ks)):6.2f}%)"
                    f" · file list {len(lk):>7,} · overlap {100 * ov:5.1f}%")
        log(f"[keys] {ds}: noise sessions {len(hit):,} → {outp}")


# ───────────────────────────── data ─────────────────────────────
def load_split(ds, split, label_map):
    """xgboost feature CSV (NaN kept). Returns: X, y, key arrays."""
    p = os.path.join(DSROOT, ds, "sizectrl", "xgboost", split, "features.csv")
    df = pd.read_csv(p, low_memory=False, na_values=["-"])
    y = df["Label"].astype(str).map(label_map)
    df, y = df[y.notna()].reset_index(drop=True), y[y.notna()].astype(np.int64).to_numpy()
    feat = [c for c in df.columns if c not in COMMON_COLS]
    X = df[feat].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    proto = pd.to_numeric(df["protocol"], errors="coerce").fillna(-1).astype(int)
    key = [f"{fn}\t{PROTO.get(p, str(p))}_{int(s)}" for fn, p, s in
           zip(df["filename"].astype(str), proto, pd.to_numeric(df["Stream_num"], errors="coerce").fillna(-1))]
    return X, y, np.array(key, dtype=object)


def noise_info(ds, keys_tr):
    """Rule-noise flag and category set of every row of the training sample."""
    kp = os.path.join(KEYS, f"{ds}_train_noise.csv")
    if not os.path.exists(kp):
        sys.exit(f"[ERROR] {kp} not found → run the keys stage first")
    kn = pd.read_csv(kp, dtype=str, keep_default_na=False)
    rmap = {f"{a}\t{b}": r for a, b, r in zip(kn["filename"], kn["session_id"], kn["rules"])}
    lst = train_list(ds)
    lkeys = set(f"{a}\t{b}" for a, b in zip(lst["filename"], lst["session_id"]))
    matched = np.mean([k in lkeys for k in keys_tr])
    rules = [rmap.get(k, "") for k in keys_tr]
    return np.array([r != "" for r in rules]), rules, matched


# ───────────────────────────── models ─────────────────────────────
def inner_split(y, seed, frac=0.1):
    """Validation split inside the training set (10% per class; classes with fewer than 5 samples stay entirely in training)."""
    rng = np.random.default_rng(seed)
    va = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) >= 5:
            va.extend(rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False))
    va = np.array(sorted(va), dtype=int)
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


def fit_predict(model, Xtr, ytr, Xtests, ncls, seed, nthread):
    if model == "xgboost":
        import xgboost as xgb
        p = dict(objective="multi:softprob", num_class=ncls, tree_method="hist", eval_metric="mlogloss",
                 seed=seed, nthread=nthread, **XGB)
        tr, va = inner_split(ytr, seed)
        dtr = xgb.DMatrix(Xtr[tr], label=ytr[tr])
        if len(va):
            b = xgb.train(p, dtr, ROUNDS, evals=[(xgb.DMatrix(Xtr[va], label=ytr[va]), "val")],
                          early_stopping_rounds=PATIENCE, verbose_eval=False)
            it = (0, b.best_iteration + 1)
        else:
            b = xgb.train(p, dtr, 200, verbose_eval=False)
            it = (0, 200)
        return [b.predict(xgb.DMatrix(X), iteration_range=it) for X in Xtests]
    from sklearn.ensemble import RandomForestClassifier
    f = lambda X: np.nan_to_num(X, nan=-1.0)
    m = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=nthread, random_state=seed)
    m.fit(f(Xtr), ytr)
    out = []
    for X in Xtests:
        pr = np.zeros((len(X), ncls), dtype=np.float32)
        pr[:, m.classes_] = m.predict_proba(f(X))
        out.append(pr)
    return out


# ───────────────────────────── cl ─────────────────────────────
def stage_cl(dss, threads):
    from sklearn.model_selection import StratifiedKFold
    try:
        from cleanlab.filter import find_label_issues
        from cleanlab.rank import get_label_quality_scores
    except ImportError:
        sys.exit("[ERROR] cleanlab is not installed → pip install cleanlab")
    import warnings
    warnings.filterwarnings("ignore")
    for ds in dss:
        outp = os.path.join(OUT, f"{ds}_cl_flags.csv")
        if os.path.exists(outp):
            log(f"[cl] {ds}: already exists → skipped")
            continue
        import json
        lm = json.load(open(os.path.join(DSROOT, ds, "00_filelist_sm", "label_map.json"), encoding="utf-8"))
        X, y, keys = load_split(ds, "train_noisy", lm)
        is_noise, rules, matched = noise_info(ds, keys)
        log(f"[cl] {ds}: training sample {len(y):,} · rule noise {is_noise.sum():,} ({100 * is_noise.mean():.2f}%) "
            f"· overlap with the current file list {100 * matched:.1f}% (for reference)")
        present = np.unique(y)                      # only the classes present in the sample are given to CL
        remap = {c: i for i, c in enumerate(present)}
        yl = np.array([remap[c] for c in y])
        ncls = len(present)
        probs = np.zeros((len(y), ncls), dtype=np.float32)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for k, (tr, te) in enumerate(skf.split(X, yl), 1):
            probs[te] = fit_predict("xgboost", X[tr], yl[tr], [X[te]], ncls, 42, threads)[0]
            log(f"   fold {k}/5")
        probs = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
        issue = find_label_issues(labels=yl, pred_probs=probs, filter_by="prune_by_noise_rate")
        q = get_label_quality_scores(labels=yl, pred_probs=probs, method="self_confidence")
        m = int(is_noise.sum())
        budget = np.zeros(len(y), bool)
        if m:
            budget[np.argsort(q, kind="stable")[:m]] = True
        pd.DataFrame({"key": keys, "label": y, "rule_noise": is_noise.astype(int), "rules": rules,
                      "cl_issue": issue.astype(int), "cl_budget": budget.astype(int),
                      "quality": q}).to_csv(outp, index=False)
        log(f"[cl] {ds}: CL default {issue.sum():,} ({100 * issue.mean():.2f}%) · budget m={m:,} → {outp}")


# ───────────────────────────── run ─────────────────────────────
_D = {}


def _prep(ds):
    """Read one dataset and build the keep mask of every condition (once, on first use inside a worker process)."""
    import json
    fl = pd.read_csv(os.path.join(OUT, f"{ds}_cl_flags.csv"), keep_default_na=False)
    lm = json.load(open(os.path.join(DSROOT, ds, "00_filelist_sm", "label_map.json"), encoding="utf-8"))
    X, y, keys = load_split(ds, "train_noisy", lm)
    assert list(keys) == list(fl["key"]), f"{ds}: cl_flags order mismatch"
    Xr, yr, _ = load_split(ds, "test_denoised", lm)
    Xn, yn, _ = load_split(ds, "test_noisy", lm)
    rn, cb, cd = (fl["rule_noise"].to_numpy() == 1, fl["cl_budget"].to_numpy() == 1, fl["cl_issue"].to_numpy() == 1)
    m = int(rn.sum())
    rand = {}
    for s in SEEDS:
        k = np.ones(len(y), bool)
        if m:
            k[np.random.default_rng(s).choice(len(y), m, replace=False)] = False
        rand[s] = k
    return dict(X=X, y=y, Xr=Xr, yr=yr, Xn=Xn, yn=yn, ncls=len(lm), random=rand,
                keep={"raw": np.ones(len(y), bool), "rule": ~rn, "cl_budget": ~cb, "cl_default": ~cd})


def _winit(threads):
    _D["threads"] = threads


def _job(job):
    ds, model, seed, cond = job
    if ds not in _D:
        _D[ds] = _prep(ds)
    d = _D[ds]
    keep = d["keep"][cond] if cond != "random" else d["random"][seed]
    t0 = time.time()
    p_ref, p_noisy = fit_predict(model, d["X"][keep], d["y"][keep], [d["Xr"], d["Xn"]], d["ncls"], seed, _D["threads"])
    from sklearn.metrics import accuracy_score, f1_score
    r = []
    for pr, yt in ((p_ref, d["yr"]), (p_noisy, d["yn"])):
        yp = pr.argmax(1)
        r += [accuracy_score(yt, yp), f1_score(yt, yp, average="macro")]
    return [ds, model, seed, cond, int(keep.sum())] + [round(v, 6) for v in r] + [round(time.time() - t0, 1)]


def stage_run(dss, workers, threads):
    """Worker processes are started with spawn. (Forking after the parent has already initialised XGBoost/OpenMP in the
    cl stage deadlocks OpenMP in the children.) Each worker process reads the data itself."""
    import multiprocessing as mp
    accp = os.path.join(OUT, "acc_long.csv")
    done = set()
    if os.path.exists(accp):
        for r in csv.DictReader(open(accp, encoding="utf-8")):
            done.add((r["dataset"], r["model"], int(r["seed"]), r["cond"]))
    else:
        with open(accp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["dataset", "model", "seed", "cond", "n_train", "acc_ref", "f1_ref",
                                    "acc_noisy", "f1_noisy", "sec"])
    jobs = []
    for ds in dss:
        fl = pd.read_csv(os.path.join(OUT, f"{ds}_cl_flags.csv"), usecols=["rule_noise", "cl_issue"])
        m, n = int((fl["rule_noise"] == 1).sum()), len(fl)
        # if the rules remove less than 0.1% (cispec and cic17: 0, tls1.3: 4), rule, cl_budget and random ≈ raw
        # → raw is used instead (saves computation)
        conds = CONDS if m >= max(1, 0.001 * n) else ["raw", "cl_default"]
        for model in MODELS:
            for s in SEEDS:
                for c in conds:
                    if (ds, model, s, c) not in done:
                        jobs.append((ds, model, s, c))
        log(f"[run] {ds}: train {n:,} (rule removal {m:,}, CL default removal {int((fl['cl_issue'] == 1).sum()):,})")
    log(f"[run] remaining jobs {len(jobs)} · workers {workers} × threads {threads} (spawn)")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_winit, initargs=(threads,)) as pool,             open(accp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i, r in enumerate(pool.imap(_job, jobs, chunksize=1), 1):
            w.writerow(r)
            f.flush()
            log(f"   {i}/{len(jobs)} {r[0]} {r[1]} s{r[2]} {r[3]:<10} acc_ref {r[5]:.4f} acc_noisy {r[7]:.4f} ({r[9]}s)")


# ───────────────────────────── sum ─────────────────────────────
def stage_sum(dss):
    from scipy import stats
    rows = []
    for ds in dss:
        p = os.path.join(OUT, f"{ds}_cl_flags.csv")
        if not os.path.exists(p):
            continue
        fl = pd.read_csv(p, keep_default_na=False)
        R, C, B = fl["rule_noise"] == 1, fl["cl_issue"] == 1, fl["cl_budget"] == 1
        n, nr, nc, nb = len(fl), int(R.sum()), int(C.sum()), int((R & C).sum())
        row = dict(dataset=ds, n_train=n, rule_n=nr, rule_pct=round(100 * nr / n, 2), cl_n=nc,
                   cl_pct=round(100 * nc / n, 2), both=nb,
                   cl_precision=round(nb / nc, 3) if nc else "", cl_recall=round(nb / nr, 3) if nr else "",
                   jaccard=round(nb / (nr + nc - nb), 3) if (nr + nc - nb) else "",
                   budget_hit=round(int((R & B).sum()) / nr, 3) if nr else "")
        for c in CATS:
            inc = fl["rules"].str.contains(rf"(^|;){c}_", regex=True)
            row[f"{c}_n"] = int(inc.sum())
            row[f"{c}_cl_recall"] = round(int((inc & C).sum()) / int(inc.sum()), 3) if inc.sum() else ""
        rows.append(row)
    ov = pd.DataFrame(rows)
    ov.to_csv(os.path.join(OUT, "overlap.csv"), index=False)
    log("[sum] overlap (CL default removal vs. rule removal)\n" + ov[["dataset", "n_train", "rule_pct", "cl_pct", "cl_precision",
                                                                    "cl_recall", "budget_hit"]].to_string(index=False))
    accp = os.path.join(OUT, "acc_long.csv")
    if not os.path.exists(accp):
        return
    a = pd.read_csv(accp)
    a = a[a["dataset"].isin(dss)]
    out = []
    for (ds, model), g in a.groupby(["dataset", "model"]):
        raw = g[g["cond"] == "raw"].set_index("seed")
        for cond in CONDS:
            h = g[g["cond"] == cond].set_index("seed")
            if h.empty and cond in ("rule", "cl_budget", "random"):
                h = raw                                          # m=0 → identical to raw
            if h.empty:
                continue
            s = h.index.intersection(raw.index)
            d = (h.loc[s, "acc_ref"] - raw.loc[s, "acc_ref"]) * 100
            out.append(dict(dataset=ds, model=model, cond=cond, n_seeds=len(h), n_train=int(h["n_train"].iloc[0]),
                            acc_ref=round(100 * h["acc_ref"].mean(), 2), acc_noisy=round(100 * h["acc_noisy"].mean(), 2),
                            d_vs_raw_pp=round(d.mean(), 2)))
        # paired comparison with the same number removed: rule − cl_budget, rule − random (refined test set, pp)
        for other in ("cl_budget", "random"):
            r_, o_ = g[g["cond"] == "rule"].set_index("seed"), g[g["cond"] == other].set_index("seed")
            s = r_.index.intersection(o_.index)
            if len(s) >= 2:
                dd = (r_.loc[s, "acc_ref"] - o_.loc[s, "acc_ref"]).to_numpy() * 100
                ci = stats.t.ppf(0.975, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
                out.append(dict(dataset=ds, model=model, cond=f"rule-minus-{other}", n_seeds=len(dd),
                                d_vs_raw_pp=round(dd.mean(), 2), ci95_pp=round(ci, 2)))
    sm = pd.DataFrame(out)
    sm.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    log("[sum] accuracy summary (refined test acc_ref, noisy test acc_noisy, Δ vs. raw)\n" + sm.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["keys", "cl", "run", "sum"])
    ap.add_argument("--datasets", nargs="*", default=list(MAP))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    bad = [d for d in a.datasets if d not in MAP]
    if bad:
        sys.exit(f"[ERROR] unknown datasets {bad}")
    os.makedirs(OUT, exist_ok=True)
    log(f"stages={a.stages} datasets={a.datasets} procs={a.procs} workers={a.workers} threads={a.threads}")
    if "keys" in a.stages:
        stage_keys(a.datasets, a.procs)
    if "cl" in a.stages:
        stage_cl(a.datasets, a.workers * a.threads)
    if "run" in a.stages:
        stage_run(a.datasets, a.workers, a.threads)
    if "sum" in a.stages:
        stage_sum(a.datasets)
    log("done")


if __name__ == "__main__":
    main()
