#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a5_treeshap.py — Illusion 5 (feature importance errors/instability) TreeSHAP, sizectrl tree (rf/xgb).
  Pass exp1 (refined training) vs exp3 (noisy training) models over 'the same refined test input' → compare mean |SHAP|.
  Checks concentration on artifacts such as ip_id without the high-cardinality bias of impurity importance.
  Output: shap/{model}_{ds}_treeshap_full.csv / _noSII.csv + shap/_summary_{model}.csv
  Run: CUDA_VISIBLE_DEVICES=0 python3 a5_treeshap.py --model rf
        CUDA_VISIBLE_DEVICES=1 python3 a5_treeshap.py --model xgboost
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os, re, csv, json, argparse
import numpy as np
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.normpath(os.path.join(HERE, "..", "01_dataset"))
VARIANT = os.environ.get("SCIE_VARIANT", "")          # full|sizectrl → separate paths/outputs
PARAM = str(RP.PARAM)
OUT   = os.path.join(str(RP.SHAP_TREE), VARIANT); os.makedirs(OUT, exist_ok=True)

DSS = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]
COMMON_COLS = ["Label", "filename", "Stream_num", "protocol",
               "srcip", "srcport", "dstip", "dstport"]
FILL_NA = -1.0
# artifact/collection-dependent (SII-type): raw header byte fields + 5-tuple
ARTIFACT = re.compile(r"(ip_id|checksum|ip_ttl|dsfield|ip_flags|_frag|ip_ver|"
                      r"ip_total_len|ip_hdr|port|addr|(^|_)src|(^|_)dst)", re.I)


def load_X(ds, model, n):
    """Same composition as training (_load_csv/_load): all columns except COMMON_COLS as feat_cols,
    to_numeric(coerce) turns array-strings→NaN. rf: NaN→-1.0, xgb: keeps NaN (native).
    Returns: (X float32 numpy, feat_cols)."""
    d = os.path.join(DATA, ds, VARIANT, model, "test_denoised")
    df = pd.read_csv(os.path.join(d, "features.csv"), low_memory=False).head(n)
    feat_cols = [c for c in df.columns if c not in COMMON_COLS]
    Xdf = df.reindex(columns=feat_cols).apply(pd.to_numeric, errors="coerce")
    if model == "rf":
        Xdf = Xdf.fillna(FILL_NA)
    X = Xdf.to_numpy(dtype=np.float32)
    return X, feat_cols


def load_model(model, ds, exp):
    base = os.path.join(PARAM, VARIANT, model, ds, f"exp{exp}")
    if model == "rf":
        import joblib
        return joblib.load(os.path.join(base, "best_model.joblib"))
    if model == "xgboost":
        import xgboost as xgb
        b = xgb.Booster(); b.load_model(os.path.join(base, "model.json")); return b
    raise ValueError(model)


def shap_importance(model_obj, model, X, feat_cols):
    """X: float32 numpy. For xgb, clear feature_names and predict with a position-based DMatrix."""
    import shap
    nfeat = X.shape[1]
    if model == "xgboost":
        model_obj.feature_names = None            # position-based → avoid DMatrix name mismatch
    expl = shap.TreeExplainer(model_obj)
    sv = expl.shap_values(X, check_additivity=False)
    if isinstance(sv, list):                      # multiclass: list of per-class arrays
        arr = np.mean([np.abs(s).mean(0) for s in sv], 0)
    else:
        a = np.abs(np.asarray(sv))
        if a.ndim == 3:                           # (n,f,c) etc. → keep only the feature axis and average
            faxis = [i for i, s in enumerate(a.shape) if s == nfeat][0]
            arr = a.mean(axis=tuple(i for i in range(3) if i != faxis))
        else:
            arr = a.mean(0)
    return pd.Series(np.asarray(arr).ravel(), index=feat_cols)


def run(model, n, topk=30):
    summ = []
    for ds in DSS:
        try:
            m1, m3 = load_model(model, ds, 1), load_model(model, ds, 3)
            X, feat_cols = load_X(ds, model, n)   # same composition as training, numpy
            imp1 = shap_importance(m1, model, X, feat_cols)
            imp3 = shap_importance(m3, model, X, feat_cols)
        except Exception as e:
            print(f"  [skip] {model}/{ds}: {e}"); continue
        p1 = imp1 / imp1.sum() if imp1.sum() else imp1
        p3 = imp3 / imp3.sum() if imp3.sum() else imp3
        out = pd.DataFrame({"exp1_den": p1, "exp3_noisy": p3})
        out["delta"] = out["exp3_noisy"] - out["exp1_den"]
        out.sort_values("exp3_noisy", ascending=False).head(topk).round(6) \
           .to_csv(os.path.join(OUT, f"{model}_{ds}_treeshap_full.csv"), encoding="utf-8-sig")
        nosii = out[~out.index.to_series().str.contains(ARTIFACT)]
        nosii.sort_values("exp3_noisy", ascending=False).head(topk).round(6) \
             .to_csv(os.path.join(OUT, f"{model}_{ds}_treeshap_noSII.csv"), encoding="utf-8-sig")
        rho = p1.rank().corr(p3.rank())
        t1, t3 = set(p1.nlargest(10).index), set(p3.nlargest(10).index)
        jac = len(t1 & t3) / len(t1 | t3)
        am = out.index.to_series().str.contains(ARTIFACT)
        art1, art3 = p1[am].sum(), p3[am].sum()
        summ.append([ds, round(rho, 3), round(jac, 3),
                     round(art1*100, 1), round(art3*100, 1),
                     f"{p1.idxmax()}({p1.max()*100:.1f}%)",
                     f"{p3.idxmax()}({p3.max()*100:.1f}%)"])
        print(f"  {model}/{ds:<8} rankρ={rho:.2f} top10overlap={jac:.2f} "
              f"artifact {art1*100:.1f}%→{art3*100:.1f}%  "
              f"top1 {p1.idxmax()}→{p3.idxmax()}({p3.max()*100:.1f}%)")
    with open(os.path.join(OUT, f"_summary_{model}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "rank_corr", "top10_jaccard",
                    "artifact_mass_exp1(%)", "artifact_mass_exp3(%)",
                    "top1_exp1", "top1_exp3"])
        w.writerows(summ)
    print(f"[{model}] → {OUT}/_summary_{model}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rf", "xgboost"])
    ap.add_argument("--n", type=int, default=2000, help="number of SHAP explanation samples")
    a = ap.parse_args()
    run(a.model, a.n)
