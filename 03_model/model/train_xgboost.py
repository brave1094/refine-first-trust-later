#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/model/train_xgboost.py
─────────────────────────────────────────────────────────────────────────────
XGBoost classifier training (called by 01_train.py as run(args)).

Input data: extractor output CSV
  {train_dir}/train_features.csv , {test_dir}/test_features.csv
  (common 8 columns: Label|filename|Stream_num|protocol|srcip|srcport|dstip|dstport
   + 1205 feature columns. Label is group_key, integer-encoded via label_map.json)

Two modes:
  (1) single training  : takes 9 hyperparameters as arguments and trains once
  (2) --optuna   : automatic tuning over the search space in 03_model/own_models/02_xgboost/optuna_space.py
                   → retrain with the best parameters and save the final results

9 hyperparameters (arguments for single training):
  n_estimators max_depth learning_rate subsample colsample_bytree
  min_child_weight gamma reg_alpha reg_lambda
  (+ early_stopping: 0=off)

Optuna arguments:
  --optuna              enable
  --optuna_bas f1|acc   optimization criterion (default acc). Both are recorded.
  --optuna_trials N     number of trials (default: lib-defined value)
  --optuna_timeout S    timeout in seconds

Output (result_dir):
  result.csv               per-round train/test mlogloss (best model)
  metrics.csv              accuracy, f1_macro, best_iteration, ...
  report.csv               per-class precision/recall/f1/support
  confusion_matrix.png
  feature_importance.csv   weight/gain/cover (sorted by gain, descending)
  best_params.json         (optuna) best hyperparameters
  optuna_trials.csv        (optuna) all trial records (params + acc + f1)
Output (param_dir):
  model.json               XGBoost native model
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths: static definitions (featset/optuna) in 03_model/own_models/02_xgboost/, results/tables in 03_model/
_MODEL_DIR = Path(__file__).resolve().parent          # …/03_model/model/
_CODE_DIR  = _MODEL_DIR.parent                        # …/03_model/
_XGB_CODE  = RP.OWN_MODELS / "02_xgboost"                # feature sets and search space
sys.path.insert(0, str(_XGB_CODE))                    # optuna.py etc.
sys.path.insert(0, str(_XGB_CODE / "feat"))           # xgboost_feature_all (for featset filtering)
sys.path.insert(0, str(_CODE_DIR / "lib"))            # result_table.py
sys.path.insert(0, str(_CODE_DIR))

# common 8 columns (excluded from features)
COMMON_COLS = ["Label", "filename", "Stream_num", "protocol",
               "srcip", "srcport", "dstip", "dstport"]
LABEL_COL = "Label"

# single-training defaults (when arguments are not given)
DEFAULTS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
    gamma=0.0, reg_alpha=0.0, reg_lambda=1.0, early_stopping=50,
)


# ═══════════════ Data loading (extractor CSV) ═══════════════
def _select_featset(featset: str, feat_names: list) -> list:
    """featset name → list of columns to use.
      all                : all
      only_{cat}         : that category only
      no_{cat}           : exclude that category
      no_headerflags     : exclude header + tcpflags
    Category is determined by reusing xgboost_feature_all.category_of (same criterion as extraction).
    """
    if not featset or featset == "all":
        return feat_names
    try:
        import xgboost_feature_all as fa
    except Exception:
        # if the feat module path is not on sys.path, keep as is (no filter)
        return feat_names
    cat_of = fa.category_of

    if featset == "no_headerflags":
        excl = {"header", "tcpflags"}
        return [c for c in feat_names if cat_of(c) not in excl]
    if featset.startswith("only_"):
        keep = featset[len("only_"):]
        return [c for c in feat_names if cat_of(c) == keep]
    if featset.startswith("no_"):
        drop = featset[len("no_"):]
        return [c for c in feat_names if cat_of(c) != drop]
    # unknown featset → all
    return feat_names


def _load_csv(feat_csv: Path, label_map: dict, featset: str = None):
    """train_features.csv → (X float32, y int64, feat_names). '-'(MISSING)→NaN.
    If featset is given, select only the columns of that category from the all CSV (no re-parsing)."""
    df = pd.read_csv(feat_csv, encoding="utf-8", low_memory=False)
    feat_names = [c for c in df.columns if c not in COMMON_COLS]
    if featset:
        feat_names = _select_featset(featset, feat_names)
    X = (df[feat_names]
         .apply(pd.to_numeric, errors="coerce")     # '-' → NaN (XGBoost handles NaN)
         .to_numpy(dtype=np.float32))
    y_mapped = df[LABEL_COL].astype(str).map(label_map)
    keep = y_mapped.notna().to_numpy()               # drop labels not in label_map
    if not keep.all():
        X = X[keep]
        y_mapped = y_mapped[keep]
    y = y_mapped.to_numpy(dtype=np.int64)
    return X, y, feat_names


def _find_features_csv(dataset_dir: Path, split_role: str):
    """
    split_role: 'train' or 'test'.
    New layout: {dataset_dir}/features.csv  (directory name is split_mode)
    Old layout fallback: {dataset_dir}/{split_role}_features.csv
    """
    for name in ("features.csv", f"{split_role}_features.csv"):
        p = Path(dataset_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Feature CSV not found: {dataset_dir}/features.csv\n"
        f"       Run 06_make_dataset.py --model xgboost first.")


# ═══════════════ confusion matrix ═══════════════
def _save_confusion_matrix(y_true, y_pred, class_names, save_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _cuda_ok():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


def _device(args):
    use_gpu = getattr(args, "gpu", None) is not None and _cuda_ok()
    return f"cuda:{args.gpu}" if use_gpu else "cpu"


# ═══════════════ Single training (with params dict) ═══════════════
class _NativeXGB:
    """Wrapper that wraps a native xgb.train Booster like an XGBClassifier.

    The LabelEncoder of XGBClassifier requires labels to be contiguous 0..N-1,
    so training fails when some classes vanish under denoised (label values become sparse).
    Native xgb.train has no such check, so with num_class fixed to the full label_map size
    vanished classes are treated as 'classes absent from the training data' and training proceeds.
    (those classes get predicted probabilities close to 0)

    Provides XGBClassifier-compatible methods:
      predict / get_booster / evals_result / best_iteration / save_model
    """
    def __init__(self, booster, evals_result, best_iteration, num_class):
        self._booster = booster
        self._evals = evals_result
        self.best_iteration = best_iteration
        self._num_class = num_class

    def get_booster(self):
        return self._booster

    def evals_result(self):
        return self._evals

    def predict(self, X):
        import xgboost as xgb
        import numpy as np
        dm = xgb.DMatrix(X)
        # softprob → (n, num_class) probabilities → labels via argmax
        proba = self._booster.predict(dm)
        if proba.ndim == 1:                       # if it comes back 1D, keep as is
            return proba.astype(int)
        return np.asarray(proba).argmax(axis=1)

    def save_model(self, path):
        self._booster.save_model(str(path))


def _fit(params, X_train, y_train, X_test, y_test, num_classes, device,
         seed, early_stopping, verbose=False):
    """Train with native xgb.train (num_class fixed to the full label_map size).
    Returns: _NativeXGB (XGBClassifier-compatible wrapper)."""
    import xgboost as xgb

    p = dict(params)
    n_estimators = int(p.pop("n_estimators", 100))   # used as num_boost_round
    booster_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "seed": seed,
        "nthread": 12,
    }
    # map XGBClassifier-style names → native names
    rename = {"learning_rate": "eta", "reg_alpha": "alpha", "reg_lambda": "lambda"}
    for k, v in p.items():
        booster_params[rename.get(k, k)] = v

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)
    evals = [(dtrain, "validation_0"), (dtest, "validation_1")]
    evals_result = {}

    kw = dict(
        params=booster_params, dtrain=dtrain, num_boost_round=n_estimators,
        evals=evals, evals_result=evals_result,
        verbose_eval=(max(1, n_estimators // 20) if verbose else False),
    )
    if early_stopping and early_stopping > 0:
        kw["early_stopping_rounds"] = early_stopping

    booster = xgb.train(**kw)
    best_it = getattr(booster, "best_iteration", None)
    if best_it is None:
        best_it = n_estimators - 1
    return _NativeXGB(booster, evals_result, best_it, num_classes)


def _score(clf, X_test, y_test):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 balanced_accuracy_score)
    # set device to cpu before prediction (avoids GPU training + CPU data mismatch warning)
    try:
        clf.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass
    y_pred = clf.predict(X_test)   # wrapper returns 1D labels via argmax
    return (accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred, average="macro"),
            balanced_accuracy_score(y_test, y_pred), y_pred)


# ═══════════════ Optuna tuning ═══════════════
def _run_optuna(args, X_train, y_train, X_test, y_test, num_classes,
                device, result_dir):
    import optuna
    import optuna_space as space

    basis = getattr(args, "optuna_bas", None) or "acc"      # default acc
    n_trials = (getattr(args, "optuna_trials", None)
                or space.DEFAULT_N_TRIALS)
    timeout = getattr(args, "optuna_timeout", None) or space.DEFAULT_TIMEOUT
    early = (args.early_stopping if getattr(args, "early_stopping", None) is not None
             else DEFAULTS["early_stopping"])

    trials_log = []

    def objective(trial):
        params = space.suggest_params(trial)
        clf = _fit(params, X_train, y_train, X_test, y_test, num_classes,
                   device, args.seed, early, verbose=False)
        acc, f1m, bal, _ = _score(clf, X_test, y_test)
        trial.set_user_attr("accuracy", acc)
        trial.set_user_attr("f1_macro", f1m)
        trial.set_user_attr("balanced_acc", bal)
        row = {"trial": trial.number, "accuracy": round(acc, 6),
               "f1_macro": round(f1m, 6), "balanced_acc": round(bal, 6),
               **params}
        trials_log.append(row)
        return {"acc": acc, "f1": f1m, "bal": bal}[basis]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    print(f"[OPTUNA] criterion={basis}  trials={n_trials}  "
          f"timeout={timeout}  device={device}")
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=True)

    best = study.best_params
    print(f"[OPTUNA] best {basis}={study.best_value:.4f}")
    print(f"[OPTUNA] best params: {best}")

    # CSV of all trial records
    result_dir.mkdir(parents=True, exist_ok=True)
    if trials_log:
        keys = ["trial", "accuracy", "f1_macro", "balanced_acc"] + list(best.keys())
        sort_key = {"acc": "accuracy", "f1": "f1_macro",
                    "bal": "balanced_acc"}[basis]
        with open(result_dir / "optuna_trials.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in sorted(trials_log, key=lambda x: x[sort_key], reverse=True):
                w.writerow(r)
    with open(result_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({"basis": basis, "best_value": study.best_value,
                   "params": best}, f, ensure_ascii=False, indent=2)
    return best


# ═══════════════ Entry point ═══════════════
def run(args):
    from sklearn.metrics import classification_report

    train_dir = Path(args.dataset_dir)
    test_dir  = Path(getattr(args, "test_dataset_dir", args.dataset_dir))
    # detailed outputs are saved under model_results/<variant>/{model}/{dataset}/exp{N}/
    from lib.train_common import _vroot
    result_dir = (_vroot("results") / args.model / args.dataset
                  / f"exp{args.exp}")
    param_dir  = Path(args.param_dir)
    # if featset is not all/None, put results in a featset subfolder (avoid overwriting).
    # all keeps the existing path unchanged.
    _featset = getattr(args, "featset", None)
    if _featset and _featset != "all":
        result_dir = result_dir / _featset
        param_dir  = param_dir / _featset
    result_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    with open(args.label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    class_names = [inv[i] for i in range(len(inv))]
    num_classes = len(class_names)

    featset = getattr(args, "featset", None)
    X_train, y_train, feat_names = _load_csv(
        _find_features_csv(train_dir, "train"), label_map, featset)
    X_test, y_test, _ = _load_csv(
        _find_features_csv(test_dir, "test"), label_map, featset)

    fs_txt = f"  featset={featset}" if featset else ""
    print(f"[INFO] X_train={X_train.shape} X_test={X_test.shape} "
          f"classes={num_classes} features={len(feat_names)}{fs_txt}")

    device = _device(args)
    early = (args.early_stopping if getattr(args, "early_stopping", None) is not None
             else DEFAULTS["early_stopping"])

    # ── Hyperparameter selection: optuna or arguments ──
    if getattr(args, "optuna", False):
        best = _run_optuna(args, X_train, y_train, X_test, y_test,
                           num_classes, device, result_dir)
        params = best
    else:
        def pick(name):
            v = getattr(args, name, None)
            return v if v is not None else DEFAULTS[name]
        params = {k: pick(k) for k in (
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "min_child_weight", "gamma",
            "reg_alpha", "reg_lambda")}

    # ── Final (or single) training ──
    clf = _fit(params, X_train, y_train, X_test, y_test, num_classes,
               device, args.seed, early, verbose=True)

    ev = clf.evals_result()
    tr_ll = ev["validation_0"]["mlogloss"]
    te_ll = ev["validation_1"]["mlogloss"]
    best_it = getattr(clf, "best_iteration", None)
    if best_it is None:
        best_it = len(tr_ll) - 1

    acc, f1m, bal, y_pred = _score(clf, X_test, y_test)
    print(f"[RESULT] exp{args.exp}  acc={acc:.4f}  f1_macro={f1m:.4f}  "
          f"bal_acc={bal:.4f}  best_iteration={best_it}")

    # ── Save ──
    with open(result_dir / "result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "train_mlogloss", "test_mlogloss"])
        for i, (a, b) in enumerate(zip(tr_ll, te_ll)):
            w.writerow([i, round(a, 6), round(b, 6)])

    with open(result_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["accuracy", "f1_macro", "balanced_acc", "best_iteration",
                    "n_train", "n_test", "n_features", "tuned"])
        w.writerow([round(acc, 6), round(f1m, 6), round(bal, 6), best_it,
                    len(y_train), len(y_test), X_train.shape[1],
                    bool(getattr(args, "optuna", False))])

    rep = classification_report(y_test, y_pred, labels=list(range(num_classes)),
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    with open(result_dir / "report.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for name in class_names + ["macro avg", "weighted avg"]:
            r = rep.get(name, {})
            w.writerow([name, round(r.get("precision", 0), 4),
                        round(r.get("recall", 0), 4),
                        round(r.get("f1-score", 0), 4),
                        int(r.get("support", 0))])

    _save_confusion_matrix(
        y_test, y_pred, class_names, result_dir / "confusion_matrix.png",
        f"XGBoost [{args.dataset}/exp{args.exp}] "
        f"acc={acc:.4f} f1={f1m:.4f}")

    booster = clf.get_booster()
    booster.feature_names = feat_names
    imp = {t: booster.get_score(importance_type=t)
           for t in ("weight", "gain", "cover")}
    rows = [[name, imp["weight"].get(name, 0.0), imp["gain"].get(name, 0.0),
             imp["cover"].get(name, 0.0)] for name in feat_names]
    rows.sort(key=lambda r: r[2], reverse=True)
    with open(result_dir / "feature_importance.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", "weight", "gain", "cover"])
        for r in rows:
            w.writerow([r[0], round(r[1], 4), round(r[2], 4), round(r[3], 4)])

    clf.save_model(param_dir / "model.json")
    with open(param_dir / "params_used.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] results -> {result_dir}")
    print(f"[SAVE] model   -> {param_dir / 'model.json'}")

    # ── Standard result CSV + wrong_list + result_table (framework-wide convention) ──
    from train_common import (ResultLogger, save_wrong_list,
                              update_result_table)
    train_acc = None
    try:
        y_pred_tr = clf.predict(X_train)
        train_acc = float((y_pred_tr == y_train).mean())
    except Exception:
        pass
    logger = ResultLogger(args.model, args.dataset, args.exp)
    logger.log(best_it, tr_ll[min(best_it, len(tr_ll) - 1)], train_acc,
               te_ll[min(best_it, len(te_ll) - 1)], acc)
    logger.close()

    # wrong-answer list: features.csv row order = X_test order (reflects dropping labels outside label_map)
    try:
        df_te = pd.read_csv(_find_features_csv(test_dir, "test"),
                            usecols=["Label", "filename", "Stream_num"],
                            dtype=str, encoding="utf-8")
        keep = df_te["Label"].map(label_map).notna()
        df_te = df_te[keep].reset_index(drop=True)
        files_te = pd.DataFrame({"filename": df_te["filename"],
                                 "session_id": df_te["Stream_num"],
                                 "group_key": df_te["Label"]})
        save_wrong_list(args.model, args.dataset, args.exp,
                        files_te, y_test, y_pred, label_map)
    except Exception as e:
        print(f"[wrong_list] generation failed (ignored): {e}")
    update_result_table(args.model, args.dataset, args.exp, acc)