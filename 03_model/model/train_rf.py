#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rf.py — Random Forest (scikit-learn) training.

Input: 01_dataset/{ds}/rf/{split_mode}/features.csv (1,205 features — same as xgboost)
  - RF does not support NaN → missing ('-') replaced with -1
Results: the results CSV has a single row (epoch=1). best param = trained model (joblib).
Options: --n_estimators(default 300) --max_depth(default None) --optuna(light search)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lib.train_common import (ResultLogger, load_label_map, save_wrong_list,
                              update_result_table)

DEFAULTS = dict(n_estimators=300, max_depth=None, min_samples_leaf=1,
                max_features="sqrt")

COMMON_COLS = ["Label", "filename", "Stream_num", "protocol",
               "srcip", "srcport", "dstip", "dstport"]
FILL_NA = -1.0


def _find_csv(d):
    d = Path(d)
    for name in ("features.csv", "train_features.csv", "test_features.csv"):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"features.csv not found: {d}")


def _load(d, label_map, feat_cols=None):
    df = pd.read_csv(_find_csv(d), na_values=["-"], low_memory=False)
    y = df["Label"].astype(str).map(label_map)
    keep = y.notna()
    df, y = df[keep], y[keep].astype(np.int64).to_numpy()
    files = pd.DataFrame({
        "filename": df["filename"].astype(str),
        "session_id": df.get("Stream_num", "-").astype(str),
        "group_key": df["Label"].astype(str),
    }).reset_index(drop=True)
    if feat_cols is None:
        feat_cols = [c for c in df.columns if c not in COMMON_COLS]
    X = (df.reindex(columns=feat_cols)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(FILL_NA).to_numpy(dtype=np.float32))
    return X, y, files, feat_cols


def run(args):
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    label_map = load_label_map(args.label_map_path)
    X_tr, y_tr, _, feat_cols = _load(args.train_dataset_dir, label_map)
    X_te, y_te, files_te, _ = _load(args.test_dataset_dir, label_map, feat_cols)
    print(f"  train: {X_tr.shape}  test: {X_te.shape}  classes: {len(label_map)}")

    params = dict(
        n_estimators=args.n_estimators or DEFAULTS["n_estimators"],
        max_depth=args.max_depth or DEFAULTS["max_depth"],
        max_features=DEFAULTS["max_features"],
        min_samples_leaf=DEFAULTS["min_samples_leaf"],
        n_jobs=12, random_state=args.seed,
    )

    if args.optuna:
        import optuna
        import numpy as np
        from sklearn.model_selection import train_test_split
        # stratified split impossible if the smallest class has fewer than 2 samples
        #   (e.g. when sizectrl subsampling reduces a class to 1 sample)
        #   This split is only for the Optuna validation set, so falling back to a plain random split is fine.
        _cls, _cnt = np.unique(y_tr, return_counts=True)
        _strat = y_tr if _cnt.min() >= 2 else None
        if _strat is None:
            print(f"  [rf] smallest class sample count {int(_cnt.min())} "
                  f"→ train/val split without stratify")
        Xa, Xv, ya, yv = train_test_split(
            X_tr, y_tr, test_size=0.2, stratify=_strat, random_state=args.seed)

        def objective(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 8, 40),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.3]),
                n_jobs=12, random_state=args.seed,
            )
            m = RandomForestClassifier(**p).fit(Xa, ya)
            return m.score(Xv, yv)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials or 30,
                       timeout=args.optuna_timeout)
        params.update(study.best_params)
        params.update(n_jobs=12, random_state=args.seed)
        print(f"  [optuna] best={study.best_value:.4f}  params={study.best_params}")

    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    train_acc = model.score(X_tr, y_tr)
    y_pred = model.predict(X_te)
    test_acc = float((y_pred == y_te).mean())
    print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    logger = ResultLogger(args.model, args.dataset, args.exp)
    logger.log(1, 0.0, train_acc, 0.0, test_acc)
    logger.close()

    joblib.dump(model, Path(args.param_dir) / "best_model.joblib")
    # save feature importance (impurity-based) — input for axis-⑤ analysis
    pd.DataFrame({"feature": feat_cols,
                  "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(Path(args.param_dir) / "feature_importance.csv", index=False)

    save_wrong_list(args.model, args.dataset, args.exp,
                    files_te, y_te, y_pred, label_map)
    update_result_table(args.model, args.dataset, args.exp, test_acc)
