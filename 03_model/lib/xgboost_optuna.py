#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/lib/xgboost_optuna.py
─────────────────────────────────────────────────────────────────────────────
XGBoost Optuna hyperparameter search space definition.

When train_xgboost.py is called with --optuna, it uses suggest_params(trial) of this module
to sample hyperparameters for each trial. To change the search space, edit only this file.

9 hyperparameters:
  n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
  min_child_weight, gamma, reg_alpha, reg_lambda
"""

# fixed (not searched) defaults — merged with the suggest_params result to form the final params
FIXED = dict(
    objective="multi:softprob",
    tree_method="hist",
    eval_metric="mlogloss",
)


def suggest_params(trial):
    """Optuna trial → XGBoost hyperparameter dict (9 entries)."""
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth        = trial.suggest_int("max_depth", 3, 12),
        learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    )


# Optuna search setting defaults (can be overridden by train_xgboost arguments)
DEFAULT_N_TRIALS = 50
DEFAULT_TIMEOUT  = None          # seconds. None=unlimited (limited only by n_trials)
