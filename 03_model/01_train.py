#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/01_train.py
─────────────────────────────────
Model training entry point (router, 8 models).

Experiment groups:
  --exp 1 : train=denoised / test=denoised
  --exp 2 : train=denoised / test=noisy
  --exp 3 : train=noisy    / test=denoised
  --exp 4 : train=noisy    / test=noisy

Data location (06_make_dataset outputs):
  01_dataset/{dataset}/{model}/{train_noisy|train_denoised|test_noisy|test_denoised}/
  01_dataset/{dataset}/00_filelist/label_map.json

Outputs:
  results/{model}/{dataset}/{model}_{dataset}_exp{N}.csv
      (epoch, train_loss, train_accuracy, test_loss, test_accuracy)
  param/{model}/{dataset}/exp{N}/            ← saved only when test performance reaches a new best
  wrong_list/{model}/{dataset}/exp{N}_wrong.csv  ← list of misclassified sessions (for Exp.2 analysis)
  result_table/{model}.csv                   ← dataset | exp1 | exp2 | exp3 | exp4

Usage:
  python3 01_train.py --model xgboost --dataset vpn16 --exp 1
  python3 01_train.py --model rf      --dataset vpn16 --exp 4
  python3 01_train.py --model 2dcnn   --dataset vpn16 --exp 1 --gpu 0 --epochs 40
  python3 01_train.py --model etbert  --dataset vpn16 --exp 2 --gpu 1 --batch_size 32
  python3 01_train.py --model yatc    --dataset ustc16 --exp 1 --gpu 2
  python3 01_train.py --model netmamba --dataset ustc16 --exp 1 --gpu 3
  python3 01_train.py --model trafficformer --dataset vpn16 --exp 1 --gpu 0
  python3 01_train.py --model mm4flow --dataset vpn16 --exp 1 --gpu 1
"""
import argparse
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent       # …/03_model/
WORK_DIR = CODE_DIR.parent                        # repository root

EX_MAP = {
    1: ("denoised", "denoised"),
    2: ("denoised", "noisy"),
    3: ("noisy",    "denoised"),
    4: ("noisy",    "noisy"),
}

MODELS = ["rf", "xgboost", "2dcnn", "etbert", "yatc", "netmamba",
          "trafficformer", "mm4flow", "netfound"]


VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}


def common_paths(args):
    variant = getattr(args, "variant", "full")
    base = WORK_DIR / "01_dataset" / args.dataset
    train_dir = base / variant / args.model / f"train_{args.train_mode}"
    test_dir = base / variant / args.model / f"test_{args.test_mode}"
    label_map = base / VARIANT_FL[variant] / "label_map.json"
    from lib.train_common import param_dir_path
    param_dir = param_dir_path(args.model, args.dataset, args.exp)   # separated by SCIE_VARIANT
    return train_dir, test_dir, label_map, param_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   required=True, choices=MODELS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp",     required=True, type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--variant", choices=["full", "sizectrl", "strat"], default="full",
                    help="dataset variant to read + output separation folder (SCIE_VARIANT)")
    # common
    ap.add_argument("--gpu",           type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--epochs",        type=int, default=None)
    ap.add_argument("--batch_size",    type=int, default=None)
    ap.add_argument("--lr",            type=float, default=None)
    ap.add_argument("--test_interval", type=int, default=None)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--no_amp",        action="store_true")
    # tree hyperparameters shared by rf / xgboost
    ap.add_argument("--n_estimators",     type=int,   default=None)
    ap.add_argument("--max_depth",        type=int,   default=None)
    ap.add_argument("--learning_rate",    type=float, default=None)
    ap.add_argument("--subsample",        type=float, default=None)
    ap.add_argument("--colsample_bytree", type=float, default=None)
    ap.add_argument("--min_child_weight", type=int,   default=None)
    ap.add_argument("--gamma",            type=float, default=None)
    ap.add_argument("--reg_alpha",        type=float, default=None)
    ap.add_argument("--reg_lambda",       type=float, default=None)
    ap.add_argument("--early_stopping",   type=int,   default=None)
    ap.add_argument("--optuna", action="store_true",
                    help="[rf/xgboost] automatic hyperparameter tuning with Optuna")
    ap.add_argument("--featset", default=None,
                    help="[rf/xgboost] feature category combination code (unset=all 1,205)")
    ap.add_argument("--optuna_bas", choices=["f1", "acc", "bal"], default="acc")
    ap.add_argument("--optuna_trials", type=int, default=None)
    ap.add_argument("--optuna_timeout", type=int, default=None)
    args = ap.parse_args()
    # outputs (param/results/result_table/…) are separated by variant. seed≠42 goes under variant_seed{N}
    # as a further split → multi-seed repeats never overwrite the existing seed42 results.
    # (input paths common_paths use args.variant directly, so unaffected — training on the same data)
    os.environ["SCIE_VARIANT"] = args.variant + (
        f"_seed{args.seed}" if args.seed != 42 else "")

    train_mode, test_mode = EX_MAP[args.exp]
    args.train_mode, args.test_mode = train_mode, test_mode

    sys.path.insert(0, str(CODE_DIR))
    train_dir, test_dir, label_map, param_dir = common_paths(args)
    args.train_dataset_dir = train_dir
    args.test_dataset_dir = test_dir
    args.dataset_dir = train_dir                 # backward compat
    args.label_map_path = label_map
    args.param_dir = param_dir

    for d, role in ((train_dir, "train"), (test_dir, "test")):
        if not Path(d).exists():
            print(f"[ERROR] {role} data not found: {d}\n"
                  f"        run 06_make_dataset.py --model {args.model} first")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  model     : {args.model}")
    print(f"  dataset   : {args.dataset}")
    print(f"  exp       : Exp.{args.exp}  (train={train_mode} → test={test_mode})")
    print(f"  gpu       : {args.gpu}")
    print(f"  train_dir : {train_dir}")
    print(f"  test_dir  : {test_dir}")
    print(f"  param_dir : {param_dir}")
    print(f"{'='*70}\n")

    MODEL_DIR = CODE_DIR / "model"
    sys.path.insert(0, str(MODEL_DIR))
    mod_name = f"train_{args.model}"
    mod = __import__(mod_name)
    mod.run(args)


if __name__ == "__main__":
    main()
