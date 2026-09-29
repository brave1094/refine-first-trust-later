#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_2dcnn.py — 2D-CNN (Wang et al. 2017) training.

Input: 01_dataset/{ds}/2dcnn/{split_mode}/x_data.npy (N,28,28) float32
Model: 03_model/own_models/03_2dcnn/model.py CNN2D
Hyperparams: same as KNOM (epochs 40, batch 64, lr 1e-3, test_interval 5)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lib.train_common import (load_label_map, load_npy_split,
                              torch_train_loop)

DEFAULTS = dict(epochs=40, batch_size=64, lr=1e-3, test_interval=5)

CODE_DIR = Path(__file__).resolve().parent          # …/03_model/model/
WORK_DIR = CODE_DIR.parent.parent                    # repository root


def _load_cnn2d():
    model_py = RP.OWN_MODELS / "03_2dcnn" / "model.py"
    spec = importlib.util.spec_from_file_location("_cnn2d_model", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CNN2D


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")

    label_map = load_label_map(args.label_map_path)
    X_tr, Y_tr, _ = load_npy_split(args.train_dataset_dir)
    X_te, Y_te, files_te = load_npy_split(args.test_dataset_dir)
    print(f"  train: {X_tr.shape}  test: {X_te.shape}  classes: {len(label_map)}")

    bs = args.batch_size or DEFAULTS["batch_size"]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                      torch.from_numpy(Y_tr).long()),
        batch_size=bs, shuffle=True, num_workers=2, drop_last=False)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te).float(),
                      torch.from_numpy(Y_te).long()),
        batch_size=bs, shuffle=False, num_workers=2)

    CNN2D = _load_cnn2d()
    model = CNN2D(num_classes=len(label_map)).to(device)

    save_path = Path(args.param_dir) / "best_model.pt"
    torch_train_loop(model=model, train_loader=train_loader,
                     test_loader=test_loader, device=device, args=args,
                     defaults=DEFAULTS, files_test=files_te,
                     label_map=label_map, save_path=save_path)
