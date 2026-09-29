#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_netmamba.py — NetMamba (ICNP 2024) fine-tuning adapter.

Input : 01_dataset/{ds}/netmamba/{split_mode}/x_data.npy (N,40,40) uint8
Model : 00_assets/models/06_netmamba/src/models_net_mamba.py net_mamba_classifier
       (requires a mamba_ssm 1.1.1 CUDA build — see dependencies in INTEGRATION_SPEC.md)
Pre-training: 00_assets/models/06_netmamba/pre-train.pth (HuggingFace wangtz/NetMamba)
          if missing, training from scratch (prints a warning)
Hyperparameters (original): AdamW, lr=blr*batch/256 (blr 2e-3), wd 0.05, warmup 20ep + cosine,
  smoothing 0.1, batch 64, epochs 120, AMP off recommended (--no_amp applied by default)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lib.train_common import (load_label_map, load_npy_split,
                              torch_train_loop)

DEFAULTS = dict(epochs=120, batch_size=64, blr=2e-3, test_interval=5,
                warmup_epochs=20, weight_decay=0.05, min_lr=1e-6,
                smoothing=0.1)

CODE_DIR = Path(__file__).resolve().parent
WORK_DIR = CODE_DIR.parent.parent
NM_ROOT = RP.UPSTREAM / "06_netmamba"
PRETRAINED = NM_ROOT / "pre-train.pth"


def _load_model_mod():
    src = NM_ROOT / "src"
    for p in (str(NM_ROOT), str(src)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        "_models_netmamba", src / "models_net_mamba.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(x_uint8):
    x = torch.from_numpy(x_uint8).float().div(255).sub(0.5).div(0.5)
    return x.unsqueeze(1)                       # (N,1,40,40)


def _warmup_cosine(optimizer, epochs, warmup, base_lr, min_lr):
    def f(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(epochs - warmup, 1)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("  [warning] NetMamba requires mamba_ssm CUDA kernels — likely cannot run on CPU")

    label_map = load_label_map(args.label_map_path)
    X_tr, Y_tr, _ = load_npy_split(args.train_dataset_dir)
    X_te, Y_te, files_te = load_npy_split(args.test_dataset_dir)
    print(f"  train: {X_tr.shape}  test: {X_te.shape}  classes: {len(label_map)}")

    bs = args.batch_size or DEFAULTS["batch_size"]
    train_loader = DataLoader(TensorDataset(_norm(X_tr),
                                            torch.from_numpy(Y_tr).long()),
                              batch_size=bs, shuffle=True, num_workers=2,
                              drop_last=True)
    test_loader = DataLoader(TensorDataset(_norm(X_te),
                                           torch.from_numpy(Y_te).long()),
                             batch_size=bs, shuffle=False, num_workers=2)

    mod = _load_model_mod()
    model = mod.net_mamba_classifier(num_classes=len(label_map))

    if PRETRAINED.exists():
        ckpt = torch.load(PRETRAINED, map_location="cpu")
        state = ckpt.get("model", ckpt)
        for k in ("head.weight", "head.bias"):
            state.pop(k, None)
        missing = model.load_state_dict(state, strict=False)
        print(f"  [pretrained] {PRETRAINED.name} loaded "
              f"(missing={len(missing.missing_keys)})")
        try:
            nn.init.trunc_normal_(model.head.weight, std=2e-5)
        except Exception:
            pass
    else:
        print(f"  [warning] no pre-trained checkpoint → training from scratch\n"
              f"         expected location: {PRETRAINED} (HF wangtz/NetMamba pre-train.pth)")

    model = model.to(device)
    epochs = args.epochs or DEFAULTS["epochs"]
    lr = args.lr or DEFAULTS["blr"] * bs / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=DEFAULTS["weight_decay"])
    scheduler = _warmup_cosine(optimizer, epochs, DEFAULTS["warmup_epochs"],
                               lr, DEFAULTS["min_lr"])
    criterion = nn.CrossEntropyLoss(label_smoothing=DEFAULTS["smoothing"])

    save_path = Path(args.param_dir) / "best_model.pt"
    torch_train_loop(model=model, train_loader=train_loader,
                     test_loader=test_loader, device=device, args=args,
                     defaults=DEFAULTS, files_test=files_te,
                     label_map=label_map, optimizer=optimizer,
                     scheduler=scheduler, criterion=criterion,
                     save_path=save_path)
