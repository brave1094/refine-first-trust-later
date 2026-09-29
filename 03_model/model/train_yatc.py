#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yatc.py — YaTC (AAAI 2023) fine-tuning adapter.

Input : 01_dataset/{ds}/yatc/{split_mode}/x_data.npy (N,40,40) uint8 (MFR)
Model : 00_assets/models/05_yatc/models_YaTC.py TraFormer_YaTC  (requires timm 0.3.2)
Pretrained: 00_assets/models/05_yatc/output_dir/pretrained-model.pth (Google Drive, see SPEC)
          if missing, train from scratch (prints a warning)
Hyperparameters (original fine-tune.py): AdamW, lr=blr*batch/256 (blr 2e-3), wd 0.05,
  warmup 20ep + cosine, label smoothing 0.1, batch 64, epochs 200 (default, can be reduced)
  ※ layer-wise lr decay is not applied in this adapter (simplification — see README)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lib.train_common import (load_label_map, load_npy_split,
                              torch_train_loop)

DEFAULTS = dict(epochs=200, batch_size=64, blr=2e-3, test_interval=5,
                warmup_epochs=20, weight_decay=0.05, min_lr=1e-6,
                smoothing=0.1)

CODE_DIR = Path(__file__).resolve().parent
WORK_DIR = CODE_DIR.parent.parent
YATC_ROOT = RP.UPSTREAM / "05_yatc"
PRETRAINED = YATC_ROOT / "output_dir" / "pretrained-model.pth"


def _load_model_cls():
    import sys
    sys.path.insert(0, str(YATC_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_models_yatc", YATC_ROOT / "models_YaTC.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm(x_uint8):
    x = torch.from_numpy(x_uint8).float().div(255).sub(0.5).div(0.5)
    return x.unsqueeze(1)                       # (N,1,40,40)


def _warmup_cosine(optimizer, epochs, warmup, base_lr, min_lr):
    def f(ep):                                  # ep: 0-based
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

    mod = _load_model_cls()
    model = mod.TraFormer_YaTC(num_classes=len(label_map), drop_path_rate=0.1)

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
        print(f"  [warning] no pretrained checkpoint → training from scratch\n"
              f"         expected location: {PRETRAINED} (download link in INTEGRATION_SPEC.md)")

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
