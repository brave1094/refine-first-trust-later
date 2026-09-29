#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_model/lib/train_common.py
─────────────────────────────
Helpers shared by the 8 train_{model}.py scripts.

  - ResultLogger  : results/{model}/{dataset}/{model}_{dataset}_exp{N}.csv
                    (epoch, train_loss, train_accuracy, test_loss, test_accuracy)
  - BestTracker   : saves/updates param only at the best test performance + keeps predictions at the best point
  - save_wrong_list : list of misclassified sessions → wrong_list/{model}/{dataset}/exp{N}_wrong.csv
                    (for Exp.2 error analysis — tracks which files were misclassified)
  - update_result_table : result_table/{model}.csv (dataset | exp1 | exp2 | exp3 | exp4)
  - load_npy_split / load_files_csv : load shaping outputs
  - torch_train_loop : (optional) shared training loop for image-style models
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent      # …/03_model/

# If NM_NOSAVE=1, skip writing result_table and wrong_list.
# (so one-off probes such as batch-size search do not contaminate the actual result table)
_NOSAVE = os.environ.get("NM_NOSAVE") == "1"


# ═══════════════ paths ═══════════════
# If SCIE_VARIANT (full|sizectrl) is set, insert that subfolder into the output root
# so full/sizectrl results do not overwrite each other. (If unset, the previous path is used as is.)
_ROOTS = {"results": RP.MODEL_RESULTS, "param": RP.PARAM, "wrong_list": RP.WRONG_LIST, "result_table": RP.RESULT_TABLES}


def _vroot(name):
    v = os.environ.get("SCIE_VARIANT", "")
    return (_ROOTS[name] / v) if v else _ROOTS[name]


def result_csv_path(model, dataset, exp):
    d = _vroot("results") / model / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model}_{dataset}_exp{exp}.csv"


def param_dir_path(model, dataset, exp):
    d = _vroot("param") / model / dataset / f"exp{exp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def wrong_list_path(model, dataset, exp):
    d = _vroot("wrong_list") / model / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"exp{exp}_wrong.csv"


# ═══════════════ result logger ═══════════════
class ResultLogger:
    COLS = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]

    def __init__(self, model, dataset, exp):
        self.path = result_csv_path(model, dataset, exp)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.COLS)

    def log(self, epoch, train_loss, train_acc, test_loss=None, test_acc=None):
        self._w.writerow([
            epoch,
            f"{train_loss:.6f}" if train_loss is not None else "",
            f"{train_acc:.4f}" if train_acc is not None else "",
            f"{test_loss:.6f}" if test_loss is not None else "",
            f"{test_acc:.4f}" if test_acc is not None else "",
        ])
        self._f.flush()

    def close(self):
        self._f.close()


# ═══════════════ best tracking ═══════════════
class BestTracker:
    """Calls save_fn() to overwrite param only when the test acc reaches a new best."""

    def __init__(self, save_fn=None):
        self.best_acc = -1.0
        self.best_epoch = -1
        self.best_pred = None       # test predictions at the best point (for wrong_list)
        self.best_true = None
        self._save_fn = save_fn

    def update(self, epoch, test_acc, y_true=None, y_pred=None):
        if test_acc is None or test_acc <= self.best_acc:
            return False
        self.best_acc = test_acc
        self.best_epoch = epoch
        if y_true is not None:
            self.best_true = np.asarray(y_true)
        if y_pred is not None:
            self.best_pred = np.asarray(y_pred)
        if self._save_fn is not None:
            self._save_fn()
        return True


# ═══════════════ wrong list ═══════════════
def save_wrong_list(model, dataset, exp, files_df, y_true, y_pred, label_map):
    """files_df: files.csv of the test split (in x order). y_true/y_pred: int arrays."""
    if _NOSAVE:
        return None
    inv = {v: k for k, v in label_map.items()}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = np.nonzero(y_true != y_pred)[0]
    path = wrong_list_path(model, dataset, exp)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "filename", "session_id", "group_key",
                    "true_label", "pred_label"])
        for i in wrong:
            if files_df is not None and i < len(files_df):
                r = files_df.iloc[i]
                fn, sid, gk = r["filename"], r["session_id"], r["group_key"]
            else:
                fn, sid, gk = "-", "-", "-"
            w.writerow([int(i), fn, sid, gk,
                        inv.get(int(y_true[i]), y_true[i]),
                        inv.get(int(y_pred[i]), y_pred[i])])
    print(f"  [wrong_list] {len(wrong):,} / {len(y_true):,} → {path}")
    return path


# ═══════════════ result_table ═══════════════
def update_result_table(model, dataset, exp, best_acc):
    """result_table/{model}.csv  (dataset | exp1 | exp2 | exp3 | exp4) upsert."""
    if _NOSAVE:
        print(f"  [NM_NOSAVE] skipping result_table write ({dataset} exp{exp}={best_acc:.4f})")
        return
    d = _vroot("result_table")
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{model}.csv"
    cols = ["dataset", "exp1", "exp2", "exp3", "exp4"]
    if path.exists():
        df = pd.read_csv(path, dtype=str).reindex(columns=cols)
    else:
        df = pd.DataFrame(columns=cols)
    if dataset not in set(df["dataset"]):
        df = pd.concat([df, pd.DataFrame([{"dataset": dataset}])],
                       ignore_index=True)
    df.loc[df["dataset"] == dataset, f"exp{exp}"] = f"{best_acc:.4f}"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  [result_table] {path}  ({dataset} exp{exp} = {best_acc:.4f})")


# ═══════════════ data loading ═══════════════
def load_label_map(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npy_split(split_dir):
    """Load shaping npy outputs → (X, Y, files_df)"""
    split_dir = Path(split_dir)
    x = np.load(split_dir / "x_data.npy")
    y = np.load(split_dir / "y_data.npy")
    files = load_files_csv(split_dir)
    return x, y, files


def load_files_csv(split_dir):
    p = Path(split_dir) / "files.csv"
    if p.exists():
        return pd.read_csv(p, dtype=str, na_filter=False, encoding="utf-8")
    return None


# ═══════════════ shared torch loop (image-style: 2dcnn/yatc/netmamba) ═══════════════
def torch_train_loop(*, model, train_loader, test_loader, device, args,
                     defaults, files_test, label_map, optimizer=None,
                     scheduler=None, criterion=None, save_path=None):
    """
    epoch loop + evaluation every test_interval + best-param saving + wrong_list + result_table.
    scheduler steps per epoch. Returns: (best_acc, best_epoch)
    """
    import torch
    import torch.nn as nn

    epochs = args.epochs or defaults["epochs"]
    test_interval = args.test_interval or defaults.get("test_interval", 1)
    criterion = criterion or nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr or defaults["lr"])

    logger = ResultLogger(args.model, args.dataset, args.exp)

    def _save():
        if save_path is not None:
            torch.save(model.state_dict(), save_path)

    # test-peeking removed: use the parameters/accuracy of the 'last evaluated epoch' instead of best-over-epochs.
    last_acc, last_epoch = -1.0, -1
    last_true, last_pred = None, None

    def _evaluate():
        model.eval()
        tot_loss, tot_n, correct = 0.0, 0, 0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                tot_loss += loss.item() * len(yb)
                tot_n += len(yb)
                p = out.argmax(1)
                correct += (p == yb).sum().item()
                preds.append(p.cpu().numpy())
                trues.append(yb.cpu().numpy())
        return (tot_loss / max(tot_n, 1), correct / max(tot_n, 1),
                np.concatenate(trues) if trues else np.array([]),
                np.concatenate(preds) if preds else np.array([]))

    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss, tot_n, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            tot_n += len(yb)
            correct += (out.argmax(1) == yb).sum().item()
        train_loss = tot_loss / max(tot_n, 1)
        train_acc = correct / max(tot_n, 1)
        if scheduler is not None:
            scheduler.step()

        if epoch % test_interval == 0 or epoch == epochs:
            test_loss, test_acc, y_true, y_pred = _evaluate()
            last_acc, last_epoch = test_acc, epoch
            last_true, last_pred = y_true, y_pred
            logger.log(epoch, train_loss, train_acc, test_loss, test_acc)
            print(f"  epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f}  test_loss={test_loss:.4f} "
                  f"test_acc={test_acc:.4f}", flush=True)
        else:
            logger.log(epoch, train_loss, train_acc)
            print(f"  epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f}", flush=True)

    logger.close()
    _save()   # save last-epoch parameters (fixed rule, no peeking)
    if last_true is not None:
        save_wrong_list(args.model, args.dataset, args.exp,
                        files_test, last_true, last_pred,
                        label_map)
    update_result_table(args.model, args.dataset, args.exp, last_acc)
    print(f"  [last] epoch={last_epoch}  test_acc={last_acc:.4f}")
    return last_acc, last_epoch
