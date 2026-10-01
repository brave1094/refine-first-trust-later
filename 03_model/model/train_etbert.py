#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_etbert.py — ET-BERT (WWW 2022) fine-tuning adapter.

Input : 01_dataset/{ds}/etbert/{split_mode}/data.tsv (label \t text_a, with header)
Upstream : 00_assets/models/04_etbert/ (uer/, original repo)
Pretrained: 00_assets/models/04_etbert/models/pre-trained_model.bin
Vocab : 00_assets/models/04_etbert/models/encryptd_vocab.txt
Config: 00_assets/models/04_etbert/bert_base_config.json (part of the ET-BERT repository)

Default hyperparameters: epochs 10, batch 32, lr 2e-5, seq 128, warmup 0.1 (the article: run_pipeline.sh)
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import csv
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent           # …/03_model/model/
WORK_DIR = CODE_DIR.parent.parent                     # repository root
ETBERT_ROOT = RP.UPSTREAM / "04_etbert"

PRETRAINED_MODEL = str(ETBERT_ROOT / "models" / "pre-trained_model.bin")
VOCAB_PATH = str(ETBERT_ROOT / "models" / "encryptd_vocab.txt")
CONFIG_PATH = str(ETBERT_ROOT / "bert_base_config.json")

sys.path.insert(0, str(ETBERT_ROOT))
_ETBERT_IMPORT_ERROR = None
try:
    from uer.layers import *          # noqa: F401,F403
    from uer.encoders import *        # noqa: F401,F403
    from uer.utils.constants import *  # noqa: F401,F403
    from uer.utils import *           # noqa: F401,F403
    from uer.utils.optimizers import *  # noqa: F401,F403
    from uer.utils.config import load_hyperparam
    from uer.utils.seed import set_seed
    from uer.opts import finetune_opts
except ImportError as _e:
    _ETBERT_IMPORT_ERROR = _e

from lib.train_common import (ResultLogger, save_wrong_list,
                              update_result_table, load_files_csv)

DEFAULTS = dict(epochs=10, batch_size=32, lr=2e-5, test_interval=1,
                seq_length=128, warmup=0.1, dropout=0.5)


# ── Tokenize (multiprocessing) ────────────────────────────────────────────
_tokenizer_g = None
_seq_length_g = None


def _init_worker(tokenizer, seq_length):
    global _tokenizer_g, _seq_length_g
    _tokenizer_g = tokenizer
    _seq_length_g = seq_length


def _tokenize_row(row):
    tgt, text_a = row
    src = _tokenizer_g.convert_tokens_to_ids(
        ["[CLS]"] + _tokenizer_g.tokenize(text_a))
    if len(src) > _seq_length_g:
        src = src[:_seq_length_g]
    while len(src) < _seq_length_g:
        src.append(0)
    return (src, tgt, [1] * _seq_length_g)


def _read_dataset(args, tsv_path: Path, n_workers=8):
    from tqdm import tqdm as _tqdm
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append((int(row["label"]), row["text_a"]))
    dataset = []
    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(args.tokenizer, args.seq_length)) as pool:
        for r in _tqdm(pool.imap(_tokenize_row, rows, chunksize=256),
                       total=len(rows), desc="  Tokenizing", unit="sample",
                       dynamic_ncols=True, disable=None):   # disable the bar for files (non-TTY)
            dataset.append(r)
    return dataset


def _batch_loader(batch_size, src, tgt, seg):
    n = src.size(0)
    for i in range(n // batch_size):
        yield (src[i*batch_size:(i+1)*batch_size],
               tgt[i*batch_size:(i+1)*batch_size],
               seg[i*batch_size:(i+1)*batch_size])


def _evaluate(args, model, dataset):
    import torch
    model.eval()
    correct = total = 0
    tot_loss = 0.0
    all_pred, all_true = [], []
    batch_size = args.batch_size * 2
    src = torch.LongTensor([d[0] for d in dataset]).to(args.device)
    tgt = torch.LongTensor([d[1] for d in dataset]).to(args.device)
    seg = torch.LongTensor([d[2] for d in dataset]).to(args.device)
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            sb, tb, gb = src[i:i+batch_size], tgt[i:i+batch_size], seg[i:i+batch_size]
            loss, logits = model(sb, tb, gb)
            tot_loss += loss.item() * tb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == tb).sum().item()
            total += tb.size(0)
            all_pred.extend(pred.cpu().tolist())
            all_true.extend(tb.cpu().tolist())
    model.train()
    return (tot_loss / max(total, 1), correct / max(total, 1),
            all_true, all_pred)


def run(args):
    if _ETBERT_IMPORT_ERROR is not None:
        raise ImportError(
            f"00_assets/models/04_etbert/uer import failed ({_ETBERT_IMPORT_ERROR})")

    import torch
    import torch.nn as nn
    import tqdm as tqdm_module

    for fp in (VOCAB_PATH, CONFIG_PATH):
        if not Path(fp).exists():
            raise FileNotFoundError(
                f"required file missing: {fp}\nPlease run 00_setup_assets.sh.")
    # if the pretrained bin is missing, warn and use scratch (handled in load_or_initialize)

    epochs = args.epochs or DEFAULTS["epochs"]
    batch_size = args.batch_size or DEFAULTS["batch_size"]
    lr = args.lr or DEFAULTS["lr"]
    test_interval = args.test_interval or DEFAULTS["test_interval"]

    with open(args.label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    labels_num = len(class_names)

    param_dir = Path(args.param_dir)
    model_path = param_dir / "best_model.pt"

    # ── build standard ET-BERT args ──
    helper = argparse.ArgumentParser()
    finetune_opts(helper)
    for action in helper._actions:
        action.required = False
    for k, v in vars(helper.parse_args([])).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args.pretrained_model_path = PRETRAINED_MODEL
    args.vocab_path = VOCAB_PATH
    args.config_path = CONFIG_PATH
    args.output_model_path = str(model_path)
    args.seq_length = DEFAULTS["seq_length"]
    args.learning_rate = lr
    args.warmup = DEFAULTS["warmup"]
    args.dropout = DEFAULTS["dropout"]
    args.batch_size = batch_size
    args.epochs_num = epochs
    args.labels_num = labels_num
    args.pooling = "first"
    args.soft_targets = False
    args.soft_alpha = 0.5
    set_seed(args.seed)
    args = load_hyperparam(args)
    if not hasattr(args, "tokenizer") or isinstance(args.tokenizer, str):
        args.tokenizer = str2tokenizer["bert"](args)
    args.device = torch.device(f"cuda:{args.gpu}"
                               if torch.cuda.is_available() else "cpu")

    print(f"  device={args.device} epochs={epochs} batch={batch_size} "
          f"lr={lr} seq={args.seq_length} classes={labels_num}")

    class Classifier(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embedding = str2embedding[args.embedding](
                args, len(args.tokenizer.vocab))
            self.encoder = str2encoder[args.encoder](args)
            self.labels_num = args.labels_num
            self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        def forward(self, src, tgt, seg):
            emb = self.embedding(src, seg)
            output = self.encoder(emb, seg)
            output = output[:, 0, :]
            output = torch.tanh(self.output_layer_1(output))
            logits = self.output_layer_2(output)
            if tgt is not None:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
                return loss, logits
            return None, logits

    model = Classifier(args).to(args.device)
    if os.path.exists(args.pretrained_model_path):
        msg = model.load_state_dict(
            torch.load(args.pretrained_model_path,
                       map_location=str(args.device)), strict=False)
        print(f"  pretrained loaded (missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)})")
    else:
        for n, p in model.named_parameters():
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
        print("  [warning] no pretrained → scratch")

    # ── Data ──
    train_path = Path(args.train_dataset_dir) / "data.tsv"
    test_path = Path(args.test_dataset_dir) / "data.tsv"
    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"TSV missing: {p} — run 06_make_dataset.py first")
    if os.environ.get("A3_EXTRACT"):        # Illusion 3 embedding extraction (guard): dump penultimate instead of training
        import numpy as _np
        if model_path.exists():
            model.load_state_dict(torch.load(str(model_path),
                                             map_location=args.device), strict=False)
        model.eval()
        _ts = _read_dataset(args, test_path)
        _src = torch.LongTensor([d[0] for d in _ts]).to(args.device)
        _seg = torch.LongTensor([d[2] for d in _ts]).to(args.device)
        _y = _np.array([d[1] for d in _ts])
        _buf = {}
        _h = model.output_layer_2.register_forward_pre_hook(
            lambda _m, i: _buf.__setitem__("x", i[0].detach()))
        _f = []
        with torch.no_grad():
            for _i in range(0, len(_ts), 256):
                model(_src[_i:_i+256], None, _seg[_i:_i+256])
                _f.append(_buf["x"].reshape(_buf["x"].size(0), -1).cpu().numpy())
        _h.remove()
        sys.path.insert(0, str(RP.ANALYSIS))
        import a3_metrics
        _emb = _np.concatenate(_f)
        if os.environ.get("A3_TSNE"):
            a3_metrics.save_tsne_coords(
                str(RP.EMB_DIR / "a3_tsne" / os.environ.get("SCIE_VARIANT", "sizectrl") / args.dataset
                    / f"{args.model}_exp{args.exp}.csv"), _emb, _y)
            print(f"[A3-tsne] {args.model}/{args.dataset} exp{args.exp}")
        else:
            _sil, _sep = a3_metrics.append_long(str(RP.EMB_DIR / f"a3_emb_long_{os.environ.get('SCIE_VARIANT', 'sizectrl')}.csv"),
                                                args.model, args.dataset, args.exp,
                                                "test_denoised", _emb, _y)
            print(f"[A3] {args.model}/{args.dataset} exp{args.exp} sil={_sil:.3f} sep={_sep:.2f}")
        return
    trainset = _read_dataset(args, train_path)
    random.shuffle(trainset)
    testset = _read_dataset(args, test_path)
    files_te = load_files_csv(args.test_dataset_dir)
    print(f"  train={len(trainset):,}  test={len(testset):,}")

    src = torch.LongTensor([d[0] for d in trainset])
    tgt = torch.LongTensor([d[1] for d in trainset])
    seg = torch.LongTensor([d[2] for d in trainset])

    args.train_steps = int(len(trainset) * epochs / batch_size) + 1

    no_decay = ["bias", "gamma", "beta"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay_rate": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay_rate": 0.0},
    ]
    optimizer = str2optimizer["adamw"](grouped, lr=lr, correct_bias=False)
    scheduler = str2scheduler["linear"](
        optimizer, args.train_steps * args.warmup, args.train_steps)

    logger = ResultLogger(args.model, args.dataset, args.exp)
    best_acc, best_epoch = -1.0, -1
    best_true = best_pred = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = correct = total = 0
        for sb, tb, gb in tqdm_module.tqdm(
                _batch_loader(batch_size, src, tgt, seg),
                total=len(trainset) // batch_size,
                desc=f"Epoch {epoch:>3}", leave=False, disable=None):
            sb, tb, gb = (sb.to(args.device), tb.to(args.device),
                          gb.to(args.device))
            optimizer.zero_grad()
            loss, logits = model(sb, tb, gb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * tb.size(0)
            correct += (logits.argmax(-1) == tb).sum().item()
            total += tb.size(0)
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        if epoch % test_interval == 0 or epoch == epochs:
            test_loss, test_acc, y_true, y_pred = _evaluate(args, model, testset)
            # no test-peeking: use the values/params of the last evaluated epoch
            best_acc, best_epoch = test_acc, epoch
            best_true, best_pred = y_true, y_pred
            state = (model.module.state_dict()
                     if hasattr(model, "module") else model.state_dict())
            torch.save(state, model_path)
            logger.log(epoch, train_loss, train_acc, test_loss, test_acc)
            print(f"[Epoch {epoch:>3}/{epochs}] train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                  f"({time.time()-t0:.1f}s)")
        else:
            logger.log(epoch, train_loss, train_acc)
            print(f"[Epoch {epoch:>3}/{epochs}] train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} ({time.time()-t0:.1f}s)")

    logger.close()
    if best_true is not None:
        save_wrong_list(args.model, args.dataset, args.exp,
                        files_te, best_true, best_pred, label_map)
    update_result_table(args.model, args.dataset, args.exp, best_acc)
    print(f"\n  last: epoch={best_epoch} test_acc={best_acc:.4f} → {model_path}")
