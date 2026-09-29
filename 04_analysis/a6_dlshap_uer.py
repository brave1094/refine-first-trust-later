#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a6_dlshap_uer.py — Illusion 5-B: token attribution extraction for the UER family (etbert/trafficformer).

Inputs are discrete tokens, so IG is applied in 'embedding space' (standard NLP practice):
  emb  = model.embedding(src, seg)                 ← same path as the training forward
  emb0 = model.embedding(src_pad, seg)             ← everything PAD(0) except [CLS]
  attr_token(i) = Σ_h | (emb−emb0) ⊙ (1/S)Σ_s ∇_emb f_c |  (summed over hidden dim)
  · f_c = its own argmax logit.  Everything after the encoder reproduces the training forward exactly.
  · tokenization = identical to _tokenize_row in train_etbert/_trafficformer
    ([CLS]+tokenize, seq truncation/0-padding, seg=1) — data.tsv is the same file too.

Output: shap_dl/{variant}/{model}/{ds}/exp{e}_attr.npy  (n, seq_len) float32
      shap_dl/{variant}/{model}/{ds}/exp{e}_pred.csv , meta.json
  ※ row order = top n rows of data.tsv = top n rows of files.csv → aligned with field_id.npy.
  ※ etbert seq=128 (bigram token i ↔ leading byte i), trafficformer seq=320 (5 packets×64).

Run (in each container):
  docker exec -w <repo>/03_model torch_etbert \
      python3 a6_dlshap_uer.py --model etbert --gpu 3
  docker exec -w <repo>/03_model torch_trafficformer \
      python3 a6_dlshap_uer.py --model trafficformer --gpu 3
  Options: --datasets vpn16,tor16  --n 500  --steps 16  --variant sizectrl
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent            # …/03_model/
WORK_DIR = CODE_DIR.parent
DATA = WORK_DIR / "01_dataset"
PARAM = RP.PARAM
DSS = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]
VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm",
              "strat": "00_filelist_strat"}

# Per-model constants — same as train_etbert.py / train_trafficformer.py
CONF = {
    "etbert": dict(root=RP.UPSTREAM / "04_etbert", seq=128,
                   vocab="models/encryptd_vocab.txt",
                   config="bert_base_config.json", dropout=0.5),
    "trafficformer": dict(root=RP.UPSTREAM / "07_trafficformer", seq=320,
                          vocab="models/encryptd_vocab.txt",
                          config="models/bert/base_config.json", dropout=0.1),
}


def build(model_name, labels_num, gpu):
    """Rebuild UER args + Classifier (same procedure as run() in train_*)."""
    c = CONF[model_name]
    sys.path.insert(0, str(c["root"]))
    from uer.layers import str2embedding            # noqa: F401
    from uer.encoders import str2encoder            # noqa: F401
    from uer.utils import str2tokenizer             # noqa: F401
    from uer.utils.config import load_hyperparam
    from uer.opts import finetune_opts

    helper = argparse.ArgumentParser()
    finetune_opts(helper)
    for action in helper._actions:
        action.required = False
    args = helper.parse_args([])
    args.vocab_path = str(c["root"] / c["vocab"])
    args.config_path = str(c["root"] / c["config"])
    args.seq_length = c["seq"]
    args.dropout = c["dropout"]
    args.labels_num = labels_num
    args.pooling = "first"
    args.embedding = "word_pos_seg"
    args.encoder = "transformer"
    args.mask = "fully_visible"
    args.soft_targets = False
    args.soft_alpha = 0.5
    # the trafficformer UER fork adds MoE args outside finetune_opts (run_classifier.py)
    # → inject the same defaults as train_trafficformer.py (harmless for etbert via hasattr)
    for k, v in dict(is_moe=False, vocab_size=None,
                     moebert_expert_dim=3072, moebert_expert_num=None,
                     moebert_route_method="hash-random",
                     moebert_route_hash_list=None,
                     moebert_load_balance=0.0, earlystop=5).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args = load_hyperparam(args)
    if not hasattr(args, "tokenizer") or isinstance(args.tokenizer, str):
        args.tokenizer = str2tokenizer["bert"](args)
    args.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    class Classifier(nn.Module):
        """Same structure as the Classifier in train_etbert/_trafficformer."""
        def __init__(self, args):
            super().__init__()
            self.embedding = str2embedding[args.embedding](
                args, len(args.tokenizer.vocab))
            self.encoder = str2encoder[args.encoder](args)
            self.labels_num = args.labels_num
            self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        def logits_from_emb(self, emb, seg):
            output = self.encoder(emb, seg)
            output = output[:, 0, :]
            output = torch.tanh(self.output_layer_1(output))
            return self.output_layer_2(output)

        def forward(self, src, seg):
            return self.logits_from_emb(self.embedding(src, seg), seg)

    return args, Classifier(args)


def stratified_idx(labels, n):
    """Select up to n rows by per-class round robin — guarantees every class is included.
    (data.tsv is sorted by class, so head(n) would drop the later classes entirely)"""
    y = np.asarray(labels)
    n = min(n, len(y))
    cls = {c: np.where(y == c)[0] for c in np.unique(y)}
    picked, i = [], 0
    while len(picked) < n:
        added = False
        for c in sorted(cls):
            if i < len(cls[c]):
                picked.append(int(cls[c][i]))
                added = True
                if len(picked) >= n:
                    break
        if not added:
            break
        i += 1
    return np.array(sorted(picked))


def tokenize_tsv(tsv_path, tokenizer, seq_length, n):
    """Reproduces _tokenize_row of train_*: [CLS]+tokens, truncation/0-padding, seg=1.
    Pass 1: read labels only to compute stratified indices → pass 2: tokenize only the selected rows."""
    CLS = "[CLS]"
    with open(tsv_path, encoding="utf-8") as f:
        rd = csv.reader(f, delimiter="\t", quotechar=None)
        header = next(rd)
        ci = {k: i for i, k in enumerate(header)}
        rows = list(rd)
    ridx = stratified_idx([int(r[ci["label"]]) for r in rows], n)
    keep = set(ridx.tolist())
    src_all, tgt_all = [], []
    for i in sorted(keep):
        row = rows[i]
        src = tokenizer.convert_tokens_to_ids(
            [CLS] + tokenizer.tokenize(row[ci["text_a"]]))
        src = src[:seq_length] + [0] * max(0, seq_length - len(src))
        src_all.append(src)
        tgt_all.append(int(row[ci["label"]]))
    return torch.LongTensor(src_all), torch.LongTensor(tgt_all), ridx


def gradient_shap_tokens(model, src, seg, nsamples, nref, batch, device,
                         noise=0.2, seed=42):
    """Embedding-space GradientSHAP (expected gradients). Returns |attr| (n, seq), pred, prob.
       baseline = embeddings of random reference sessions (background distribution). Computed with
       the own embeddings of exp1·exp3 (embeddings differ per model). Unlike IG with a fixed [CLS]+PAD baseline,
       the baseline is drawn from the real data distribution."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = src.size(0)
    # background embedding pool — nref random samples from src
    K = min(nref, n)
    bidx = torch.randperm(n, generator=g)[:K]
    bg_list = []
    with torch.no_grad():
        for j in range(0, K, batch):
            sb = src[bidx[j:j + batch]].to(device)
            gb = seg[bidx[j:j + batch]].to(device)
            bg_list.append(model.embedding(sb, gb).detach())
    bg = torch.cat(bg_list, 0)                          # (K, seq, hidden)
    estd = float(bg.reshape(K, -1).std().clamp_min(1e-6))
    attrs, preds, probs = [], [], []
    for i in range(0, n, batch):
        sb = src[i:i + batch].to(device)
        gb = seg[i:i + batch].to(device)
        B = sb.size(0)
        with torch.no_grad():
            emb = model.embedding(sb, gb)
            logit = model.logits_from_emb(emb, gb)
            p = torch.softmax(logit, dim=1)
            tgt = logit.argmax(1)
        preds.append(tgt.cpu().numpy())
        probs.append(p.max(1).values.cpu().numpy())
        acc = torch.zeros_like(emb)
        for _ in range(nsamples):
            ridx = torch.randint(0, K, (B,), generator=g).to(device)
            b = bg[ridx]                                # (B, seq, hidden)
            alpha = torch.rand((B, 1, 1), generator=g).to(device)
            ep = b + alpha * (emb - b)
            if noise:
                ep = ep + noise * estd * torch.randn(
                    emb.shape, generator=g).to(device)
            ep = ep.detach().requires_grad_(True)
            out = model.logits_from_emb(ep, gb)
            sel = out.gather(1, tgt.unsqueeze(1)).sum()
            grad, = torch.autograd.grad(sel, ep)
            acc += grad * (emb - b)
        attr = (acc / nsamples).abs().sum(-1)           # sum over hidden → (B, seq)
        attrs.append(attr.detach().cpu().numpy())
    return (np.concatenate(attrs).astype(np.float32),
            np.concatenate(preds), np.concatenate(probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["etbert", "trafficformer"])
    ap.add_argument("--datasets", default=",".join(DSS))
    ap.add_argument("--variant", default="sizectrl",
                    choices=["full", "sizectrl", "strat"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--nsamples", type=int, default=32,
                    help="GradientSHAP sample count (n encoder forwards — 32 for transformers)")
    ap.add_argument("--nref", type=int, default=64, help="number of background baselines")
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        lm_path = DATA / ds / VARIANT_FL[args.variant] / "label_map.json"
        lm = json.load(open(lm_path, encoding="utf-8"))
        uargs, model = build(args.model, len(lm), args.gpu)
        device = uargs.device
        tsv = DATA / ds / args.variant / args.model / "test_denoised" / "data.tsv"
        src, y, ridx = tokenize_tsv(tsv, uargs.tokenizer, uargs.seq_length, args.n)
        seg = torch.ones_like(src)
        outd = RP.SHAP_DL / args.variant / args.model / ds
        outd.mkdir(parents=True, exist_ok=True)
        np.save(outd / "rows_idx.npy", ridx)     # data.tsv(=files.csv) row numbers, for a7 alignment
        print(f"── {args.model} / {ds} / {args.variant}  n={len(y)} seq={uargs.seq_length}",
              flush=True)
        for exp in (1, 3):
            outp = outd / f"exp{exp}_attr.npy"
            if outp.exists() and not args.force:
                print(f"  [skip] exp{exp} — already exists (recompute with --force)")
                continue
            ck = PARAM / args.variant / args.model / ds / f"exp{exp}" / "best_model.pt"
            if not ck.exists():
                print(f"  [missing] {ck}")
                continue
            sd = torch.load(ck, map_location="cpu")
            sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
            model.load_state_dict(sd)
            model = model.to(device)
            attr, pred, prob = gradient_shap_tokens(
                model, src, seg, args.nsamples, args.nref, args.batch, device,
                noise=args.noise)
            np.save(outp, attr)
            with open(outd / f"exp{exp}_pred.csv", "w", encoding="utf-8") as f:
                f.write("row_idx,y_true,y_pred,prob\n")
                for i, (yt, yp, pb) in enumerate(zip(y.numpy(), pred, prob)):
                    f.write(f"{i},{yt},{yp},{pb:.4f}\n")
            acc = float((pred == y.numpy()).mean())
            print(f"  [saved] exp{exp}  attr{attr.shape}  pred_match={acc:.4f}",
                  flush=True)
            torch.cuda.empty_cache()
        json.dump(dict(model=args.model, dataset=ds, variant=args.variant,
                       n=int(len(y)), method="GradientSHAP", nsamples=args.nsamples,
                       nref=args.nref, seq=uargs.seq_length, split="test_denoised",
                       align="class-stratified (round robin) — rows_idx.npy holds files.csv row numbers"),
                  open(outd / "meta.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
