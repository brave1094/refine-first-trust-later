#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a3_extract_emb_uer.py — Illusion 3 (class-boundary collapse) extension to the UER family: etbert / trafficformer.

  a3_extract_emb.py handles only the 3 npy models (2dcnn/yatc/netmamba). This script
  fills in the other 2 models with the same convention, completing a3_emb_long_{variant}.csv with 5 models.

  The same test_denoised sessions are passed through the exp1 (refined training)·exp3 (noisy training) models
  to extract penultimate embeddings and record silhouette / davies_bouldin / sep_ratio.
  Since the inputs are identical, the difference between the two is purely due to the 'learned representation'.

  penultimate = tanh(output_layer_1(encoder_out[:, 0, :]))   ← right before output_layer_2
    This is the same path as the Classifier in train_etbert / train_trafficformer,
    and matches the approach a3_extract_emb.py uses for npy models (forward_pre_hook on the last Linear).

  Model construction·tokenization import build() / tokenize_tsv() from a6_dlshap_uer.py as is.
  (A reimplementation with even slightly different tokenization breaks comparability with the 3 models — always reuse.)

  The --n default is 'all'. tokenize_tsv returns all rows in original order when n >= row count, so
  conditions match the 3 npy models (all → a3_metrics.compute applies the same sampling with cap=3000).

Run (in each container):
  docker exec -w <repo>/03_model torch_etbert \
      python3 a3_extract_emb_uer.py --model etbert --gpu 2 --tsne
  docker exec -w <repo>/03_model torch_trafficformer \
      python3 a3_extract_emb_uer.py --model trafficformer --gpu 3 --tsne
  Options: --datasets vpn16,tor16   --variant sizectrl   --n 3000   --batch 64
  ※ (model,dataset,exp,split) already in the long CSV are skipped → no duplicate append.
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a3_metrics                                            # noqa: E402
from a6_dlshap_uer import (build, tokenize_tsv, DSS,          # noqa: E402
                           VARIANT_FL, DATA, PARAM)


@torch.no_grad()
def penultimate(model, src, seg, device, batch):
    """Collect the representation right before output_layer_2 (n, hidden) in batches."""
    model.eval()
    buf, out = {}, []
    h = model.output_layer_2.register_forward_pre_hook(
        lambda _m, i: buf.__setitem__("x", i[0].detach()))
    try:
        for i in range(0, src.size(0), batch):
            model(src[i:i + batch].to(device), seg[i:i + batch].to(device))
            x = buf["x"]
            out.append(x.reshape(x.size(0), -1).float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out)


def load_done(long_path, split):
    """Set of (model,dataset,exp) already in the long CSV — prevents duplicates on rerun."""
    done = set()
    if long_path.exists():
        for r in csv.DictReader(open(long_path, encoding="utf-8-sig")):
            if r.get("split") == split:
                done.add((r["model"], r["dataset"], str(r["exp"])))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["etbert", "trafficformer"])
    ap.add_argument("--datasets", default=",".join(DSS))
    ap.add_argument("--variant", default="sizectrl",
                    choices=["full", "sizectrl", "strat"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="test_denoised")
    ap.add_argument("--n", type=int, default=10 ** 9,
                    help="max number of sessions (default=all). all is recommended for the same conditions as the 3 npy models")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tsne", action="store_true",
                    help="also save t-SNE coordinates (a3_tsne/{variant}/{ds}/{model}_exp{e}.csv)")
    args = ap.parse_args()

    long_path = RP.EMB_DIR / f"a3_emb_long_{args.variant}.csv"
    done = load_done(long_path, args.split)
    tsne_dir = HERE / "a3_tsne" / args.variant

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        need_long = [e for e in (1, 3) if (args.model, ds, str(e)) not in done]
        need_tsne = args.tsne and not all(
            (tsne_dir / ds / f"{args.model}_exp{e}.csv").exists() for e in (1, 3))
        if not need_long and not need_tsne:
            print(f"[skip] {ds} — long CSV·t-SNE both exist", flush=True)
            continue

        lm_path = DATA / ds / VARIANT_FL[args.variant] / "label_map.json"
        tsv = DATA / ds / args.variant / args.model / args.split / "data.tsv"
        if not lm_path.exists() or not tsv.exists():
            print(f"[missing] {tsv if lm_path.exists() else lm_path}", flush=True)
            continue

        lm = json.load(open(lm_path, encoding="utf-8"))
        uargs, model = build(args.model, len(lm), args.gpu)
        device = uargs.device
        src, y, _ = tokenize_tsv(tsv, uargs.tokenizer, uargs.seq_length, args.n)
        seg = torch.ones_like(src)
        yv = y.numpy()
        print(f"── {args.model} / {ds} / {args.variant}  "
              f"n={len(yv)} cls={len(set(yv.tolist()))} seq={uargs.seq_length}",
              flush=True)

        for e in (1, 3):
            ck = PARAM / args.variant / args.model / ds / f"exp{e}" / "best_model.pt"
            if not ck.exists():
                print(f"  [missing] {ck}", flush=True)
                continue
            sd = torch.load(ck, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            model.load_state_dict(sd)
            model.to(device)

            emb = penultimate(model, src, seg, device, args.batch)

            if (args.model, ds, str(e)) not in done:
                sil, sep = a3_metrics.append_long(
                    str(long_path), args.model, ds, e, args.split, emb, yv)
                done.add((args.model, ds, str(e)))
                print(f"  [saved] exp{e}  dim={emb.shape[1]}  "
                      f"sil={sil:+.3f}  sep={sep:.2f}", flush=True)
            else:
                print(f"  [exists] exp{e} — long CSV kept", flush=True)

            if args.tsne:
                p = tsne_dir / ds / f"{args.model}_exp{e}.csv"
                if not p.exists():
                    a3_metrics.save_tsne_coords(str(p), emb, yv)
                    print(f"  [t-SNE] {p.name}", flush=True)

            del emb
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    print(f"[done] {args.model} → {long_path}", flush=True)


if __name__ == "__main__":
    main()
