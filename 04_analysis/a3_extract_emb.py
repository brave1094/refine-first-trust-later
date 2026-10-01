#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a3_extract_emb.py — Illusion 3 (class-boundary collapse): penultimate-embedding metrics of the 3 npy models (2dcnn/yatc/netmamba).
  Passes the same test_denoised through exp1 (refined training) and exp3 (noisy training) models → appends silhouette/sep
  to the long format (a3_emb_long.csv). (etbert/trafficformer use 01_train.py + A3_EXTRACT=1)
  Run: python3 a3_extract_emb.py --gpu 0 --models 2dcnn
        docker exec ... torch_yatc python3 a3_extract_emb.py --gpu 0 --models yatc
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import sys
import argparse
import importlib.util

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, str(RP.MODEL))                       # lib.train_common lives in 03_model
from lib.train_common import load_label_map, load_npy_split
import a3_metrics

DATA = os.path.join(WORK, "01_dataset")
PARAM = str(RP.PARAM)
CODES = str(RP.UPSTREAM)
VARIANT = os.environ.get("SCIE_VARIANT", "sizectrl")  # full|sizectrl|strat; the article uses sizectrl
VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}[VARIANT]
LONG = os.path.join(str(RP.EMB_DIR), f"a3_emb_long_{VARIANT}.csv")
NPY_MODELS = ["2dcnn", "yatc", "netmamba"]
DSS = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]


def _spec(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(model_name, num_classes):
    if model_name == "2dcnn":
        return _spec("_cnn2d", os.path.join(str(RP.OWN_MODELS), "03_2dcnn", "model.py")).CNN2D(
            num_classes=num_classes)
    if model_name == "yatc":
        root = os.path.join(CODES, "05_yatc"); sys.path.insert(0, root)
        return _spec("_yatc", os.path.join(root, "models_YaTC.py")).TraFormer_YaTC(
            num_classes=num_classes, drop_path_rate=0.1)
    if model_name == "netmamba":
        root = os.path.join(CODES, "06_netmamba")
        for p in (root, os.path.join(root, "src")):
            sys.path.insert(0, p)
        return _spec("_nm", os.path.join(root, "src", "models_net_mamba.py")).net_mamba_classifier(
            num_classes=num_classes)
    raise ValueError(model_name)


def prep(model_name, X):
    t = torch.from_numpy(X).float()
    return t if model_name == "2dcnn" else t.div(255).sub(0.5).div(0.5).unsqueeze(1)


@torch.no_grad()
def embed(model_name, ds, exp, split, device):
    lm = load_label_map(os.path.join(DATA, ds, VARIANT_FL, "label_map.json"))
    X, Y, _ = load_npy_split(os.path.join(DATA, ds, VARIANT, model_name, split))
    ckpt = os.path.join(PARAM, VARIANT, model_name, ds, f"exp{exp}", "best_model.pt")
    if not os.path.exists(ckpt):
        return None, None
    model = build_model(model_name, len(lm))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval().to(device)
    last_lin = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    buf = {}
    h = last_lin.register_forward_pre_hook(lambda _m, i: buf.__setitem__("x", i[0].detach()))
    Xt = prep(model_name, X)
    feats = []
    for i in range(0, len(Xt), 256):
        model(Xt[i:i+256].to(device))
        feats.append(buf["x"].reshape(buf["x"].size(0), -1).cpu().numpy())
    h.remove()
    return np.concatenate(feats), Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="test_denoised")
    ap.add_argument("--models", nargs="+", default=NPY_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DSS, help="Restrict the datasets to process")
    ap.add_argument("--tsne", action="store_true",
                    help="Save t-SNE coordinates (skips the long-CSV append in this mode → avoids duplicates)")
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for m in args.models:
        for ds in args.datasets:
            for e in (1, 3):
                emb, y = embed(m, ds, e, args.split, dev)
                if emb is None:
                    continue
                if args.tsne:                       # t-SNE coordinates (for figures)
                    a3_metrics.save_tsne_coords(
                        os.path.join(str(RP.EMB_DIR), "a3_tsne", VARIANT, ds, f"{m}_exp{e}.csv"), emb, y)
                # Quantitative metrics (silhouette/sep) are always recorded on the original embeddings
                sil, sep = a3_metrics.append_long(LONG, m, ds, e, args.split, emb, y)
                print(f"  {m:<10} {ds:<9} exp{e}  sil={sil:.3f} sep={sep:.2f}")
    print("[a3] done")


if __name__ == "__main__":
    main()
