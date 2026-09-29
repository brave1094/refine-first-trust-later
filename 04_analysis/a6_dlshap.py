#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a6_dlshap.py — Illusion 5-B: input attribution extraction for DL byte models (2dcnn/yatc/netmamba).

Method: GradientSHAP (= gradient approximation of SHAP, expected gradients — pure torch)
  attr_i(x) = E_{b~background, α~U(0,1)} [ (x_i − b_i) · ∂f_c/∂x_i |_{x'=b+α(x−b)+ε} ]
  · Standard SHAP approximation: baselines are drawn from the data distribution (background) with random interpolation.
    (IG uses a fixed baseline and uniform interpolation, a different method — hence the switch to GradientSHAP)
  · background = --nref random sessions from the same test_denoised (the actual input distribution
    of each refined/noise model). ε = --noise·std Gaussian (same as captum GradientShap default behavior).
  · f_c = logit of the predicted class (argmax) of the model itself  (each of exp1·exp3 uses its own prediction)
  · The input for both exp1/exp3 is 'the same test_denoised'.

Output: shap_dl/{variant}/{model}/{ds}/exp{e}_attr.npy   (n, L) float32, |GradientSHAP|
      shap_dl/{variant}/{model}/{ds}/exp{e}_pred.csv   (row_idx, y_true, y_pred, prob)
      shap_dl/{variant}/{model}/{ds}/meta.json
  ※ row order of (n,L) = top n rows of test_denoised files.csv → aligned with field_id.npy from 08_make_field_map.
  ※ If old IG results exist, overwrite with --force (recomputation needed since the method changed).

Run (inside each model container):
  docker exec -w <repo>/03_model torch_cnn      python3 a6_dlshap.py --model 2dcnn    --gpu 0 --force
  docker exec -w <repo>/03_model torch_yatc     python3 a6_dlshap.py --model yatc     --gpu 1 --force
  docker exec -w <repo>/03_model torch_netmamba python3 a6_dlshap.py --model netmamba --gpu 2 --force
  Options: --datasets vpn16,tor16  --n 500  --nsamples 64  --nref 100  --variant sizectrl
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import os
import sys
import json
import argparse
import importlib.util

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from lib.train_common import load_label_map, load_npy_split, load_files_csv  # noqa: E402

DATA = os.path.join(WORK, "01_dataset")
PARAM = str(RP.PARAM)
CODES = str(RP.UPSTREAM)
DSS = ["vpn16", "tor16", "tls1.3", "cispec", 
       "ustc16", "cic17", "cic18", "iot23"]
VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm",
              "strat": "00_filelist_strat"}
MASK2D = 257.0 / 256.0          # 2dcnn padding (after normalization) — verified to match the empirical mode of x_data


def _spec(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(model_name, num_classes):
    """Same as a3_extract_emb.build_model — reconstructs the same class as in training."""
    if model_name == "2dcnn":
        return _spec("_cnn2d", os.path.join(str(RP.OWN_MODELS), "03_2dcnn", "model.py")).CNN2D(
            num_classes=num_classes)
    if model_name == "yatc":
        root = os.path.join(CODES, "05_yatc")
        sys.path.insert(0, root)
        return _spec("_yatc", os.path.join(root, "models_YaTC.py")).TraFormer_YaTC(
            num_classes=num_classes, drop_path_rate=0.1)
    if model_name == "netmamba":
        root = os.path.join(CODES, "06_netmamba")
        for p in (root, os.path.join(root, "src")):
            sys.path.insert(0, p)
        return _spec("_nm", os.path.join(root, "src", "models_net_mamba.py")
                     ).net_mamba_classifier(num_classes=num_classes)
    raise ValueError(model_name)


def prep(model_name, X):
    """Same preprocessing as training (train_*). Returns (tensor, baseline_tensor)."""
    t = torch.from_numpy(X).float()
    if model_name == "2dcnn":
        return t, torch.full_like(t, MASK2D)
    t = t.div(255).sub(0.5).div(0.5).unsqueeze(1)      # (N,1,40,40)
    return t, torch.full_like(t, -1.0)


def gradient_shap(model, x, bg, nsamples, batch, device, noise=0.2, seed=42):
    """GradientSHAP (expected gradients). Returns: |attr| (n, L), pred, prob.
       x  : (N, ...) prepped input
       bg : (K, ...) prepped background (baseline distribution)
       nsamples times per sample: random baseline b from background, α~U(0,1),
       x' = b + α(x−b) + ε(Gaussian): take the gradient there, multiply by (x−b), and average."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    K = bg.shape[0]
    xstd = float(x.reshape(len(x), -1).std().clamp_min(1e-6))
    n = x.shape[0]
    attrs, preds, probs = [], [], []
    for i in range(0, n, batch):
        xb = x[i:i + batch].to(device)
        B = len(xb)
        with torch.no_grad():
            logit = model(xb)
            p = torch.softmax(logit, dim=1)
            tgt = logit.argmax(1)
        preds.append(tgt.cpu().numpy())
        probs.append(p.max(1).values.cpu().numpy())
        acc = torch.zeros_like(xb)
        vshape = [B] + [1] * (xb.dim() - 1)
        for _ in range(nsamples):
            ridx = torch.randint(0, K, (B,), generator=g)
            b = bg[ridx].to(device)
            alpha = torch.rand(vshape, generator=g).to(device)
            xp = b + alpha * (xb - b)
            if noise:
                xp = xp + noise * xstd * torch.randn(
                    xb.shape, generator=g).to(device)
            xp.requires_grad_(True)
            out = model(xp)
            sel = out.gather(1, tgt.unsqueeze(1)).sum()
            grad, = torch.autograd.grad(sel, xp)
            acc += grad * (xb - b)
        attr = (acc / nsamples).abs().reshape(B, -1)
        attrs.append(attr.detach().cpu().numpy())
    return (np.concatenate(attrs).astype(np.float32),
            np.concatenate(preds), np.concatenate(probs))


def stratified_idx(Y, n):
    """Select up to n rows by per-class round-robin — guarantees all classes are included.
    (files.csv is sorted by class, so head(n) drops trailing classes entirely)"""
    n = min(n, len(Y))
    cls = {c: np.where(Y == c)[0] for c in np.unique(Y)}
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


def run_one(model_name, ds, variant, n, nsamples, nref, batch, device,
            noise, force):
    lm = load_label_map(os.path.join(DATA, ds, VARIANT_FL[variant], "label_map.json"))
    split_dir = os.path.join(DATA, ds, variant, model_name, "test_denoised")
    X, Y, files = load_npy_split(split_dir)
    ridx = stratified_idx(Y, n)
    Xs, Ys = X[ridx], Y[ridx]
    xt, _ = prep(model_name, Xs)
    # background = nref random samples from the entire test_denoised (baseline distribution)
    rng = np.random.default_rng(42)
    bidx = rng.permutation(len(X))[:min(nref, len(X))]
    bg, _ = prep(model_name, X[bidx])
    outd = os.path.join(str(RP.SHAP_DL), variant, model_name, ds)
    os.makedirs(outd, exist_ok=True)
    np.save(os.path.join(outd, "rows_idx.npy"), ridx)   # files.csv row indices (for a7 alignment)
    done_any = False
    for exp in (1, 3):
        ck = os.path.join(PARAM, variant, model_name, ds, f"exp{exp}", "best_model.pt")
        outp = os.path.join(outd, f"exp{exp}_attr.npy")
        if os.path.exists(outp) and not force:
            print(f"  [skip] {model_name}/{ds} exp{exp} — already exists (recompute with --force)")
            continue
        if not os.path.exists(ck):
            print(f"  [missing] {ck}")
            continue
        model = build_model(model_name, len(lm))
        sd = torch.load(ck, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd)
        model = model.to(device)
        attr, pred, prob = gradient_shap(model, xt, bg, nsamples, batch,
                                         device, noise=noise)
        np.save(outp, attr)
        with open(os.path.join(outd, f"exp{exp}_pred.csv"), "w", encoding="utf-8") as f:
            f.write("row_idx,y_true,y_pred,prob\n")
            for i, (yt, yp, pb) in enumerate(zip(Ys, pred, prob)):
                f.write(f"{i},{yt},{yp},{pb:.4f}\n")
        acc = float((pred == Ys).mean())
        print(f"  [saved] {model_name}/{ds} exp{exp}  attr{attr.shape}  "
              f"pred_agreement={acc:.4f}", flush=True)
        done_any = True
        del model
        torch.cuda.empty_cache()
    if done_any or not os.path.exists(os.path.join(outd, "meta.json")):
        meta = dict(model=model_name, dataset=ds, variant=variant, n=int(len(Ys)),
                    method="GradientSHAP", nsamples=nsamples, nref=int(len(bidx)),
                    noise=noise, split="test_denoised",
                    align="class-stratified (round-robin) — rows_idx.npy holds files.csv row indices")
        json.dump(meta, open(os.path.join(outd, "meta.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["2dcnn", "yatc", "netmamba"])
    ap.add_argument("--datasets", default=",".join(DSS))
    ap.add_argument("--variant", default="sizectrl",
                    choices=["full", "sizectrl", "strat"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=500,
                    help="upper bound on the number of sessions to explain (class-stratified)")
    ap.add_argument("--nsamples", type=int, default=64,
                    help="number of GradientSHAP expected-gradients samples")
    ap.add_argument("--nref", type=int, default=100, help="number of background baselines")
    ap.add_argument("--noise", type=float, default=0.2,
                    help="input Gaussian noise scale (similar to captum GradientShap default)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.model == "netmamba" and not torch.cuda.is_available():
        print("[warning] netmamba requires mamba_ssm CUDA kernels — may not run on CPU")
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"── {args.model} / {ds} / {args.variant}  "
              f"(n≤{args.n}, nsamples={args.nsamples}, nref={args.nref})", flush=True)
        run_one(args.model, ds, args.variant, args.n, args.nsamples, args.nref,
                args.batch, device, args.noise, args.force)


if __name__ == "__main__":
    main()
