#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_make_dataset.py
─────────────────────────────────────────────────────────────────────────────
filelist (05 output, 00_filelist) → build final per-model training datasets (router).

A single run builds all splits used by the 4 experiment groups:
  train_noisy / train_denoised / test_noisy / test_denoised
(exp1~4 are formed by 03_model/01_train.py by cross-combining these 4)

8 models:
  rf / xgboost          : 1,205 statistical features (03_model/own_models/02_xgboost/extractor.py, direct pcap parsing)
                          rf uses the same features as xgboost → reuses (copies) the xgboost output if present
  2dcnn / etbert / yatc / netmamba / trafficformer / mm4flow
                        : dpkt parser (lib/parser) + per-model shaping (lib/shaping) path

Extraction (extract):
  --parser {dpkt,tshark} : packet parser (dpkt implemented, tshark reserved)
  --extract-mode {auto,whole,session} :
      auto    = dataset default (ustc16/cic17/cic18/iot23=whole, others=session)
      session = individual session files in 02_session
      whole   = extract {proto}.stream=={stream} from the whole pcap in 01_pcap (tshark stream map)
  L2 is always removed. Both IPv4/IPv6 are supported.

Masking (byte-based models):
  --ip_mask / --port_mask / --l3_mask(entire IP header) / --l4_mask(entire L4 header)

Transformation (shaping): 02_preprocess/lib/shaping/shaping_{model}.py (saved in the
  shape/format (npy/tsv/csv.gz) each model expects. See the docstring at the top of each file)

Input : 01_dataset/{dataset}/00_filelist/list_{train|test}_{noisy|denoised}.csv
       01_dataset/{dataset}/00_filelist/label_map.json
Output : 01_dataset/{dataset}/{model}/{train_noisy|train_denoised|test_noisy|test_denoised}/...

Usage:
  python3 06_make_dataset.py --dataset vpn16 --model xgboost --workers 16
  python3 06_make_dataset.py --dataset vpn16 --model 2dcnn --ip_mask --port_mask --workers 16
  python3 06_make_dataset.py --dataset ustc16 --model yatc --ip_mask --port_mask --workers 8
  python3 06_make_dataset.py --dataset vpn16 --model trafficformer --tf_enhance 5 --workers 16
  python3 06_make_dataset.py --dataset all --model netmamba --ip_mask --workers 8
"""
import sys as _sys, pathlib as _pl  # noqa: E401  (repository layout: repo_paths.py at the root)
_sys.path.insert(0, str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "repo_paths.py").exists())))
import repo_paths as RP  # noqa: E402
import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred")

CODE_DIR = Path(__file__).resolve().parent            # …/02_preprocess/
WORK_DIR = CODE_DIR.parent                             # repository root
LIST_BASE = WORK_DIR / "01_dataset"
MODEL_CODE_BASE = RP.OWN_MODELS

sys.path.insert(0, str(CODE_DIR))
from lib import datasets as ds                         # noqa: E402
from lib.parser import pcap_source                     # noqa: E402
from lib.shaping import ALL_MODELS, BYTE_MODELS, FEAT_MODELS, NETFOUND_MODELS  # noqa: E402

SPLIT_MODES = ["train_noisy", "train_denoised", "test_noisy", "test_denoised"]

# variant: full=non-sizectrl (00_filelist→full/), sizectrl=07 output (00_filelist_sm→sizectrl/)
VARIANT_FL = {"full": "00_filelist", "sizectrl": "00_filelist_sm", "strat": "00_filelist_strat"}

DATASET_EXTRACT_MODE = {
    "ustc16": "whole", "cic17": "whole", "cic18": "whole",
    "iot23": "whole", 
    "vpn16": "session", "tor16": "session", "tls1.3": "session",
    "cispec": "session", 
}


def _load_extractor():
    path = MODEL_CODE_BASE / "02_xgboost" / "extractor.py"
    spec = importlib.util.spec_from_file_location("_xgb_extractor", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_xgb_extractor"] = mod
    spec.loader.exec_module(mod)
    return mod


def _list_path(fl_dir: Path, split_mode: str) -> Path:
    split, mode = split_mode.split("_", 1)             # train_noisy → train, noisy
    return fl_dir / f"list_{split}_{mode}.csv"


def _now():
    # the datasets server environment has a clock. On failure, harmlessly an empty string.
    try:
        import datetime as _dt
        return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _save_config(model_dir: Path, args, dataset: str, extract_mode: str,
                 masked: bool):
    """Record the build settings in the model directory (01_dataset/{ds}/{model}/).
    For later tracing of 'which masking this dataset was built with'."""
    cfg = {
        "dataset": dataset,
        "model": args.model,
        "created": _now(),
        "extract": {
            "parser": getattr(args, "parser", "dpkt"),
            "extract_mode": extract_mode,
        },
        "mask": {
            "ip_mask": bool(args.ip_mask),
            "port_mask": bool(args.port_mask),
            "l3_mask": bool(args.l3_mask),
            "l4_mask": bool(args.l4_mask),
            # human-readable summary: none / ip+port / l3+l4 / ...
            "summary": (
                "+".join(k for k, v in [
                    ("l3", args.l3_mask), ("l4", args.l4_mask),
                    ("ip", args.ip_mask and not args.l3_mask),
                    ("port", args.port_mask and not args.l4_mask),
                ] if v) or "none"
            ),
            "applied": bool(masked),   # feat models (rf/xgboost) do not apply the byte mask → False
        },
        "splits": list(SPLIT_MODES),
        "model_opts": {
            "feat": getattr(args, "feat", None) if args.model in FEAT_MODELS else None,
            "tf_enhance": getattr(args, "tf_enhance", None) if args.model == "trafficformer" else None,
            "tf_randomize": (not getattr(args, "tf_no_randomize", False)) if args.model == "trafficformer" else None,
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dataset_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def build_feat_model(args, dataset: str, fl_dir: Path, out_base: Path,
                     label_map: dict, splits: list):
    """rf / xgboost : 1,205-feature extraction (extractor path). out_base=…/{ds}/{variant}."""
    extract_mode = (args.extract_mode if args.extract_mode != "auto"
                    else DATASET_EXTRACT_MODE.get(dataset, "session"))
    if any([args.ip_mask, args.port_mask, args.l3_mask, args.l4_mask]):
        print("  [note] rf/xgboost do not apply the byte mask — feature-category ablation is "
              "controlled at training time via --featset")

    # Reuse is 'one-way' only: rf copies the xgboost output, xgboost always extracts directly.
    #   (if two-way, xgboost could copy a stale old rf file, causing contamination)
    other = "xgboost" if args.model == "rf" else None
    ex = None
    feat_mod = None
    for split_mode in splits:
        out_dir = out_base / args.model / split_mode
        out_csv = out_dir / "features.csv"
        # rf only: reuse the xgboost output if present (only within the same variant)
        peer = (out_base / other / split_mode / "features.csv"
                if other else None)
        if peer is not None and peer.exists() and not args.force:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(peer, out_csv)
            pj = peer.parent / "feature_columns.json"
            if pj.exists():
                shutil.copyfile(pj, out_dir / "feature_columns.json")
            print(f"  [{split_mode}] reusing {other} output → {out_csv}")
            continue

        list_csv = _list_path(fl_dir, split_mode)
        if not list_csv.exists():
            print(f"  [{split_mode}] no filelist → skipped ({list_csv.name})")
            continue
        if ex is None:
            ex = _load_extractor()
            feat_mod = ex.load_feat_module(args.feat)
        out_dir.mkdir(parents=True, exist_ok=True)
        roots = ([ds.pcap_dir(dataset)] if extract_mode == "whole"
                 else [ds.session_dir(dataset)])
        ex.extract_list(feat_mod, args.feat, list_csv, roots, out_csv,
                        args.workers, extract_mode=extract_mode)
        (out_dir / "feature_columns.json").write_text(
            json.dumps(ex.COMMON_COLS + feat_mod.feature_columns(),
                       ensure_ascii=False, indent=2), encoding="utf-8")

    # record build settings (rf/xgboost do not apply the byte mask → applied=False)
    _save_config(out_base / args.model, args, dataset,
                 extract_mode, masked=False)


def _nf_mask_worker(task):
    """One session: masked_ints masking via the framework parser (dpkt) + Ethernet wrapping → dst pcap.
    (data extracted with the 'same parser and same masking' as the byte models. Parallel worker.)"""
    src, dst, ipm, pm, l3m, l4m = task
    try:
        from lib.parser.dpkt_parser import mask_wrap_to_ethernet
        n = mask_wrap_to_ethernet(src, dst, ip_mask=ipm, port_mask=pm,
                                  l3_mask=l3m, l4_mask=l4m)
        return 1 if n > 0 else 0
    except Exception:
        return 0


def build_netfound(args, dataset: str, fl_dir: Path, out_base: Path,
                   label_map: dict, splits: list):
    """NetFound: filelist sessions → {int_label}/raw/*.pcap → (container) preprocess_data.py → arrow.
    Output: {out_base}/netfound/{split}/_input/final/combined/*.arrow  (used as train_dir for training).
    ※ NetFound preprocess runs via docker exec since the C++/tokenizer lives in the torch_netfound container.
      Only session mode (vpn16 etc.) is supported — session pcap extraction for whole mode (cic18 etc.) is TODO."""
    import subprocess
    from lib.parser.pcap_source import build_pcap_index
    NF_REPO = os.environ.get(
        "NETFOUND_REPO",
        "netFound-main")
    CONT = os.environ.get("NETFOUND_CONTAINER", "torch_netfound")
    TOK_CONF = "configs/DefaultConfigNoTCPOptions.json"
    extract_mode = (args.extract_mode if args.extract_mode != "auto"
                    else DATASET_EXTRACT_MODE.get(dataset, "session"))
    if extract_mode != "session":
        print(f"  [netfound] {dataset}: whole-mode session pcap extraction not implemented (TODO) — session mode only")
        return
    pcap_index = build_pcap_index(ds.session_dir(dataset))
    for split_mode in splits:
        list_csv = _list_path(fl_dir, split_mode)
        if not list_csv.exists():
            print(f"  [{split_mode}] no filelist → skipped ({list_csv.name})")
            continue
        nf_dir = out_base / args.model / split_mode
        work = nf_dir / "_input"
        if work.exists():
            shutil.rmtree(work)
        import pandas as _pd
        from concurrent.futures import ProcessPoolExecutor
        lst = _pd.read_csv(list_csv, dtype=str, na_filter=False, encoding="utf-8-sig")
        lst.columns = [c.strip().lstrip("﻿") for c in lst.columns]
        n_miss = n_unk = 0
        tasks = []
        for _, r in lst.iterrows():
            gk = str(r["group_key"])
            if gk not in label_map:
                n_unk += 1; continue
            src = pcap_index.get(r["filename"])
            if src is None:
                n_miss += 1; continue
            raw = work / "raw" / str(label_map[gk])   # NetFound finetune layout: raw/{int_label}/*.pcap
            raw.mkdir(parents=True, exist_ok=True)
            tasks.append((str(src), str(raw / r["filename"]),
                          bool(args.ip_mask), bool(args.port_mask),
                          bool(args.l3_mask), bool(args.l4_mask)))
        if not tasks:
            print(f"  [{split_mode}] netfound: 0 valid sessions (missing={n_miss} unk={n_unk}) → skip")
            continue
        # ★ host-side parallel: masked_ints masking via the framework parser (dpkt) + Ethernet wrapping
        #   → data extracted with the 'same parser and same masking' as the byte models (extraction identical, only tokenization is netfound)
        print(f"  [{split_mode}] netfound: sessions {len(tasks)} (miss={n_miss} unk={n_unk}) "
              f"→ dpkt masking+Ethernet (ip={args.ip_mask} port={args.port_mask}, workers={args.workers})")
        n_ok = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for ok in ex.map(_nf_mask_worker, tasks, chunksize=16):
                n_ok += ok
        print(f"    → masking done {n_ok}/{len(tasks)}")
        par = os.path.join(str(CODE_DIR), "nf_preprocess_par.py")   # parallel preprocessing (per-session Tokenize)
        cmd = ["docker", "exec", "-w", NF_REPO, CONT, "python3", par,
               os.path.abspath(str(work)), os.path.join(NF_REPO, TOK_CONF),
               str(args.workers), NF_REPO]
        import time
        logf = nf_dir / "preprocess.log"
        comb = work / "final" / "combined"
        print(f"    running preprocess (progress=arrow files created/{n_ok}, details→{logf.name})")
        with open(logf, "w") as lg:
            proc = subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT)
            t0 = time.time()
            while proc.poll() is None:
                time.sleep(3)
                done = len(list(comb.glob("*.arrow"))) if comb.exists() else 0
                el = int(time.time() - t0)
                print(f"\r      arrow {done}/{n_ok}  ({el}s)   ", end="", flush=True)
            proc.wait()
        n_arrow = len(list(comb.glob("*.arrow"))) if comb.exists() else 0
        print(f"\r    → rc={proc.returncode}  arrow {n_arrow} files @ {comb}"
              + ("" if n_arrow else f"  (if 0, check {logf})") + " " * 10)
    _save_config(out_base / args.model, args, dataset, extract_mode,
                 masked=any([args.ip_mask, args.port_mask]))


def build_byte_model(args, dataset: str, fl_dir: Path, out_base: Path,
                     label_map: dict, splits: list):
    """6 byte-based models : parser + shaping path. out_base=…/{ds}/{variant}."""
    extract_mode = (args.extract_mode if args.extract_mode != "auto"
                    else DATASET_EXTRACT_MODE.get(dataset, "session"))
    roots = ([ds.pcap_dir(dataset)] if extract_mode == "whole"
             else [ds.session_dir(dataset)])
    shaping = importlib.import_module(f"lib.shaping.shaping_{args.model}")

    for split_mode in splits:
        list_csv = _list_path(fl_dir, split_mode)
        if not list_csv.exists():
            print(f"  [{split_mode}] no filelist → skipped ({list_csv.name})")
            continue
        out_dir = out_base / args.model / split_mode
        opt = {
            "ip_mask": args.ip_mask, "port_mask": args.port_mask,
            "l3_mask": args.l3_mask, "l4_mask": args.l4_mask,
            "parser": args.parser,
            "split_role": split_mode.split("_", 1)[0],       # train / test
            "tf_enhance": args.tf_enhance,
            "tf_randomize": not args.tf_no_randomize,
            # ★ extraction cache: in whole mode, per-session raw packets are scanned only once per (variant,split)→pkl,
            #   shared by the 7 models (avoids rescanning 400GB). Ignored in session mode.
            "_extract_cache": str(out_base / "_extract_cache" / split_mode),
        }
        writer = shaping.Writer(out_dir, opt)
        unknown = [0]

        def on_result(meta, sample, _w=writer, _u=unknown):
            gk = str(meta["group_key"])
            if gk not in label_map:
                _u[0] += 1
                return
            _w.add(meta, sample, label_map[gk])

        n_ok, n_missing, n_failed = pcap_source.run_split(
            list_csv, roots, extract_mode, args.workers,
            args.model, opt, on_result, desc=f"{dataset}/{split_mode}")
        n = writer.close()
        msg = (f"  [{split_mode}] saved={n}  ok={n_ok} missing_pcap={n_missing} "
               f"failed/filtered={n_failed}")
        if unknown[0]:
            msg += f"  not_in_label_map={unknown[0]}"
        print(msg)

    # record build settings (byte models actually apply the mask)
    masked = any([args.ip_mask, args.port_mask, args.l3_mask, args.l4_mask])
    _save_config(out_base / args.model, args, dataset,
                 extract_mode, masked=masked)


def build_byte_multi(args, dataset, fl_dir, out_base, label_map, splits, models):
    """★ Build N byte-based models simultaneously with 'a single parse' (parse_session once per session).
       The extraction cache (whole) and parsing are shared by all models → removes the per-model repeated parsing."""
    import importlib, copy
    extract_mode = (args.extract_mode if args.extract_mode != "auto"
                    else DATASET_EXTRACT_MODE.get(dataset, "session"))
    roots = ([ds.pcap_dir(dataset)] if extract_mode == "whole"
             else [ds.session_dir(dataset)])
    shapers = {m: importlib.import_module(f"lib.shaping.shaping_{m}") for m in models}
    masked = any([args.ip_mask, args.port_mask, args.l3_mask, args.l4_mask])

    for split_mode in splits:
        list_csv = _list_path(fl_dir, split_mode)
        if not list_csv.exists():
            print(f"  [{split_mode}] no filelist → skipped ({list_csv.name})")
            continue
        opt = {
            "ip_mask": args.ip_mask, "port_mask": args.port_mask,
            "l3_mask": args.l3_mask, "l4_mask": args.l4_mask,
            "parser": args.parser, "split_role": split_mode.split("_", 1)[0],
            "tf_enhance": args.tf_enhance, "tf_randomize": not args.tf_no_randomize,
            "_extract_cache": str(out_base / "_extract_cache" / split_mode),
        }
        writers = {m: shapers[m].Writer(out_base / m / split_mode, opt) for m in models}
        unk = [0]

        def on_result(meta, samples, _w=writers, _lm=label_map, _u=unk):
            gk = str(meta["group_key"])
            if gk not in _lm:
                _u[0] += 1
                return
            lbl = _lm[gk]
            for m, s in samples.items():
                if s is not None:
                    _w[m].add(meta, s, lbl)

        n_ok, n_missing, n_failed = pcap_source.run_split_multi(
            list_csv, roots, extract_mode, args.workers, list(models), opt,
            on_result, desc=f"{dataset}/{split_mode}[{'+'.join(models)}]")
        saved = {m: writers[m].close() for m in models}
        print(f"  [{split_mode}] ok={n_ok} missing_pcap={n_missing} "
              f"failed/filtered={n_failed} not_in_label_map={unk[0]} · saved="
              + " ".join(f"{m}:{saved[m]}" for m in models))

    for m in models:
        a2 = copy.copy(args); a2.model = m
        _save_config(out_base / m, a2, dataset, extract_mode, masked=masked)


def main():
    ap = argparse.ArgumentParser(
        description="filelist → build per-model 4-split datasets")
    ap.add_argument("--dataset", required=True, nargs="+",
                    help="dataset alias (multiple allowed, 'all'=all)")
    ap.add_argument("--model", nargs="*", default=None, choices=ALL_MODELS,
                    help="if omitted, all 7 models (rf xgboost 2dcnn etbert yatc netmamba trafficformer). "
                         "If given, only those models. Multiple byte-based models are built together with a single parse")
    ap.add_argument("--variant", choices=["full", "sizectrl", "strat"], default="full",
                    help="full=00_filelist→full/ (default, non-sizectrl). "
                         "sizectrl=00_filelist_sm→sizectrl/ (07 output). "
                         "Usually sizectrl is invoked by 07_make_sm_dataset.py instead.")
    ap.add_argument("--splits", nargs="+", default=SPLIT_MODES,
                    choices=SPLIT_MODES,
                    help="splits to build (default: all 4)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="[rf/xgboost] force re-extraction instead of reusing the output of the other model")
    # ── extraction ──
    ap.add_argument("--parser", choices=["dpkt", "tshark"], default="dpkt",
                    help="packet parser for byte models (only dpkt implemented)")
    ap.add_argument("--extract-mode", choices=["auto", "whole", "session"],
                    default="auto",
                    help="pcap extraction mode. auto=dataset default "
                         "(ustc16/cic17/cic18/iot23=whole, others=session). "
                         "Can be forced to whole/session.")
    # ── masking (byte models) ──
    ap.add_argument("--ip_mask", action="store_true",
                    help="mask IP addresses (IPv4/IPv6)")
    ap.add_argument("--port_mask", action="store_true",
                    help="mask src/dst ports")
    ap.add_argument("--l3_mask", action="store_true",
                    help="mask the entire L3 (IP) header")
    ap.add_argument("--l4_mask", action="store_true",
                    help="mask the entire L4 (TCP/UDP) header")
    # ── per-model ──
    ap.add_argument("--feat", default="all",
                    help="[rf/xgboost] feat code (default all = all 1,205 features. "
                         "feat/xgboost_feature_{feat}.py)")
    ap.add_argument("--tf_enhance", type=int, default=1,
                    help="[trafficformer] train RIFA augmentation factor (default 1=off, paper uses 5)")
    ap.add_argument("--tf_no_randomize", action="store_true",
                    help="[trafficformer] disable IP/port/TS randomization (on by default in the original)")
    args = ap.parse_args()
    if not args.model:                       # --model omitted → all 7 models
        args.model = ["xgboost", "rf", "2dcnn", "etbert", "yatc",
                      "netmamba", "trafficformer"]

    requested = []
    for d in args.dataset:
        requested.extend(ds.ALL_DATASETS if d == "all" else [d])
    seen = set()
    targets = [d for d in requested if not (d in seen or seen.add(d))]

    for dataset in targets:
        fl_dir = LIST_BASE / dataset / VARIANT_FL[args.variant]
        out_base = LIST_BASE / dataset / args.variant          # …/{ds}/{full|sizectrl}
        lm_path = fl_dir / "label_map.json"
        if not lm_path.exists():
            need = "07_make_sm_filelist.py" if args.variant == "sizectrl" else "05_make_filelist.py"
            print(f"[SKIP] {dataset}: no {fl_dir.name}/label_map — run {need} first")
            continue
        label_map = json.loads(lm_path.read_text(encoding="utf-8"))

        import copy
        feat = [m for m in args.model if m in FEAT_MODELS]
        nf   = [m for m in args.model if m in NETFOUND_MODELS]
        byte = [m for m in args.model if m not in FEAT_MODELS and m not in NETFOUND_MODELS]

        print(f"\n{'='*64}")
        print(f"  dataset  : {dataset}")
        print(f"  model    : {'+'.join(args.model)}")
        print(f"  variant  : {args.variant}")
        print(f"  filelist : {fl_dir.name}  →  out: {out_base.name}/")
        print(f"  splits   : {args.splits}")
        print(f"  masks    : ip={args.ip_mask} port={args.port_mask} "
              f"l3={args.l3_mask} l4={args.l4_mask}")
        print(f"  classes  : {len(label_map)}")
        print(f"  groups   : feat={feat}  nf={nf}  byte={byte}")
        print(f"{'='*64}")

        # feat: xgboost first (rf reuses it)
        for m in sorted(feat, key=lambda x: 0 if x == "xgboost" else 1):
            a2 = copy.copy(args); a2.model = m
            build_feat_model(a2, dataset, fl_dir, out_base, label_map, args.splits)
        for m in nf:
            a2 = copy.copy(args); a2.model = m
            build_netfound(a2, dataset, fl_dir, out_base, label_map, args.splits)
        if byte:                                  # ★ N byte models = built together with a single parse
            build_byte_multi(args, dataset, fl_dir, out_base, label_map, args.splits, byte)

        print(f"[done] {dataset}/{'+'.join(args.model)} ({args.variant})")


if __name__ == "__main__":
    main()
