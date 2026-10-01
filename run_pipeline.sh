#!/usr/bin/env bash
# run_pipeline.sh — steps 3 to 7 with the settings used in the article (see 99_documents/PIPELINE.md for each step).
#
#   bash run_pipeline.sh <step> [datasets...]
#     step: 3 | align | 4 | 5 | 6 | train | 7 | all      datasets: default = all eight public datasets
#
#   environment:
#     NM_DATASET_ROOT   dataset root (default <repo>/00_assets/datasets), filled by step 1 (02_preprocess/00_rename.py)
#     SPLITCAP          SplitCap.exe for step 3 (default 00_assets/tools/SplitCap/; bash 00_assets/tools/setup_splitcap.sh)
#     CIC17_LABELS      folder with the official CIC-IDS2017 "TrafficLabelling" CSVs (cic17 labelling in step 3)
#     WORKERS           parallel workers for steps 3-6 (default 16)
#     GPU               GPU index for training (default 0)
#     LISTS             published (default: align and use the released session lists) | rebuild (redo step 5)
#     VARIANTS          default "full sizectrl strat"      SEEDS default "42 1 7 2024 31337"
#
# Steps that need GPUs (train, and the DL parts of 7) expect the environment of each model (99_documents/MODELS.md).
# This driver lists the commands of the article's runs in order; resume or parallelise them as your hardware allows.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NM_DATASET_ROOT="${NM_DATASET_ROOT:-$REPO/00_assets/datasets}"
STEP="${1:?usage: run_pipeline.sh <3|align|4|5|6|train|7|all> [datasets...]}"; shift || true
DATASETS="${*:-vpn16 tor16 tls1.3 cispec ustc16 cic17 cic18 iot23}"
WORKERS="${WORKERS:-16}"; GPU="${GPU:-0}"; LISTS="${LISTS:-published}"
VARIANTS="${VARIANTS:-full sizectrl strat}"; SEEDS="${SEEDS:-42 1 7 2024 31337}"
MODELS="xgboost rf 2dcnn etbert yatc netmamba trafficformer"
BYTE_MODELS=" 2dcnn etbert yatc netmamba trafficformer "
declare -A BS=([etbert]=64 [2dcnn]=256 [yatc]=256 [netmamba]=256 [trafficformer]=32)
declare -A EXTRA=([2dcnn]="--epochs 100" [etbert]="--epochs 20 --test_interval 3" [yatc]="--epochs 20 --test_interval 3"
                  [netmamba]="--epochs 20 --test_interval 3" [trafficformer]="--epochs 20 --test_interval 3"
                  [xgboost]="--optuna --optuna_trials 10 --optuna_bas acc" [rf]="--optuna --optuna_trials 10")
declare -A DIR=([ustc16]=01_USTC-TFC_2016 [cic17]=02_CIC-IDS-2017 [cic18]=03_CIC-IDS-2018 [iot23]=04_CIC_IoT_Dataset_2023
                [vpn16]=21_ISCX-VPN-2016 [tor16]=22_ISCX-TOR-2016 [tls1.3]=23_CSTNET_TLS1.3 [cispec]=24_CipherSpectrum)
run() { echo "+ $*"; "$@"; }

step3() {   # session split and per-session statistics (+ official labels for cic17 / cic18)
  cd "$REPO/02_preprocess"
  for ds in $DATASETS; do
    run python3 01_session_split.py --dataset "$ds" --workers "$WORKERS"
    run python3 03_session_stat.py --dataset "$ds" --workers "$WORKERS"
    case "$ds" in
      cic17) run python3 "$REPO/00_assets/datasets/_tools/cic17/label_cic17.py" --cic-dir "${CIC17_LABELS:?set CIC17_LABELS}" \
               --in-dir "$NM_DATASET_ROOT/${DIR[$ds]}/03_session_stat" --workers "$WORKERS" ;;
      cic18) run python3 "$REPO/00_assets/datasets/_tools/cic18/label_cic18.py" \
               --in-dir "$NM_DATASET_ROOT/${DIR[$ds]}/03_session_stat" --workers "$WORKERS" ;;
    esac
    run python3 "$REPO/00_assets/datasets/_tools/verify_counts.py" --dataset "$ds"
  done
}
step_align() { for ds in $DATASETS; do run python3 "$REPO/02_preprocess/align_session_lists.py" --dataset "$ds"; done; }
step4() {   # rule marking with the 50 rules
  cd "$REPO/02_preprocess"
  for ds in $DATASETS; do
    extra=""; [[ "$ds" == cic17 || "$ds" == cic18 ]] && extra="--input-subdir 03_session_stat_labeled"
    run python3 04_noise_labeling.py --dataset "$ds" --workers "$WORKERS" $extra
  done
}
step5() {   # session lists: full (1,000 per class, hash split 8:2, seed 42), sizectrl, sizectrl+strat
  if [[ "$LISTS" == published ]]; then echo "LISTS=published: using the released lists (step 5 skipped)"; return; fi
  cd "$REPO/02_preprocess"
  for ds in $DATASETS; do
    run python3 05_make_filelist.py --dataset "$ds" --sample_per_class 1000
    run python3 07_make_sm_filelist.py --dataset "$ds" --seed 42
    run python3 make_refinement_counts.py "$ds"
    run python3 07b_make_inject_sm.py --dataset "$ds" --inject 0.05
    run python3 make_dataset_stat.py --dataset "$ds"
  done
}
step6() {   # model inputs for the seven models and the three variants (the five byte models share one parse)
  cd "$REPO/02_preprocess"
  for ds in $DATASETS; do for v in $VARIANTS; do
    run python3 06_make_dataset.py --dataset "$ds" --model xgboost rf --variant "$v" --workers "$WORKERS"
    run python3 06_make_dataset.py --dataset "$ds" --model $BYTE_MODELS --variant "$v" --ip_mask --port_mask --workers "$WORKERS"
  done; done
}
step_train() {   # Exp1-Exp4 x variants x seeds x models
  cd "$REPO/03_model"
  for v in $VARIANTS; do for s in $SEEDS; do for ds in $DATASETS; do for m in $MODELS; do for e in 1 2 3 4; do
    bs=""; [[ -n "${BS[$m]:-}" ]] && bs="--batch_size ${BS[$m]}"
    run python3 -u 01_train.py --model "$m" --dataset "$ds" --exp "$e" --variant "$v" --gpu "$GPU" --seed "$s" $bs ${EXTRA[$m]}
  done; done; done; done; done
  run python3 02_result_table.py
}
step7() {   # analyses behind Illusions 1-5 (99_documents/PAPER_MAP.md maps each table and figure to its script)
  export SCIE_VARIANT=sizectrl                  # Illusions 3-5 are analysed on the sizectrl variant
  cd "$REPO/02_preprocess"
  run python3 08_make_field_map.py --dataset $DATASETS --variant sizectrl --split test --mode denoised --workers "$WORKERS"
  cd "$REPO/04_analysis"
  run python3 seed_stats.py
  run python3 a3_extract_emb.py --gpu "$GPU" --tsne
  for m in etbert trafficformer; do run python3 a3_extract_emb_uer.py --model "$m" --gpu "$GPU" --tsne; done
  run python3 a3_summary.py
  run python3 a4_exp2_noise.py
  run python3 a4_eval_gen.py
  for m in rf xgboost; do run python3 a5_treeshap.py --model "$m"; done
  for m in 2dcnn yatc netmamba; do run python3 a6_dlshap.py --model "$m" --gpu "$GPU"; done
  for m in etbert trafficformer; do run python3 a6_dlshap_uer.py --model "$m" --gpu "$GPU"; done
  run python3 a6_audit.py
  run python3 a7_field_agg.py
  for s in 01_target_signal 02_convergence 03_boundary 04_fail_to_eval 05_feat_importance 05b_treeshap 05c_uncond_dist \
           strat_analysis compare_sz_strat; do
    run python3 "$s.py"
  done
}

case "$STEP" in
  3) step3 ;; align) step_align ;; 4) step4 ;; 5) step5 ;; 6) step6 ;; train) step_train ;; 7) step7 ;;
  all) step3; [[ "$LISTS" == published ]] && step_align; step4; step5; step6; step_train; step7 ;;
  *) echo "unknown step: $STEP"; exit 2 ;;
esac
