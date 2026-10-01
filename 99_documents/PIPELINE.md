# Pipeline (steps 3-7)

`bash run_pipeline.sh <step> [datasets...]` runs a step with the settings of the article; the table lists the scripts
it runs (the exact flags, e.g. the per-model training settings of 99_documents/MODELS.md, are in `run_pipeline.sh`).
`NM_DATASET_ROOT` points to the captures placed by step 1 (`02_preprocess/00_rename.py`), default
`<repository>/00_assets/datasets`. Step 3 needs SplitCap and mono: `bash 00_assets/tools/setup_splitcap.sh`.

| Step | What it does | Command (per dataset `<ds>`) | Output |
|---|---|---|---|
| 3a | split captures into sessions (session datasets) | `02_preprocess/01_session_split.py --dataset <ds>` | `<dir>/02_session/` |
| 3b | per-session statistics and fields (tshark + dpkt) | `02_preprocess/03_session_stat.py --dataset <ds>` | `<dir>/03_session_stat/` |
| 3c | official labels (cic17 flow CSVs, cic18 attack schedule) | `00_assets/datasets/_tools/cic17/label_cic17.py`, `00_assets/datasets/_tools/cic18/label_cic18.py` | `<dir>/03_session_stat_labeled/` |
| 3d | check per-class counts | `00_assets/datasets/_tools/verify_counts.py --dataset <ds>` | table on screen |
| align | map the released lists to your file names | `02_preprocess/align_session_lists.py --dataset <ds>` | `01_dataset/<ds>/00_filelist*/` |
| 4 | mark every session with the 50 rules (one column per rule) | `02_preprocess/04_noise_labeling.py --dataset <ds>` (cic17, cic18: `--input-subdir 03_session_stat_labeled`) | `<dir>/04_session_noisy_labeled/` |
| 5 | session lists: full, sizectrl, sizectrl+strat (only with `LISTS=rebuild`) | `05_make_filelist.py --sample_per_class 1000`, `07_make_sm_filelist.py`, `make_refinement_counts.py <ds>`, `07b_make_inject_sm.py --inject 0.05`, `make_dataset_stat.py` | `01_dataset/<ds>/00_filelist{,_sm,_strat}/`, `99_documents/results/dataset_stats/<ds>/`, `99_documents/results/refinement_counts/` |
| 6 | model inputs (7 models x 3 variants) | `02_preprocess/06_make_dataset.py --dataset <ds> --model xgboost rf --variant <v>`; the five byte models in one call with `--ip_mask --port_mask` | `01_dataset/<ds>/<variant>/<model>/` |
| train | Exp1-Exp4 x 3 variants x 5 seeds x 7 models | `03_model/01_train.py --model <m> --dataset <ds> --exp <e> --variant <v> --seed <s>` + the per-model settings; then `03_model/02_result_table.py` | `99_documents/results/model_results/`, `wrong_lists/`, `result_tables/` |
| 7 | analyses of the five illusions (sizectrl; field maps by `02_preprocess/08_make_field_map.py`) | `04_analysis/*.py`, see `99_documents/PAPER_MAP.md` | `99_documents/results/analysis/`, `embeddings/`, `shap_tree/`, `shap_dl/` |

## The four experiments

| | Training set | Test set | Used for |
|---|---|---|---|
| Exp1 | refined | refined | Illusion 1 (Exp1 vs Exp3), 2, 3, 5 |
| Exp2 | refined | noisy | Illusion 4 (Exp2 vs Exp4) |
| Exp3 | noisy | refined | Illusion 1, 2, 3, 5 |
| Exp4 | noisy | noisy | the usual practice; Illusion 4 |

## The three variants

- **full**: refined and noisy lists drawn independently (up to 1,000 sessions per class, hash split 8:2, seed 42); the
  noisy lists are larger, as in current practice.
- **sizectrl** (main numbers): the noisy training list is cut to the per-class size of the refined one.
- **sizectrl+strat**: the refined lists of sizectrl, with the noisy lists made by adding up to 5 pp of noise to each
  class above its own noise rate (paired with the refined lists).

In the code, *denoised* means *refined* (the term of the article).

## Released results

`99_documents/results/model_results/` (per variant / seed / model / dataset / experiment), `99_documents/results/result_tables/`, and
`99_documents/results/analysis/` hold the numbers behind the article for the eight public datasets. Model weights (about 780 GB), the per-session rule
columns of cic18 and iot23 (hundreds of GB), and the captures themselves are not included; the scripts regenerate them.
