# cl_compare — rules vs. confident learning, and the rule-based noise marking

Analyses on the sizectrl variant of the eight public datasets. Every session is marked as noise with the same rule set
used for the sampled noise rates of the article (`default_clean_rules` of `02_preprocess/noise_rule/` minus `Aa_7`),
read from `04_session_noisy_labeled/` of each dataset.

Requires `cleanlab` (`pip install cleanlab`) in addition to `requirements.txt`.

## Scripts (run in this order)

| Step | Script | Computes | Article |
|---|---|---|---|
| 1 | `cl_compare.py keys cl run sum` (CPU) | `keys`: rule marking of every session of the sizectrl feature files. `cl`: 5-fold out-of-sample XGBoost probabilities → cleanlab (`cl_default`, and `cl_budget` = the m lowest-quality sessions, m = number of rule-noise sessions). `run`: XGBoost and RF trained on raw / rule / cl_budget / random / cl_default, 5 seeds. `sum`: overlap of CL and rule removals, accuracy summary with paired differences | Section 4.2 "removing only the noise"; Tables D.6–D.7 (rules vs. confident learning) |
| 2 | `build_cl_dl.py` + `run_cl_dl.sh` (GPU) | DL training subsets with the same removal sets (vpn16, tor16, iot23: cl_rule, cl_budget, cl_random_s<seed>; cic17: cl_default as control), 250 Exp3 runs of the five DL models, then `cl_dl_summary.py` | DL rows of Tables D.6–D.7 |
| 3 | `mech_true_noise.py` | Error composition of Exp2/Exp4 from the rule marking of the noisy test set (replaces the list-difference definition of `04_analysis/a4_exp2_noise.py`) | Fig. 7, Table D.5 |
| 4 | `macro_f1_check.py` | Exp1 vs. Exp3 accuracy and macro-F1 on the refined test set from the wrong lists | Table D.3 |
| 5 | `variant_noise_check.py` | Sampled noise rate of the full / sizectrl / strat feature files (cic18, iot23) | Table 4 (sampled noise rates) |

```bash
cd 04_analysis/cl_compare
python3 cl_compare.py keys --procs 16
python3 cl_compare.py cl run sum --workers 4 --threads 4
GPU=0 bash run_cl_dl.sh
python3 mech_true_noise.py --procs 16
python3 macro_f1_check.py
python3 variant_noise_check.py --procs 16
```

## Inputs

- `<NM_DATASET_ROOT>/<dataset dir>/04_session_noisy_labeled/*.csv` (per-session rule flags)
- `01_dataset/<ds>/00_filelist_sm/` (session lists, `label_map.json`) and `01_dataset/<ds>/sizectrl/xgboost/<split>/features.csv`
  (steps 1, 3, 4); `01_dataset/<ds>/{full,sizectrl,strat}/xgboost/` and `00_filelist{,_sm,_strat}/` (step 5)
- `01_dataset/<ds>/sizectrl/<model>/` (DL inputs, step 2)
- `99_documents/results/wrong_lists/sizectrl/<model>/<ds>/exp{1,2,3,4}_wrong.csv` (steps 3, 4; seed 42)
- `99_documents/results/result_tables/<variant>[_seed<N>]/<model>.csv` (step 2)

## Outputs

All in `99_documents/results/analysis/09_cl_compare/`: `keys/`, `sample_noise_check.csv`, `<ds>_cl_flags.csv`,
`acc_long.csv`, `overlap.csv`, `summary.csv`, `dl_long.csv`, `dl_summary.csv`, `mech_true_noise_long.csv`,
`mech_true_noise.csv`, `macro_f1_check.csv`, `variant_noise_check.csv`.
