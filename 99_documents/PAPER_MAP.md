# From the article to the code and the numbers

Numbers follow the Computer Networks version of the article. Rows of Private #1 and Private #2 are not part of this
repository (their data cannot be shared); every other number can be traced below.

| Article | Content | Script | Released numbers |
|---|---|---|---|
| Table 1 | the ten datasets | `02_preprocess/make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/01_origin_<ds>.csv` |
| Table 2, Fig. 1, Table B.1 | refinement practice in 68 surveyed NTC model papers | (manual coding, see the protocol) | `99_documents/survey/` |
| Fig. 2, Table 3, Supplementary S1 | taxonomy and the 50 rules | `02_preprocess/noise_rule/noise_rule_all.py`, `noise_rule/<NN>_noise_rule_<ds>.py` (rules applied per dataset) | `99_documents/Supplementary_S1.pdf` |
| Table 4 | refinement rate per dataset, and the noise rate of the sampled training lists | `02_preprocess/make_dataset_stat.py`; `04_analysis/cl_compare/variant_noise_check.py` (sampled noise rate) | `99_documents/results/dataset_stats/<ds>/02_refined_<ds>.csv`, `01_dataset/refined_list/`, `99_documents/results/analysis/09_cl_compare/variant_noise_check.csv` |
| Table C.1; Figs. C.1-C.4, C.6-C.9 | sessions marked per rule and per class | `02_preprocess/04_noise_labeling.py`, `make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/` |
| Fig. 3 | experimental design (Exp1-Exp4, three list variants) | `02_preprocess/05_make_filelist.py`, `07_make_sm_filelist.py`, `07b_make_inject_sm.py` | `01_dataset/<ds>/00_filelist*/` |
| Table 5 (sizectrl), Table D.1 (full), Table D.2 (sizectrl+strat) | accuracy of Exp1 and Exp3 | `03_model/01_train.py`, `03_model/02_result_table.py` | `99_documents/results/model_results/`, `99_documents/results/result_tables/` |
| Table 6 | Illusion 1: Delta = Exp1 - Exp3 per model, five-seed statistics | `04_analysis/01_target_signal.py`, `04_analysis/seed_stats.py` | `99_documents/results/analysis/01_target_signal/sizectrl/`, `99_documents/results/analysis/08_seed_stats/` |
| Table D.3 | accuracy and macro-F1 of Exp1 and Exp3 | `04_analysis/cl_compare/macro_f1_check.py` | `99_documents/results/analysis/09_cl_compare/macro_f1_check.csv` |
| Section 4.2 (dose-response) | sizectrl+strat injection | `04_analysis/strat_analysis.py`, `compare_sz_strat.py` | `99_documents/results/analysis/0*/strat/` |
| Section 4.2 (removing only the noise) | training on the refined set vs. removing only rule-marked sessions | `04_analysis/cl_compare/cl_compare.py` | `99_documents/results/analysis/09_cl_compare/` |
| Figs. 4-5 | Illusion 2: underfitting (Fig. 4) and overfitting (Fig. 5) | `04_analysis/02_convergence.py` | `99_documents/results/analysis/02_convergence/` |
| Fig. 6 | Illusion 3: silhouette (and Davies-Bouldin) of penultimate-layer embeddings | `04_analysis/a3_extract_emb.py`, `a3_extract_emb_uer.py`, `a3_metrics.py`, `a3_summary.py`, `04_analysis/03_boundary.py` | `99_documents/results/analysis/03_boundary/` |
| Table 7, Table D.4 | Illusion 4: Exp1 - Exp4 | `04_analysis/04_fail_to_eval.py`, `a4_eval_gen.py` | `99_documents/results/analysis/04_fail_to_eval/` |
| Fig. 7, Table D.5 | Illusion 4: error composition, with noise taken from the rule marking of each test session | `04_analysis/cl_compare/mech_true_noise.py` (replaces the list-difference definition of `a4_exp2_noise.py`) | `99_documents/results/analysis/09_cl_compare/mech_true_noise.csv`, `mech_true_noise_long.csv` |
| Tables D.6-D.7 | rules vs. confident learning under the same removal budget | `04_analysis/cl_compare/cl_compare.py` (ML models), `build_cl_dl.py`, `cl_dl_summary.py` (DL models) | `99_documents/results/analysis/09_cl_compare/` |
| Figs. 8-9, Table E.1 | Illusion 5: feature importance (tree SHAP, DL SHAP, byte-to-field mapping) | `04_analysis/a5_treeshap.py`, `a6_dlshap.py`, `a6_dlshap_uer.py`, `a6_audit.py`, `a7_field_agg.py`, `02_preprocess/08_make_field_map.py`, `04_analysis/05_feat_importance.py`, `05b_treeshap.py`, `05c_uncond_dist.py` | `99_documents/results/analysis/05_feat_importance/`, `06_dl_shap/`, `07_uncond_dist/` |
| (repository only) | content keys of the listed sessions; duplicate check | `02_preprocess/session_keys.py`, `02_preprocess/align_session_lists.py`, see `99_documents/DATASETS.md` | `01_dataset/<ds>/session_keys.csv` |

Not included: Tables A.1 and A.2 and Figs. C.5 and C.10 (private datasets).
