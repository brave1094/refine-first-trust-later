# From the article to the code and the numbers

Numbers follow the Computer Networks version of the article. Rows of Private #1 and Private #2 are not part of this
repository (their data cannot be shared); every other number can be traced below.

| Article | Content | Script | Released numbers |
|---|---|---|---|
| Table 1 | the ten datasets | `02_preprocess/make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/01_origin_<ds>.csv` |
| Fig. 2, Table 6, Table 7, Supplementary S1 | taxonomy and the 50 rules | `02_preprocess/noise_rule/noise_rule_all.py`, `noise_rule/<NN>_noise_rule_<ds>.py` (rules applied per dataset) | `99_documents/Supplementary_S1.pdf` |
| Table 8 | refinement per dataset | `02_preprocess/make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/02_refined_<ds>.csv` |
| Tables 9-11, 13-16; Figs. A.1-A.4, A.6-A.9 | sessions marked per rule and per class | `02_preprocess/04_noise_labeling.py`, `make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/` |
| Fig. 3 | experimental design | `02_preprocess/05_make_filelist.py`, `07_make_sm_filelist.py`, `07b_make_inject_sm.py` | `01_dataset/<ds>/00_filelist*/` |
| Tables 18-20 | accuracy of Exp1-Exp4 (full, sizectrl, strat) | `03_model/01_train.py`, `03_model/02_result_table.py` | `99_documents/results/model_results/`, `99_documents/results/result_tables/` |
| Table 21 | Illusion 1: Delta = Exp1 - Exp3 per model | `04_analysis/01_target_signal.py` | `99_documents/results/analysis/01_target_signal/sizectrl/` |
| Table 22 | five-seed statistics of Delta | `04_analysis/seed_stats.py` | `99_documents/results/analysis/08_seed_stats/` |
| Table 23 | sizectrl+strat check | `04_analysis/strat_analysis.py`, `compare_sz_strat.py` | `04_analysis/0*/strat/` |
| Table 24, Figs. 4-6 | Illusion 2: underfitting / memorisation / no effect | `04_analysis/02_convergence.py` | `99_documents/results/analysis/02_convergence/` |
| Figs. 7-9 | Illusion 3: silhouette, capacity, t-SNE | `04_analysis/a3_extract_emb.py`, `a3_extract_emb_uer.py`, `a3_metrics.py`, `a3_summary.py`, `04_analysis/03_boundary.py` | `99_documents/results/analysis/03_boundary/` |
| Table 25, Table 26, Fig. 10, Table B.1 | Illusion 4: Exp1 - Exp4, error decomposition | `04_analysis/a4_exp2_noise.py`, `a4_eval_gen.py`, `04_analysis/04_fail_to_eval.py` | `99_documents/results/analysis/04_fail_to_eval/` |
| Figs. 11-13, Table 27 | Illusion 5: feature importance (tree SHAP / impurity, DL SHAP, byte-to-field mapping) | `04_analysis/a5_treeshap.py`, `a6_dlshap.py`, `a6_dlshap_uer.py`, `a6_audit.py`, `a7_field_agg.py`, `02_preprocess/08_make_field_map.py`, `04_analysis/05_feat_importance.py`, `05b_treeshap.py`, `05c_uncond_dist.py` | `99_documents/results/analysis/05_feat_importance/`, `06_dl_shap/`, `07_uncond_dist/` |
| (repository only) | content keys of the listed sessions; tor16 duplicate check | `02_preprocess/session_keys.py`, `02_preprocess/align_session_lists.py`, see `99_documents/DATASETS.md` | `01_dataset/<ds>/session_keys.csv` |

Not included: Tables 2, 3, 12, 17 and Figs. A.5, A.10 (private datasets); Table 4 and Fig. 1 (literature survey,
documented in the article and its supplementary material).
