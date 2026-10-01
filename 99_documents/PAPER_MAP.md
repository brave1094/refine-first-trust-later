# From the article to the code and the numbers

Numbers follow the Computer Networks version of the article. Rows of Private #1 and Private #2 are not part of this
repository (their data cannot be shared); every other number can be traced below.

| Article | Content | Script | Released numbers |
|---|---|---|---|
| Table 1 | the ten datasets | `02_preprocess/make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/01_origin_<ds>.csv` |
| Table 4, Fig. 1, Table A.1 | refinement practice in 68 surveyed NTC model papers | (manual coding, see the protocol) | `99_documents/survey/` |
| Fig. 2, Table 5, Supplementary S1 | taxonomy and the 50 rules | `02_preprocess/noise_rule/noise_rule_all.py`, `noise_rule/<NN>_noise_rule_<ds>.py` (rules applied per dataset) | `99_documents/Supplementary_S1.pdf` |
| Table 6 | refinement per dataset | `02_preprocess/make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/02_refined_<ds>.csv` |
| Table B.1; Figs. B.1-B.4, B.6-B.9 | sessions marked per rule and per class | `02_preprocess/04_noise_labeling.py`, `make_dataset_stat.py` | `99_documents/results/dataset_stats/<ds>/` |
| Fig. 3 | experimental design (Exp1-Exp4, three list variants) | `02_preprocess/05_make_filelist.py`, `07_make_sm_filelist.py`, `07b_make_inject_sm.py` | `01_dataset/<ds>/00_filelist*/` |
| Tables 7-9 | accuracy of Exp1 and Exp3 (full, sizectrl, sizectrl+strat) | `03_model/01_train.py`, `03_model/02_result_table.py` | `99_documents/results/model_results/`, `99_documents/results/result_tables/` |
| Table 10 | Illusion 1: Delta = Exp1 - Exp3 per model, five-seed statistics | `04_analysis/01_target_signal.py`, `04_analysis/seed_stats.py` | `99_documents/results/analysis/01_target_signal/sizectrl/`, `99_documents/results/analysis/08_seed_stats/` |
| Section 4.2 (dose-response) | sizectrl+strat injection | `04_analysis/strat_analysis.py`, `compare_sz_strat.py` | `99_documents/results/analysis/0*/strat/` |
| Figs. 4-5 | Illusion 2: underfitting / memorisation / no effect | `04_analysis/02_convergence.py` | `99_documents/results/analysis/02_convergence/` |
| Fig. 6 | Illusion 3: silhouette (and Davies-Bouldin) of penultimate-layer embeddings | `04_analysis/a3_extract_emb.py`, `a3_extract_emb_uer.py`, `a3_metrics.py`, `a3_summary.py`, `04_analysis/03_boundary.py` | `99_documents/results/analysis/03_boundary/` |
| Table 11, Fig. 7, Tables C.1-C.2 | Illusion 4: Exp1 - Exp4, error decomposition with Exp2 | `04_analysis/a4_exp2_noise.py`, `a4_eval_gen.py`, `04_analysis/04_fail_to_eval.py` | `99_documents/results/analysis/04_fail_to_eval/` |
| Figs. 8-9, Table 12 | Illusion 5: feature importance (tree SHAP, DL SHAP, byte-to-field mapping) | `04_analysis/a5_treeshap.py`, `a6_dlshap.py`, `a6_dlshap_uer.py`, `a6_audit.py`, `a7_field_agg.py`, `02_preprocess/08_make_field_map.py`, `04_analysis/05_feat_importance.py`, `05b_treeshap.py`, `05c_uncond_dist.py` | `99_documents/results/analysis/05_feat_importance/`, `06_dl_shap/`, `07_uncond_dist/` |
| (repository only) | content keys of the listed sessions; duplicate check | `02_preprocess/session_keys.py`, `02_preprocess/align_session_lists.py`, see `99_documents/DATASETS.md` | `01_dataset/<ds>/session_keys.csv` |

Not included: Tables 2 and 3 and Figs. B.5 and B.10 (private datasets).
