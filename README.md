# Refine First, Trust Later

Code, refinement rules, session lists, and results for the article
**"Refine First, Trust Later: A Task-Relative Noise Refinement Framework and the Analysis of Five Illusions Noise Casts
on Network Traffic Classification"** (Yun-Seong Jang, Gyeong-Min Yu, Seung-Woo Nam, Ju-Sung Kim, Yang-Seo Choi,
Ui-Jun Baek, Myung-Sup Kim; under review at *Computer Networks*).

Noise in a traffic dataset is defined relative to the task: a session is noise when it does not carry the target signal
of the task. The framework organises such noise into three levels and six categories and implements it as 50 rules
(Aa, Bb, Cc, Cd, De, Df). Refining eight public datasets and training seven models under four conditions, three list
variants, and five seeds shows five illusions that noise casts on accuracy and its interpretation.

A guide with the overview, the pipeline, and a quick start is in [`99_documents/guide_KU.pdf`](99_documents/guide_KU.pdf).

## Repository layout

```
00_assets/       downloaded resources
  datasets/        the official captures, arranged as <dataset dir>/01_pcap/<class>/  (NM_DATASET_ROOT)
    _tools/          official labels (cic17, cic18), pcapng conversion, per-class checks
  tools/           SplitCap for step 3, fetched by setup_splitcap.sh (not redistributed)
  models/          third-party model code and weights, fetched by setup_models.sh; YaTC compatibility shim
01_dataset/      generated data: session lists, session keys, refined session lists (refined_list/), model inputs,
                 field-name maps
02_preprocess/   00_rename.py (step 1) and lib_rename/; sessions, statistics, the 50-rule engine (noise_rule/), lists, model inputs
03_model/        training (01_train.py, model/, lib/) and our own model code (own_models/: 1,205 features, 2D-CNN)
04_analysis/     analyses of the five illusions (a3-a7, 01-05*, seed statistics); cl_compare/: rules vs. confident
                 learning, error composition from the rule marking, macro-F1, sampled noise rates; cl_compare/: rules vs. confident
                 learning, error composition from the rule marking, macro-F1, sampled noise rates
99_documents/    DATASETS, MODELS, PIPELINE, PAPER_MAP, guide (PPTX/PDF), Supplementary Material S1
  results/         our numbers: model results, result tables, analysis outputs, dataset statistics, refinement counts
  survey/          the coded list of the 68 surveyed NTC model papers (Table 2, Fig. 1, Table B.1)
repo_paths.py    the single definition of this layout, used by every script
run_pipeline.sh  steps 3-7 with the settings of the article
```
In the code, *denoised* means *refined*, the term used in the article.

## Seven steps

| Step | What | Where |
|---|---|---|
| 1 | download the eight public datasets and give their files our names | [`99_documents/DATASETS.md`](99_documents/DATASETS.md), `02_preprocess/00_rename.py` |
| 2 | fetch the third-party model code and weights, and SplitCap | [`99_documents/MODELS.md`](99_documents/MODELS.md), `00_assets/models/setup_models.sh`, `00_assets/tools/setup_splitcap.sh` |
| 3 | split sessions and compute per-session statistics | `02_preprocess/01_session_split.py`, `03_session_stat.py` |
| 4 | mark sessions with the 50 rules | `02_preprocess/04_noise_labeling.py`, `02_preprocess/noise_rule/` |
| 5 | build the session lists (or align the released ones) | `02_preprocess/05_make_filelist.py`, `07_*`, `align_session_lists.py` |
| 6 | build the model inputs | `02_preprocess/06_make_dataset.py` |
| 7 | train and analyse | `03_model/`, `04_analysis/`, [`99_documents/PAPER_MAP.md`](99_documents/PAPER_MAP.md) |

## Quick start

```bash
git clone https://github.com/brave1094/refine-first-trust-later.git && cd refine-first-trust-later
pip install -r requirements.txt
bash 00_assets/models/setup_models.sh                                        # step 2
bash 00_assets/tools/setup_splitcap.sh                                       # SplitCap for step 3 (needs mono)
python 02_preprocess/00_rename.py --dataset vpn16 --src /path/to/ISCX-VPN-2016   # step 1: official files -> our names
bash run_pipeline.sh 3 vpn16            # sessions + statistics, per-class check
bash run_pipeline.sh align vpn16        # use the released session lists
bash run_pipeline.sh 4 vpn16            # 50-rule marking (step 5 is skipped: LISTS=published uses the released lists)
bash run_pipeline.sh 6 vpn16            # model inputs
GPU=0 bash run_pipeline.sh train vpn16  # Exp1-Exp4, three variants, five seeds
bash run_pipeline.sh 7                  # analyses (99_documents/PAPER_MAP.md)
```
Requirements: Python 3.10+, tshark/editcap/mergecap (Wireshark), mono (SplitCap), and `requirements.txt` for steps 1 and 3-7; each deep model
needs the environment of its upstream repository (99_documents/MODELS.md). The captures go to `00_assets/datasets/`
by default; set `NM_DATASET_ROOT` to use another location.

## Data availability

The captures are not redistributed; they are available from their publishers (99_documents/DATASETS.md). The two
private datasets of the article cannot be shared, and no data, list, or result of them is included here.

## License

Code: MIT (`LICENSE`). Session lists, statistics, and results: CC BY 4.0 (`DATA_LICENSE`); files derived from
CipherSpectrum inherit its CC BY-NC 4.0 terms. Third-party model code keeps its own license (99_documents/MODELS.md).

## Citation

Please cite the article if you use this repository (`CITATION.cff`, or "Cite this repository" on GitHub).
