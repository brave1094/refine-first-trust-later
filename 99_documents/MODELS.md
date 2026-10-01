# Models (step 2)

`bash 00_assets/models/setup_models.sh` clones the four upstream repositories at the commits below and downloads the pre-trained
weights from the links in their READMEs. The upstream code is not redistributed here; please cite the original papers.
Each pinned commit was checked file by file (169 files) against the copy used in the experiments: all identical.

| Model | Paper | Code (pinned commit) | License | Pre-trained weights | Our change |
|---|---|---|---|---|---|
| XGBoost | Chen & Guestrin, KDD 2016 | `xgboost` (pip) | Apache-2.0 | — | none; features: `03_model/own_models/02_xgboost/feat/xgboost_feature_all.py` (1,205 statistics) |
| Random Forest | Breiman, 2001 | `scikit-learn` (pip) | BSD-3 | — | none; same 1,205 features |
| 2D-CNN | Wang et al., ICOIN 2017 | our implementation, `03_model/own_models/03_2dcnn/model.py` | MIT (this repo) | — | — |
| ET-BERT | Lin et al., WWW 2022 | [linwhitehat/ET-BERT](https://github.com/linwhitehat/ET-BERT) @ `d594706` | MIT | [Google Drive](https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE) | none |
| YaTC | Zhao et al., AAAI 2023 | [NSSL-SJTU/YaTC](https://github.com/NSSL-SJTU/YaTC) @ `a220b75` | no license file | [Google Drive](https://drive.google.com/file/d/1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16) | 4-line `forward()` shim for timm>=0.9 (`00_assets/models/05_yatc/apply_timm_compat.py`) |
| NetMamba | Wang et al., ICNP 2024 | [wangtz19/NetMamba](https://github.com/wangtz19/NetMamba) @ `bef641e` | no license file | [Hugging Face](https://huggingface.co/wangtz/NetMamba) | none |
| TrafficFormer | Zhou et al., IEEE S&P 2025 | [IDP-code/TrafficFormer](https://github.com/IDP-code/TrafficFormer) @ `6d0ba64` | MIT | [Google Drive](https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7) | none |

Full commit hashes are in `00_assets/models/setup_models.sh`. `00_assets/models/<model>/INTEGRATION_SPEC.md` (YaTC, NetMamba, TrafficFormer) documents how
the model is wired to our inputs (the model-specific shaping is in `02_preprocess/lib/shaping/`, training in `03_model/model/`).

## Training settings of the article

| Model | Batch | Epochs | Other |
|---|---:|---:|---|
| XGBoost | — | — | Optuna, 10 trials, objective accuracy (`--optuna_bas acc`) |
| Random Forest | — | — | Optuna, 10 trials, objective accuracy (default) |
| 2D-CNN | 256 | 100 | |
| ET-BERT | 64 | 20 | test every 3 epochs |
| YaTC | 256 | 20 | test every 3 epochs |
| NetMamba | 256 | 20 | test every 3 epochs |
| TrafficFormer | 32 | 20 | test every 3 epochs |

Byte-input models (2D-CNN, ET-BERT, YaTC, NetMamba, TrafficFormer) use IP and port masking. Seeds: 42, 1, 7, 2024,
31337. Each model runs in the Python environment required by its upstream repository (see their `requirements.txt`);
we used one Docker image per model.
