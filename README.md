# Refine First, Trust Later

Code, refinement rules, session lists, and experimental results for the article
**"Refine First, Trust Later: A Task-Relative Noise Refinement Framework and the Analysis of Five Illusions Noise Casts on Network Traffic Classification"** (under review).

> This repository is being prepared. The rule engine, preprocessing pipeline, session lists of the eight public datasets,
> experiment scripts, result tables, and an English guide (`docs/`) will be added before the article is submitted.

## Planned contents

| Folder | Contents |
|---|---|
| `rules/` | Common rule engine with the 50 refinement rules (Aa, Bb, Cc, Cd, De, Df) |
| `preprocess/` | Session split, session statistics, rule marking, file lists, dataset building |
| `features/` | 1,205 statistical features for the tree models |
| `models/` | Training adapters for the seven models (links and patches for third-party code) |
| `experiments/` | Exp1-Exp4 under the full, sizectrl, and sizectrl+strat variants, five seeds |
| `analysis/` | Scripts that reproduce the tables and figures of the five illusions |
| `data/` | Refined and noisy session lists and dataset statistics of the eight public datasets |
| `results/` | Accuracy per model, dataset, experiment, variant, and seed; analysis outputs |
| `docs/` | Guide (PPTX, PDF) and Supplementary Material S1 (the 50 rules) |

## License

Code: MIT License (`LICENSE`). Session lists, statistics, and results: CC BY 4.0 (`DATA_LICENSE`).
Session lists derived from CipherSpectrum inherit its CC BY-NC 4.0 terms (non-commercial use only).

## Citation

If you use this repository, please cite the article (see `CITATION.cff`, or the "Cite this repository" button on GitHub).
