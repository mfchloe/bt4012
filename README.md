# BT4012 Bitcoin Transaction Fraud Detection

This repository contains the notebooks and code for the BT4012 Kaggle
competition on classifying Bitcoin transactions as licit (0) or illicit (1).
The evaluation metric is ROC-AUC. The test transactions come from later time
steps than the labeled training transactions, so validation must respect time
order.

## Data setup

Download the competition data files from the Kaggle competition page after
joining the competition. Place the files directly in `data/raw/` using these
names:

- `train.csv`
- `test.csv`
- `txs_edgelist.csv`
- `sample_submission.csv`

Files in `data/raw/` are treated as read-only; derived data goes in
`data/processed/`. Keep all competition data under `data/`; this directory is
ignored by Git and must not be committed. The notebooks and source code expect
the paths above.

## Environment

From the repository root, create and activate a virtual environment, then
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On macOS, LightGBM and XGBoost also need the OpenMP runtime:
`brew install libomp`.

Paths are resolved in `src/data.py` relative to the repository root, so the
notebooks can be launched from any directory.

## Project structure

```
data/raw/          Kaggle files (read-only, not in Git)
data/processed/    Derived data (not in Git)
notebooks/         01_eda, 02_validation_setup, 03_baseline_models
src/               data.py (loading), validation.py (temporal CV), features.py
submissions/       Submission CSVs (not in Git)
experiments.csv    Log of every experiment and its validation / leaderboard AUC
```

## Reproducibility

- Validation is temporal only: train on earlier time steps, validate on later ones.
- `time_step` is used for splitting, never as a model feature.
- All random seeds are 42.
