# BT4012 Bitcoin Transaction Fraud Detection

This repository contains the notebooks and code for the BT4012 Kaggle
competition on classifying Bitcoin transactions as licit (0) or illicit (1).
The evaluation metric is ROC-AUC. The test transactions come from later time
steps than the labeled training transactions, so validation must respect time
order.

## Final submission

**`e1249331.ipynb`** (repository root) reproduces **both submissions selected
for the final private leaderboard**. It is self-contained and covers data
preprocessing, feature engineering, temporal validation and generating the
submission CSVs. Run it top to bottom after the data and environment setup
below; it takes about 12 minutes.

| Submission | Model | Public LB | Output file |
|---|---|---|---|
| 1 | Two-stage graph LightGBM (15 leaves) | **0.96032** | `submissions/e1249331_sub1.csv` |
| 2 | Same pipeline, larger stage-2 trees (31 leaves, 60% of columns, min 100 per leaf) | 0.95680 | `submissions/e1249331_sub2.csv` |

The model is a two-stage LightGBM:

1. **Stage 1** scores every transaction from its 165 features plus counts of
   labelled vs unlabelled neighbours. Training rows get out-of-fold scores
   (blocks of time steps) so no row is scored by a model that saw it.
2. **Stage 2** adds the mean/max stage-1 score of each transaction's labelled
   neighbours (illicit transactions cluster together) and is trained with
   recency weighting (a row *k* steps before the last training step gets
   weight 0.95^*k*), averaged over 5 seeds.

| Submission | Public LB |
|---|---|
| Random Forest, 165 raw features | 0.934 |
| LightGBM, 165 raw features | 0.948 |
| **Two-stage graph LightGBM + recency weighting** | **0.960** |

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
e1249331.ipynb     Final notebook: reproduces both selected submissions (LB 0.960, 0.957)
notebooks/         Development history, read in order:
                     01_eda              data, class balance, graph structure
                     02_validation_setup temporal validation design
                     03_baseline_models  LR / RF / LightGBM baselines (LB 0.934)
                     04_graph_features   first graph attempt, LightGBM (LB 0.948)
                     05_model_comparison tuning, CatBoost / XGBoost
                     06_graph_model      how the final model was reached (LB 0.960)
src/               data.py, validation.py, graph2.py (graph features), make_submission.py (final model)
scripts/           validation runs behind notebook 06 (decay, smoothing, ensembles, settings)
submissions/       Submission CSVs (not in Git)
experiments.csv    Log of every experiment and its validation / leaderboard AUC
```

## Reproducibility

- Validation is temporal only: train on earlier time steps, validate on later ones.
- `time_step` is never a model feature; it is only used for validation splits
  and for the recency weights.
- Random seeds are fixed (42 for single models; 42, 1, 2, 3, 4 for the
  seed-averaged final model). The final notebook reproduces both submitted
  files (0.96032 and 0.95680) exactly.
- The same two submissions can also be built from the command line:
  `python -m src.make_submission 2026-10-23_lgb_graph_nb_decay95 --decay 0.95` and
  `python -m src.make_submission 2026-10-24_lgb_graph_nb_decay95_31leaves --decay 0.95 --params '{"num_leaves": 31, "colsample_bytree": 0.6, "min_child_samples": 100}'`.
