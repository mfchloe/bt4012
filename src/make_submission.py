"""Two-stage graph model: refit on all labelled steps 1-35 and write a submission.

Stage 1: LightGBM on raw + structural features. Scores for train rows are
out-of-fold (folds = blocks of time steps); test rows are scored by the full model.
Stage 2: adds mean/max of stage-1 scores over labelled neighbours, then fits the
final models. Explained step by step in notebooks/06_graph_model.ipynb.

Usage:  python -m src.make_submission <name> [--decay 0.95] [--kinds lgb,et] [--params JSON]
Best public LB (0.96032):  python -m src.make_submission 2026-10-23_lgb_graph_nb_decay95 --decay 0.95
"""
import argparse
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from src.data import FEATURE_COLS, PROCESSED_DIR, load_edges, load_test, load_train, save_submission
from src.graph2 import known_edges, known_ids, neighbour_score_features, structural_features

warnings.filterwarnings("ignore")
SEEDS = (42, 1, 2, 3, 4)
LGB = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, colsample_bytree=0.3,
           class_weight="balanced", verbose=-1, n_jobs=8)


def _read(name, loader):
    p = PROCESSED_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else loader()


def model(kind, seed, params=None):
    if kind == "lgb":
        return lgb.LGBMClassifier(**{**LGB, **(params or {})}, random_state=seed)
    if kind == "et":
        return ExtraTreesClassifier(500, max_features=0.3, min_samples_leaf=2, class_weight="balanced",
                                    n_jobs=8, random_state=seed)
    return RandomForestClassifier(500, max_features=0.3, class_weight="balanced", n_jobs=8, random_state=seed)


def fit_predict(kind, Xf, yf, Xp, seeds=SEEDS, w=None, params=None):
    if kind != "lgb":
        Xf, Xp = Xf.fillna(-1), Xp.fillna(-1)
    return np.mean([model(kind, s, params).fit(Xf, yf, sample_weight=w).predict_proba(Xp)[:, 1] for s in seeds], axis=0)


def stage1(fit, pred, feats):
    """Stage-1 score for every fit and pred row.

    Fit rows are split into 5 blocks of consecutive time steps; each block is scored
    by a model trained on the other blocks, so no row is scored by a model that saw it.
    """
    s = pd.Series(index=pd.concat([fit.txId, pred.txId]).values, dtype=float)
    for block in np.array_split(np.array(sorted(fit.time_step.unique())), 5):
        m = fit.time_step.isin(block)
        s.loc[fit.txId[m].values] = fit_predict("lgb", fit.loc[~m, feats], fit.label[~m], fit.loc[m, feats], seeds=(42,))
    s.loc[pred.txId.values] = fit_predict("lgb", fit[feats], fit.label, pred[feats], seeds=(42,))
    return s


def predict(train, test, edges, kinds=("lgb",), iters=1, decay=None, params=None, smooth=0.0):
    """Fit on all labelled train rows and return a score in [0, 1] for every test row."""
    fit = train[train.label.notna()].copy()
    fit["label"] = fit.label.astype(int)
    known = known_ids(train, test)
    ke = known_edges(edges, known)
    S = structural_features(edges, pd.concat([fit.txId, test.txId]), known)
    fit, pred = fit.join(S, on="txId"), test.join(S, on="txId")

    feats = FEATURE_COLS + list(S.columns)
    for it in range(iters):
        N = neighbour_score_features(ke, stage1(fit, pred, feats), prefix=f"nb{it}")
        fit = fit.drop(columns=N.columns, errors="ignore").join(N, on="txId")
        pred = pred.drop(columns=N.columns, errors="ignore").join(N, on="txId")
        feats = FEATURE_COLS + list(S.columns) + list(N.columns)

    # Recency weighting: a row from k steps before the last training step counts decay**k.
    w = None if decay is None else decay ** (fit.time_step.max() - fit.time_step.values)
    ranks = [rankdata(fit_predict(k, fit[feats], fit.label, pred[feats], w=w, params=params)) for k in kinds]
    p = np.mean(ranks, axis=0)
    p = (p - p.min()) / (p.max() - p.min())
    if smooth:
        # Blend each score with the mean score of its labelled neighbours.
        s = pd.Series(rankdata(p) / len(p), index=pred.txId.values)
        nb = neighbour_score_features(ke, s, prefix="sm")["sm_all_mean"]
        p = np.where(nb.notna(), (1 - smooth) * s + smooth * nb.fillna(0), s)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--kinds", default="lgb")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--decay", type=float, default=None)
    ap.add_argument("--params", type=json.loads, default=None,
                    help='final-model LightGBM overrides, e.g. \'{"num_leaves": 31}\'')
    ap.add_argument("--smooth", type=float, default=0.0, help="weight on neighbours' mean score")
    args = ap.parse_args()

    train, test, edges = _read("train", load_train), _read("test", load_test), load_edges()
    p = predict(train, test, edges, kinds=args.kinds.split(","), iters=args.iters, decay=args.decay,
                params=args.params, smooth=args.smooth)
    print(save_submission(test, p, f"{args.name}.csv"))


if __name__ == "__main__":
    main()
