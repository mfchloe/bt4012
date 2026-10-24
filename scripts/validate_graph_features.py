import sys, warnings
sys.path.insert(0, "/Users/chloe/bt4012")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from src.data import FEATURE_COLS
from src.graph2 import known_ids, structural_features, known_edges, neighbour_score_features

tr = pd.read_parquet("/Users/chloe/bt4012/data/processed/train.parquet")
te = pd.read_parquet("/Users/chloe/bt4012/data/processed/test.parquet")
e = pd.read_csv("/Users/chloe/bt4012/data/raw/txs_edgelist.csv")
L = tr[tr.label.notna()].copy(); L["label"] = L.label.astype(int)
known = known_ids(tr, te)
ke = known_edges(e, known)
S = structural_features(e, L.txId, known)
L = L.join(S, on="txId")
STRUCT = list(S.columns)

P = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, colsample_bytree=0.3,
         class_weight="balanced", verbose=-1, n_jobs=8)

def fitpred(Xf, yf, Xv, seed=42, w=None, **kw):
    m = lgb.LGBMClassifier(**{**P, **kw}, random_state=seed).fit(Xf, yf, sample_weight=w)
    return m.predict_proba(Xv)[:, 1]

def stage1_scores(fit, val, feats):
    """OOF scores on fit rows (grouped by step), full-model scores on val rows."""
    s = pd.Series(index=pd.concat([fit.txId, val.txId]).values, dtype=float)
    steps = np.array(sorted(fit.time_step.unique()))
    folds = np.array_split(steps, 5)
    for fs in folds:
        m = fit.time_step.isin(fs)
        s.loc[fit.txId[m].values] = fitpred(fit.loc[~m, feats], fit.label[~m], fit.loc[m, feats])
    s.loc[val.txId.values] = fitpred(fit[feats], fit.label, val[feats])
    return s

def run(name, train_ends=(20, 22, 24, 26, 28), feats=FEATURE_COLS, nb=False, decay=None, **kw):
    aucs = []
    for te_ in train_ends:
        fit = L[L.time_step <= te_]; val = L[L.time_step > te_]
        F = list(feats)
        if nb:
            s = stage1_scores(fit, val, FEATURE_COLS)
            N = neighbour_score_features(ke, s)
            fit = fit.join(N, on="txId"); val = val.join(N, on="txId"); F += list(N.columns)
        w = None
        if decay:
            w = decay ** (te_ - fit.time_step.values)
        aucs.append(roc_auc_score(val.label, fitpred(fit[F], fit.label, val[F], w=w, **kw)))
    print(f"{name:40s} mean {np.mean(aucs):.4f} | " + " ".join(f"{a:.4f}" for a in aucs), flush=True)

run("base 165")
run("165 + struct", feats=FEATURE_COLS + STRUCT)
run("165 + nb-score", nb=True)
run("165 + struct + nb-score", feats=FEATURE_COLS + STRUCT, nb=True)
run("165 decay 0.9", decay=0.9)
run("165 decay 0.95", decay=0.95)
run("165 unbalanced", class_weight=None)
