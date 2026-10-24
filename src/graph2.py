"""Graph features that mean the same thing in train and test.

In the test steps (36-49) only the originally-labelled transactions have
features; the 46,668 edge-only nodes are the originally-unknown ones. So every
neighbour feature here is built from *labelled* neighbours only (labelled train
rows + test rows), which keeps train and test consistent.
"""
import numpy as np
import pandas as pd

from src.data import ID_COL, LABEL_COL


def known_ids(train, test):
    return set(train.loc[train[LABEL_COL].notna(), ID_COL]) | set(test[ID_COL])


def structural_features(edges, ids, known):
    """Degree split into labelled / unlabelled neighbours."""
    e = edges.assign(k1=edges.txId1.isin(known), k2=edges.txId2.isin(known))
    f = pd.DataFrame(index=pd.Index(ids, name=ID_COL))
    f["indeg"] = e.txId2.value_counts()
    f["outdeg"] = e.txId1.value_counts()
    f["in_known"] = e[e.k1].txId2.value_counts()
    f["out_known"] = e[e.k2].txId1.value_counts()
    f = f.fillna(0)
    f["in_unk"] = f.indeg - f.in_known
    f["out_unk"] = f.outdeg - f.out_known
    f["frac_known"] = (f.in_known + f.out_known) / (f.indeg + f.outdeg).clip(lower=1)
    return f


def known_edges(edges, known):
    return edges[edges.txId1.isin(known) & edges.txId2.isin(known)]


def neighbour_score_features(kedges, score: pd.Series, prefix="nb"):
    """Mean / max of a per-node score over in-, out- and all labelled neighbours."""
    s = score
    out = {}
    for name, src, dst in [("in", "txId1", "txId2"), ("out", "txId2", "txId1")]:
        v = kedges[[src, dst]].assign(p=kedges[src].map(s)).dropna()
        g = v.groupby(dst).p
        out[f"{prefix}_{name}_mean"] = g.mean()
        out[f"{prefix}_{name}_max"] = g.max()
    both = pd.concat([
        kedges.assign(node=kedges.txId2, p=kedges.txId1.map(s)),
        kedges.assign(node=kedges.txId1, p=kedges.txId2.map(s)),
    ]).dropna(subset=["p"])
    g = both.groupby("node").p
    out[f"{prefix}_all_mean"] = g.mean()
    out[f"{prefix}_all_max"] = g.max()
    return pd.DataFrame(out).reindex(score.index)
