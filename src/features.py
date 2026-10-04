"""Graph-based feature engineering.

Never use time_step as a model feature, and never build features from
neighbours' true labels.

Status: graph features were tested in 04_graph_features.ipynb and did not
improve validation AUC, so the final models use the 165 raw features only.
`degree_features` is kept because that notebook uses it to show the test.
"""
import pandas as pd

from src.data import ID_COL


def degree_features(edges: pd.DataFrame, ids) -> pd.DataFrame:
    """In- and out-degree from the FULL edge list.

    in_degree  = number of earlier transactions whose outputs this one spends
    out_degree = number of later transactions that spend this one's outputs
    The full edge list includes neighbours that are missing from train/test,
    so degree means the same thing in train and test.
    """
    out = pd.DataFrame(index=pd.Index(ids, name=ID_COL))
    out["in_degree"] = out.index.map(edges.txId2.value_counts()).fillna(0)
    out["out_degree"] = out.index.map(edges.txId1.value_counts()).fillna(0)
    return out
