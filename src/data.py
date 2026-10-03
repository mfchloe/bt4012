"""Data loading helpers
"""
from pathlib import Path

import pandas as pd

# Repository root = parent of src/. Using an absolute path means the notebooks
# work no matter which directory Jupyter was started from.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SUBMISSIONS_DIR = ROOT / "submissions"

# Fixed seed used everywhere for reproducibility.
SEED = 42

# Column names. The Kaggle files call the features feat_1 ... feat_165.
ID_COL = "txId"
TIME_COL = "time_step"
LABEL_COL = "label"            # 1 = illicit, 0 = licit, NaN = unknown
TEST_INDEX_COL = "index"       # row id used by the submission file
FEATURE_COLS = [f"feat_{i}" for i in range(1, 166)]

# Note: in the original Elliptic data the first 93 features are "local" and the
# last 72 are neighbour aggregates. 01_eda.ipynb found this only partly supported
# (best boundary anywhere from 93 to 99), so no local/aggregated split is defined here.


def load_train() -> pd.DataFrame:
    """Training transactions (time steps 1-35). `label` is NaN when unknown."""
    return pd.read_csv(RAW_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    """Test transactions (later time steps). Has an extra `index` column."""
    return pd.read_csv(RAW_DIR / "test.csv")


def load_edges() -> pd.DataFrame:
    """Directed payment-flow edges: an output of txId1 is spent by txId2."""
    return pd.read_csv(RAW_DIR / "txs_edgelist.csv")


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a known label, with the label cast to int."""
    out = df[df[LABEL_COL].notna()].copy()
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    return out


def load_sample_submission() -> pd.DataFrame:
    """Kaggle's example submission: columns `index,target`, one row per test row."""
    return pd.read_csv(RAW_DIR / "sample_submission.csv")


def save_submission(test: pd.DataFrame, pred, filename: str) -> Path:
    """Write a submission CSV after checking it matches sample_submission.csv exactly.

    Checks: same column names in the same order, same number of rows, same
    `index` values in the same order, and every prediction is a probability in [0, 1].
    Nothing is written if any check fails.
    """
    sample = load_sample_submission()
    sub = pd.DataFrame({"index": test[TEST_INDEX_COL].values, "target": pred})

    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)} != {list(sample.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    assert (sub["index"].values == sample["index"].values).all(), "index values/order differ from sample"
    assert sub["target"].notna().all(), "missing predictions"
    assert sub["target"].between(0, 1).all(), "predictions must be probabilities in [0, 1]"

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = SUBMISSIONS_DIR / filename
    sub.to_csv(path, index=False)
    return path
