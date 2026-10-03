"""Temporal validation: train on earlier time steps, validate on later ones.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.data import LABEL_COL, ROOT, TIME_COL, labeled

# Default split: fit on steps 1-24, validate on steps 25-35.
# This mirrors the real task (35 known steps -> 14 future steps) while keeping ~1,400 illicit examples in the validation window.
TRAIN_END = 24
VAL_END = 35


def temporal_split(df, train_end=TRAIN_END, val_end=VAL_END):
    """Split labeled rows into (fit, validation) by time step.

    fit: time_step <= train_end
    validation: train_end < time_step <= val_end
    Unlabeled rows are dropped.
    """
    df = labeled(df)
    fit = df[df[TIME_COL] <= train_end]
    val = df[(df[TIME_COL] > train_end) & (df[TIME_COL] <= val_end)]
    return fit, val


def per_step_auc(y_true, y_score, steps):
    """ROC-AUC computed separately for each time step.

    Returns one row per step with its size, number of illicit rows and AUC.
    AUC is NaN for a step that contains only one class (AUC is undefined there).
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score), "step": np.asarray(steps)})
    rows = []
    for step, g in frame.groupby("step"):
        auc = roc_auc_score(g.y, g.p) if g.y.nunique() == 2 else np.nan
        rows.append({TIME_COL: step, "n": len(g), "n_illicit": int(g.y.sum()), "auc": auc})
    return pd.DataFrame(rows).set_index(TIME_COL)


def evaluate(model, features, df, train_end=TRAIN_END, val_end=VAL_END):
    """Fit `model` on early steps and score it on later steps.

    Parameters
    ----------
    model    : any scikit-learn-style classifier with predict_proba
               (it is cloned, so the object passed in is left untouched)
    features : list of column names to use as inputs (never include time_step)
    df       : training DataFrame (labeled + unlabeled; unlabeled rows are dropped)

    Returns a dict with
      overall_auc   - one AUC over all validation rows pooled together. This is the
                      headline number, because Kaggle also computes one pooled AUC.
      mean_step_auc - average of the per-step AUCs (each step weighted equally)
      std_step_auc  - how much AUC varies between steps (stability)
      per_step      - DataFrame with n, n_illicit and auc for each validation step
      model         - the fitted model
      val_pred      - validation rows with their predicted probability
    """
    if TIME_COL in features:
        raise ValueError("time_step must not be used as a model feature")

    fit, val = temporal_split(df, train_end, val_end)

    model = clone(model)
    model.fit(fit[features], fit[LABEL_COL])
    p_val = model.predict_proba(val[features])[:, 1]

    steps = per_step_auc(val[LABEL_COL], p_val, val[TIME_COL])
    return {
        "overall_auc": roc_auc_score(val[LABEL_COL], p_val),
        "mean_step_auc": steps.auc.mean(),
        "std_step_auc": steps.auc.std(),
        "per_step": steps,
        "model": model,
        "val_pred": val[[TIME_COL, LABEL_COL]].assign(pred=p_val),
    }


EXPERIMENTS_FILE = ROOT / "experiments.csv"
EXPERIMENT_COLS = ["date", "notebook", "change", "val_auc", "public_lb_auc", "notes"]


def log_experiment(date, notebook, change, val_auc, notes="", public_lb_auc=None):
    """Record one result in experiments.csv.

    One row per (notebook, change). Re-running a notebook updates its rows
    instead of adding duplicates, and keeps any public_lb_auc already filled in by hand.
    """
    log = pd.read_csv(EXPERIMENTS_FILE) if EXPERIMENTS_FILE.exists() else pd.DataFrame(columns=EXPERIMENT_COLS)
    row = {"date": date, "notebook": notebook, "change": change,
           "val_auc": round(float(val_auc), 4), "public_lb_auc": public_lb_auc, "notes": notes}

    same = (log["notebook"] == notebook) & (log["change"] == change)
    if same.any():
        if public_lb_auc is None:                      # don't wipe a leaderboard score entered earlier
            row["public_lb_auc"] = log.loc[same, "public_lb_auc"].iloc[0]
        log = log[~same]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)[EXPERIMENT_COLS]
    log.to_csv(EXPERIMENTS_FILE, index=False)
