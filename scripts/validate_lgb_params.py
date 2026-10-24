"""Stage-2 LightGBM settings on top of the graph features (decay 0.95).

Stage 1 always uses the default settings; only the final model's settings change.
"""
from pathlib import Path

exec(open(Path(__file__).with_name("validate_ensembles.py")).read().split('go("lgb struct")')[0])

ends = HARD + EASY
for end in ends:  # build stage-1 features once with the default settings
    cache[(end, True, 1, False)] = build(L[L.time_step <= end], L[L.time_step > end], True)
DEFAULT = dict(P)

GRID = {
    "default (15 leaves, cols 0.3)": {},
    "31 leaves, cols 0.6, min 100": dict(num_leaves=31, colsample_bytree=0.6, min_child_samples=100),
    "7 leaves, cols 0.3": dict(num_leaves=7),
    "cols 0.15": dict(colsample_bytree=0.15),
    "1000 trees, lr 0.01": dict(n_estimators=1000, learning_rate=0.01),
    "1000 trees, lr 0.02": dict(n_estimators=1000),
    "subsample 0.7": dict(subsample=0.7, subsample_freq=1),
    "min 50, reg_lambda 5": dict(min_child_samples=50, reg_lambda=5),
}
for name, kw in GRID.items():
    P.clear(); P.update({**DEFAULT, **kw})
    r = []
    for end in ends:
        f, v, F = cache[(end, True, 1, False)]
        r.append(roc_auc_score(v.label, preds(f, v, F, "lgb", 0.95, end)))
    print(f"{name:32s} easy {np.mean(r[2:]):.4f} hard {np.mean(r[:2]):.4f} | " + " ".join(f"{x:.4f}" for x in r), flush=True)
