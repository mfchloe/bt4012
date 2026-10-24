"""Recency-weighting strength x neighbour score smoothing, on the same splits.

Smoothing (post-hoc): final = (1-a) * own score + a * mean score of labelled
neighbours (nodes without labelled neighbours keep their own score). Scores are
ranked to [0, 1] first so the average is on a common scale.
"""
from pathlib import Path

exec(open(Path(__file__).with_name("validate_ensembles.py")).read().split('go("lgb struct")')[0])

DECAYS = (None, 0.95, 0.9, 0.85, 0.8)
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.5)


def smooth(v, p, a):
    s = pd.Series(rankdata(p) / len(p), index=v.txId.values)
    nb = neighbour_score_features(ke, s, prefix="sm")["sm_all_mean"]
    return np.where(nb.notna(), (1 - a) * s + a * nb.fillna(0), s)


ends = HARD + EASY
res = {}
for d in DECAYS:
    per = {a: [] for a in ALPHAS}
    for end in ends:
        fit = L[L.time_step <= end]; val = L[L.time_step > end]
        key = (end, True, 1, False)
        if key not in cache: cache[key] = build(fit, val, True)
        f, v, F = cache[key]
        p = preds(f, v, F, "lgb", d, end)
        for a in ALPHAS:
            per[a].append(roc_auc_score(v.label, smooth(v, p, a)))
    for a in ALPHAS:
        r = per[a]
        print(f"decay {str(d):5s} smooth {a:.1f}  easy {np.mean(r[2:]):.4f} hard {np.mean(r[:2]):.4f} | "
              + " ".join(f"{x:.4f}" for x in r), flush=True)
