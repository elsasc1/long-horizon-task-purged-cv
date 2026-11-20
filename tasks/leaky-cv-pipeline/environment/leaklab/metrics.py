import numpy as np


def auc(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y_true == 1], score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(order.size)
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, allv.size + 1, dtype=float)
    while i < allv.size:
        j = i
        while j + 1 < allv.size and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))
