"""Rank of each converged key byte vs number of traces."""

import numpy as np
from cpa import normalise, hd_model


def rank_curve(traces, ct, key, ns, window=None, model_fn=hd_model, repeats=1, seed=0):
    """Returns ranks (len(ns), 16), mean peak of correct guess and of best wrong guess.

    With repeats > 1, each N is evaluated on random subsets and ranks are averaged.
    """
    rng = np.random.default_rng(seed)
    tr = traces if window is None else traces[:, window[0]:window[1]]
    models = [model_fn(ct, i) for i in range(16)]
    ranks = np.zeros((len(ns), 16))
    best = np.zeros(len(ns))
    wrong = np.zeros(len(ns))
    for a, n in enumerate(ns):
        for r in range(repeats):
            idx = np.arange(n) if repeats == 1 else rng.choice(len(tr), n, replace=False)
            tn = normalise(tr[idx])
            for i in range(16):
                s = np.abs(normalise(models[i][idx]).T @ tn).max(1)
                ranks[a, i] += (s > s[key[i]]).sum() / repeats
                best[a] += s[key[i]] / (16 * repeats)
                wrong[a] += np.delete(s, key[i]).max() / (16 * repeats)
    return ranks, best, wrong


def min_traces(ns, ranks):
    """Smallest N from which all bytes stay at rank 0 for every larger N."""
    ok = (ranks.max(1) == 0)
    for a in range(len(ns)):
        if ok[a:].all():
            return int(ns[a])
    return None
