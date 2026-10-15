"""End-to-end pipeline: fake-key CPA, characterisation, figures, results.json"""

import json
import os
import sys
import time
import numpy as np

import load
from cpa import attack, ranks_of
from characterise import rank_curve, min_traces
from keyschedule import invert
import plots

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")

def main():
    if not os.path.exists(os.path.join(load.DATA, "traces.npy")):
        load.convert(sys.argv[1] if len(sys.argv) > 1 else "Round10_traces.csv")
    os.makedirs(OUT, exist_ok=True)
    T, C = load.load()
    res = {"n_traces": int(len(T)), "n_samples": int(T.shape[1])}

    t0 = time.time()
    key, rhos, peaks = attack(T, C)
    res["cpa_seconds_all_traces"] = round(time.time() - t0, 2)
    res["fake_rk10"] = key.hex()
    res["fake_master_if_schedule_consistent"] = invert(key).hex()
    res["peak_samples"] = peaks
    res["peak_corr"] = [round(float(rhos[i][key[i], peaks[i]]), 4) for i in range(16)]
    res["ratio_to_runner_up"] = []
    for i in range(16):
        s = np.sort(np.abs(rhos[i]).max(1))[::-1]
        res["ratio_to_runner_up"].append(round(float(s[0] / s[1]), 2))
    assert ranks_of(rhos, key) == [0] * 16
    print("fake RK10:", key.hex())

    plots.corr_grid(rhos, key, os.path.join(OUT, "corr_grid.png"))
    plots.corr_grid(rhos, key, os.path.join(OUT, "corr_grid_zoom.png"), xlim=(35, 65))
    plots.mean_trace(T, os.path.join(OUT, "mean_trace.png"),
                     marks=[(int(np.median(peaks)), "final edge")])

    ns = np.unique(np.round(np.logspace(np.log10(50), np.log10(len(T)), 40)).astype(int))
    ranks, best, wrong = rank_curve(T, C, key, ns, repeats=1)
    n_min = min_traces(ns, ranks)
    res["min_traces_first_n"] = n_min

    # Robustness: random subsets, 20 repeats, success = all 16 bytes rank 0
    ns_fine = np.unique(np.round(np.logspace(np.log10(50), np.log10(5000), 30)).astype(int))
    rng = np.random.default_rng(1)
    from cpa import normalise, hd_model
    models = [hd_model(C, i) for i in range(16)]
    success = []
    for n in ns_fine:
        ok = 0
        for _ in range(20):
            idx = rng.choice(len(T), n, replace=False)
            tn = normalise(T[idx])
            good = True
            for i in range(16):
                s = np.abs(normalise(models[i][idx]).T @ tn).max(1)
                if s.argmax() != key[i]:
                    good = False
                    break
            ok += good
        success.append(ok / 20)
    res["success_rate"] = {int(n): s for n, s in zip(ns_fine, success)}
    n90 = next((int(n) for n, s in zip(ns_fine, success) if s >= 0.9
                and all(x >= 0.9 for x in success[list(ns_fine).index(n):])), None)
    res["min_traces_90pct_success"] = n90
    print("min traces (first-N):", n_min, " 90% success:", n90)

    plots.rank_plot(ns, ranks, os.path.join(OUT, "rank_vs_n.png"), n_min=n_min)
    plots.corr_vs_n(ns, best, wrong, os.path.join(OUT, "corr_vs_n.png"))
    np.savez(os.path.join(OUT, "rank_data.npz"), ns=ns, ranks=ranks, best=best, wrong=wrong,
             ns_fine=ns_fine, success=success)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
