"""
Attempt actual round-10 key recovery via the LUT address leakage

Countermeasure: R = F XOR delta. LUT input X = C XOR delta
Assumes the LUT as a ROM whose power scales with HW(X[i])
"""

import json
import numpy as np

import load
from aes_tables import HW, INV_SBOX, INV_SR
from cpa import normalise
from keyschedule import invert
import plots

FK = bytes.fromhex("91dec068e3face4599ce38f420863a30")


def attack_byte(tn, C, i, model_fn, lo=0, hi=None):
    hi = hi or tn.shape[1]
    model = model_fn(C, i)                # (N, 256)
    rho = normalise(model).T @ tn[:, lo:hi]
    score = np.abs(rho).max(1)
    d = int(score.argmax())
    peak = int(np.abs(rho[d]).argmax()) + lo
    s = np.sort(score)[::-1]
    ratio = float(s[0] / s[1]) if s[1] > 0 else float("inf")
    return d, rho, peak, float(score[d]), ratio


def model_hw_x(C, i):
    return HW[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]


def model_hw_invsbox_x(C, i):
    return HW[INV_SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]]


def model_hd_invsbox(C, i):
    s = INV_SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]
    return HW[s ^ C[:, INV_SR[i], None]]


def scan(models, T, C, windows):
    tn = normalise(T)
    out = []
    for name, fn in models:
        for lo, hi in windows:
            deltas, ratios, peaks, corrs = [], [], [], []
            for i in range(16):
                d, _, p, c, r = attack_byte(tn, C, i, fn, lo, hi)
                deltas.append(d); ratios.append(r); peaks.append(p); corrs.append(c)
            out.append({
                "model": name,
                "window": [lo, hi],
                "median_ratio": float(np.median(ratios)),
                "min_ratio": float(np.min(ratios)),
                "rk10": bytes(FK[i] ^ deltas[i] for i in range(16)).hex(),
                "deltas": deltas,
                "peaks": peaks,
                "corrs": [round(c, 4) for c in corrs],
                "ratios": [round(r, 2) for r in ratios],
            })
    out.sort(key=lambda r: r["median_ratio"], reverse=True)
    return out


def main():
    T, C = load.load()
    print(f"traces {T.shape}, ciphertexts {C.shape}")

    models = [
        ("HW(X)",            model_hw_x),
        ("HW(InvSbox(X))",   model_hw_invsbox_x),
        ("HD(InvSbox,C_SR)", model_hd_invsbox),
    ]
    windows = [(0, 200), (45, 60), (50, 80), (80, 120), (120, 200)]

    results = scan(models, T, C, windows)

    print("\n=== Ranked by median winner/runner-up ratio ===")
    for r in results[:8]:
        print(f"  {r['model']:18s}  win={r['window']}  "
              f"med_ratio={r['median_ratio']:.2f}  min={r['min_ratio']:.2f}  "
              f"RK10={r['rk10']}")

    best = results[0]
    print("\n=== Best ===")
    print(f"  model  : {best['model']}")
    print(f"  window : {best['window']}")
    print(f"  RK10   : {best['rk10']}")
    print(f"  deltas : {[hex(d) for d in best['deltas']]}")
    print(f"  peaks  : {best['peaks']}")
    print(f"  ratios : {best['ratios']}")
    master = invert(bytes.fromhex(best["rk10"]))
    print(f"  MASTER : {master.hex()}")

    with open("../out/bonus_results.json", "w") as f:
        json.dump({"all": results, "best": best,
                   "master": master.hex()}, f, indent=2)
    print("\nwrote out/bonus_results.json")


if __name__ == "__main__":
    main()