"""faster wide search for actual round-10 key"""

import numpy as np
import load
from aes_tables import HW, SBOX, INV_SBOX, INV_SR, SR
from cpa import normalise
from keyschedule import invert

FK = bytes.fromhex("91dec068e3face4599ce38f420863a30")
G = np.arange(256, dtype=np.uint8)


def _norm_cols(x):
    x = x.astype(np.float64, copy=True)
    x -= x.mean(0)
    n = np.linalg.norm(x, axis=0)
    n[n == 0] = 1.0
    x /= n
    return x


def models():
    return {
        "HW(X)":              lambda C, i, oi: HW[C[:, i, None] ^ G],
        "HW(Sbox(X))":        lambda C, i, oi: HW[SBOX[C[:, i, None] ^ G]],
        "HW(InvSbox(X))":     lambda C, i, oi: HW[INV_SBOX[C[:, i, None] ^ G]],
        "HD(InvSbox(X),C_SR)": lambda C, i, oi: HW[INV_SBOX[C[:, i, None] ^ G]
                                                ^ C[:, oi(i), None]],
        "HD(Sbox(X),C_SR)":   lambda C, i, oi: HW[SBOX[C[:, i, None] ^ G]
                                                ^ C[:, oi(i), None]],
        "HD(X,C_SR)":         lambda C, i, oi: HW[(C[:, i, None] ^ G)
                                                ^ C[:, oi(i), None]],
    }


OTHERS = {
    "INV_SR": lambda i: int(INV_SR[i]),
    "SR":     lambda i: int(SR[i]),
    "self":   lambda i: i,
}
WINDOWS = [(0, 200), (0, 45), (40, 60), (45, 60), (50, 80),
           (55, 75), (60, 80), (80, 120), (120, 200)]


def main():
    T, C = load.load()
    print(f"traces {T.shape}")
    tn = _norm_cols(T)

    results = []
    for mname, mfn in models().items():
        for oname, ofn in OTHERS.items():
            # compute 16 rho matrices once, then evaluate windows
            per_window = {w: {"deltas": [], "peaks": [],
                              "ratios": []} for w in WINDOWS}
            for i in range(16):
                model = mfn(C, i, ofn)
                rho = _norm_cols(model).T @ tn
                absrho = np.abs(rho)
                for w in WINDOWS:
                    lo, hi = w
                    score = absrho[:, lo:hi].max(1)
                    d = int(score.argmax())
                    p = int(absrho[d, lo:hi].argmax()) + lo
                    s = np.sort(score)[::-1]
                    per_window[w]["deltas"].append(d)
                    per_window[w]["peaks"].append(p)
                    per_window[w]["ratios"].append(
                        float(s[0] / s[1]) if s[1] > 0 else 1e9)
                print(f"  {mname:25s} {oname:7s} byte {i:2d} done", flush=True)
            for w, d in per_window.items():
                rk10 = bytes(FK[i] ^ d["deltas"][i] for i in range(16))
                results.append({
                    "model": mname, "other": oname, "window": list(w),
                    "median_ratio": float(np.median(d["ratios"])),
                    "min_ratio": float(np.min(d["ratios"])),
                    "deltas": d["deltas"], "peaks": d["peaks"],
                    "ratios": [round(r, 2) for r in d["ratios"]],
                    "rk10": rk10.hex(),
                })

    results.sort(key=lambda r: r["median_ratio"], reverse=True)
    print("\n=== Top 15 ===")
    for r in results[:15]:
        print(f"  {r['model']:25s} other={r['other']:7s} win={r['window']}  "
              f"med={r['median_ratio']:6.2f} min={r['min_ratio']:6.2f}  "
              f"RK10={r['rk10']}")

    best = results[0]
    print("\n=== Best ===")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  MASTER: {invert(bytes.fromhex(best['rk10'])).hex()}")


if __name__ == "__main__":
    main()