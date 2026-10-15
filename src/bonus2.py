"""a wider search for the actual round-10 key"""

import numpy as np
import load
from aes_tables import HW, SBOX, INV_SBOX, INV_SR, SR
from cpa import normalise
from keyschedule import invert

FK = bytes.fromhex("91dec068e3face4599ce38f420863a30")


def m_hw_x(C, i, oi):
    return HW[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]

def m_hw_sbox_x(C, i, oi):
    return HW[SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]]

def m_hw_invsbox_x(C, i, oi):
    return HW[INV_SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]]

def m_hd_invsbox_csr(C, i, oi):
    s = INV_SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]
    return HW[s ^ C[:, oi(i), None]]

def m_hd_sbox_csr(C, i, oi):
    s = SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]
    return HW[s ^ C[:, oi(i), None]]

def m_hd_x_csr(C, i, oi):
    x = C[:, i, None] ^ np.arange(256, dtype=np.uint8)
    return HW[x ^ C[:, oi(i), None]]

def m_hd_sbox_sbox_csr(C, i, oi):
    s1 = SBOX[C[:, i, None] ^ np.arange(256, dtype=np.uint8)]
    s2 = SBOX[C[:, oi(i)]][:, None]
    return HW[s1 ^ s2]


def run_model(tn, C, mfn, oi, windows):
    best = None
    for lo, hi in windows:
        deltas, ratios, peaks = [], [], []
        for i in range(16):
            model = mfn(C, i, oi)
            rho = normalise(model).T @ tn[:, lo:hi]
            score = np.abs(rho).max(1)
            d = int(score.argmax())
            p = int(np.abs(rho[d]).argmax()) + lo
            s = np.sort(score)[::-1]
            ratios.append(float(s[0] / s[1]) if s[1] > 0 else 1e9)
            deltas.append(d)
            peaks.append(p)
        med = float(np.median(ratios))
        if best is None or med > best["median_ratio"]:
            best = {"window": [lo, hi], "median_ratio": med,
                    "min_ratio": float(np.min(ratios)),
                    "deltas": deltas, "peaks": peaks,
                    "ratios": [round(r, 2) for r in ratios],
                    "rk10": bytes(FK[i] ^ deltas[i] for i in range(16)).hex()}
    return best


def main():
    T, C = load.load()
    tn = normalise(T)
    print(f"traces {T.shape}")

    models = {
        "HW(X)":              m_hw_x,
        "HW(Sbox(X))":        m_hw_sbox_x,
        "HW(InvSbox(X))":     m_hw_invsbox_x,
        "HD(InvSbox(X),C_SR)": m_hd_invsbox_csr,
        "HD(Sbox(X),C_SR)":    m_hd_sbox_csr,
        "HD(X,C_SR)":          m_hd_x_csr,
        "HD(Sbox(X),Sbox(C_SR))": m_hd_sbox_sbox_csr,
    }
    other_idxs = {
        "INV_SR": lambda i: int(INV_SR[i]),
        "SR":     lambda i: int(SR[i]),
        "self":   lambda i: i,
    }
    windows = [(0, 200), (0, 45), (45, 60), (50, 80),
               (55, 75), (60, 80), (80, 120), (120, 200)]

    results = []
    for mname, mfn in models.items():
        for oname, ofn in other_idxs.items():
            r = run_model(tn, C, mfn, ofn, windows)
            r["model"] = mname
            r["other"] = oname
            results.append(r)

    results.sort(key=lambda r: r["median_ratio"], reverse=True)

    print("\n=== Ranked by median winner/runner-up ratio ===")
    for r in results[:15]:
        print(f"  {r['model']:25s} other={r['other']:7s} win={r['window']}  "
              f"med={r['median_ratio']:.2f} min={r['min_ratio']:.2f}  RK10={r['rk10']}")

    best = results[0]
    print("\n=== Best ===")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print(f"  MASTER: {invert(bytes.fromhex(best['rk10'])).hex()}")


if __name__ == "__main__":
    main()