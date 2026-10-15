"""Bivariate CPA for round-10 key"""

import numpy as np
import load
from aes_tables import HW, INV_SBOX
from keyschedule import invert

FK = bytes.fromhex("91dec068e3face4599ce38f420863a30")
G = np.arange(256, dtype=np.uint8)


def model_hw_x(C, i):
    return HW[C[:, i, None] ^ G].astype(np.float64)

def model_hw_invsbox(C, i):
    return HW[INV_SBOX[C[:, i, None] ^ G]].astype(np.float64)


def prep_products(Tc, t1_range, t2_range):
    """Return (prods_n, pairs): pairs (a,b) with a<b, prods_n shape (n_pairs, N)."""
    pairs = [(a, b) for a in t1_range for b in t2_range if a < b]
    prods = np.stack([Tc[:, a] * Tc[:, b] for a, b in pairs], axis=0)
    n = np.linalg.norm(prods, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return prods / n, pairs


def prep_model(m):
    m = m - m.mean(0)
    n = np.linalg.norm(m, axis=0, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def run(Tc, C, mfn, t1, t2, tag):
    prods_n, pairs = prep_products(Tc, t1, t2)
    print(f"\n=== {tag} — {len(pairs)} sample pairs ===")
    deltas, ratios, peaks, corrs = [], [], [], []
    for i in range(16):
        m = prep_model(mfn(C, i))
        rho = m.T @ prods_n.T
        absrho = np.abs(rho)
        score = absrho.max(1)
        d = int(score.argmax())
        pk = int(absrho[d].argmax())
        a, b = pairs[pk]
        s = np.sort(score)[::-1]
        r = float(s[0] / s[1]) if s[1] > 0 else 1e9
        deltas.append(d); ratios.append(r); peaks.append((a, b)); corrs.append(float(score[d]))
        print(f"  byte {i:2d}: delta=0x{d:02x}  RK10=0x{FK[i]^d:02x}  "
              f"peak=({a:3d},{b:3d})  |rho|={score[d]:.4f}  ratio={r:.2f}")
    print(f"  median ratio = {np.median(ratios):.2f}   min = {min(ratios):.2f}")
    print(f"  RK10 = {bytes(FK[i]^deltas[i] for i in range(16)).hex()}")
    return ratios, corrs


def main():
    T, C = load.load()
    print(f"traces {T.shape}")
    Tc = T.astype(np.float64)
    Tc -= Tc.mean(0)

    t1 = list(range(40, 60))
    t2 = list(range(42, 120))

    run(Tc, C, model_hw_x,      t1, t2, "Bivariate  model = HW(C^delta)")
    run(Tc, C, model_hw_invsbox,t1, t2, "Bivariate  model = HW(InvSbox(C^delta))")


if __name__ == "__main__":
    main()