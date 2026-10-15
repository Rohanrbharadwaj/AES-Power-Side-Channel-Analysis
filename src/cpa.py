"""CPA on the AES final-round state register."""

import numpy as np
from aes_tables import INV_SBOX, HW, INV_SR

GUESSES = np.arange(256, dtype=np.uint8)


def normalise(x):
    x = x.astype(np.float64)
    x -= x.mean(0)
    n = np.linalg.norm(x, axis=0)
    n[n == 0] = 1.0
    return x / n


def hd_model(ct, i):
    s9 = INV_SBOX[ct[:, i, None] ^ GUESSES]
    return HW[s9 ^ ct[:, INV_SR[i], None]]


def cpa(tn, model):
    return normalise(model).T @ tn


def attack(traces, ct, model_fn=hd_model):
    tn = normalise(traces)
    key, rhos, peaks = [], [], []
    for i in range(16):
        rho = cpa(tn, model_fn(ct, i))
        score = np.abs(rho).max(1)
        k = int(score.argmax())
        key.append(k)
        rhos.append(rho)
        peaks.append(int(np.abs(rho[k]).argmax()))
    return bytes(key), rhos, peaks


def ranks_of(rhos, key):
    out = []
    for i, rho in enumerate(rhos):
        score = np.abs(rho).max(1)
        out.append(int((score > score[key[i]]).sum()))
    return out
