"""
AES lookup tables: S-box, inverse S-box, Hamming weight, ShiftRows index maps
Byte numbering is column-major
"""

import numpy as np


def _build_sbox():
    sbox = [0] * 256
    p = q = 1
    while True:
        # p iterates over GF(2^8)* by multiplying by 3; q tracks p^-1
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        rot = lambda x, s: ((x << s) | (x >> (8 - s))) & 0xFF
        sbox[p] = q ^ rot(q, 1) ^ rot(q, 2) ^ rot(q, 3) ^ rot(q, 4) ^ 0x63
        if p == 1:
            break
    sbox[0] = 0x63
    return sbox


SBOX = np.array(_build_sbox(), dtype=np.uint8)
INV_SBOX = np.zeros(256, dtype=np.uint8)
INV_SBOX[SBOX] = np.arange(256, dtype=np.uint8)

HW = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
SR = np.array([(i % 4) + 4 * (((i // 4) + (i % 4)) % 4) for i in range(16)])
INV_SR = SR.copy()
SHIFT_TO = np.argsort(SR)

if __name__ == "__main__":
    assert SBOX[0x00] == 0x63 and SBOX[0x19] == 0xD4 and SBOX[0x2E] == 0x31
    assert INV_SBOX[0xE9] == 0xEB
    assert INV_SR[1] == 5 and INV_SR[0] == 0 and INV_SR[2] == 10 and INV_SR[3] == 15
    print("INV_SR =", INV_SR.tolist())
    print("tables OK")
