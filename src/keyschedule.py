"""AES-128 key schedule: forward expansion and inversion from the round-10 key."""

import numpy as np
from aes_tables import SBOX

RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


def _g(word, rnd):
    w = [int(SBOX[b]) for b in word[1:] + word[:1]]
    w[0] ^= RCON[rnd]
    return w


def expand(key):
    """16-byte key to a list of 11 round keys"""
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        t = _g(w[i - 1], i // 4 - 1) if i % 4 == 0 else w[i - 1]
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return [bytes(sum(w[4 * r:4 * r + 4], [])) for r in range(11)]


def invert(rk10):
    """master key from the round-10 key by running the schedule backwards"""
    w = [None] * 44
    for j in range(4):
        w[40 + j] = list(rk10[4 * j:4 * j + 4])
    for i in range(43, 3, -1):
        t = _g(w[i - 1], i // 4 - 1) if i % 4 == 0 else w[i - 1]
        w[i - 4] = [a ^ b for a, b in zip(w[i], t)]
    return bytes(sum(w[0:4], []))


if __name__ == "__main__":
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    rks = expand(key)
    assert rks[1].hex() == "a0fafe1788542cb123a339392a6c7605"
    assert rks[10].hex() == "d014f9a8c9ee2589e13f0cc8b6630ca6"
    assert invert(rks[10]) == key
    print("key schedule OK")
