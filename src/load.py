"""Parse Round10_traces.csv and cache as .npy files."""

import os
import sys
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def convert(csv_path):
    os.makedirs(DATA, exist_ok=True)
    df = pd.read_csv(csv_path, header=None, dtype={0: str})
    ct = np.array([bytes.fromhex(s.strip()) for s in df[0]], dtype=object)
    ct = np.frombuffer(b"".join(ct), dtype=np.uint8).reshape(-1, 16)
    tr = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    np.save(os.path.join(DATA, "traces.npy"), tr)
    np.save(os.path.join(DATA, "ct.npy"), ct)
    print("traces", tr.shape, "ciphertexts", ct.shape)


def load():
    return (np.load(os.path.join(DATA, "traces.npy")),
            np.load(os.path.join(DATA, "ct.npy")))


if __name__ == "__main__":
    convert(sys.argv[1] if len(sys.argv) > 1 else "Round10_traces.csv")
