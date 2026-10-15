# Fake-Key CPA on FPGA AES-128

Side-channel analysis of 100,000 power traces from an AES-128 FPGA core
protected by a fake-key countermeasure on its final round.

**Team Synapse**: Rohan Rajeev Bharadwaj, Pugazhendhi G S, Aryan Anish

## Results

| Quantity | Value |
| --- | --- |
| Recovered fake round-10 key | `91dec068e3face4599ce38f420863a30` |
| Leaking samples | 49–50 (final register update) |
| Peak \|correlation\| per byte | 0.12 – 0.18 |
| Traces for full recovery (first N) | 2,028 |
| Traces for 90% success on random subsets | 3,105 |
| Actual round-10 key | not recoverable (see report) |

## Layout

    src/       attack pipeline (pure Python + NumPy)
    report/    IEEE-format report (main.tex, main.pdf, figures)
    out/       generated figures and results.json (not tracked)
    data/      input traces (not tracked)

## Setup

    pip install -r requirements.txt

## Run

    python src/load.py data/Round10_traces.csv     # parse CSV → .npy cache
    python src/run_all.py                          # fake-key CPA + characterisation

Figures and `out/results.json` are written to `out/`. Full pipeline runs
in about 2 minutes on a laptop.

## Bonus (actual-key recovery)

    python src/bonus.py         # univariate models on the LUT
    python src/bonus3.py        # wide model sweep
    python src/bonus_bivar.py   # bivariate centered-product CPA

All attempts return median winner/runner-up ratios ≈ 1 on the actual-key
candidates. The countermeasure holds against first- and second-order CPA. See report for the algebraic argument.

## Self-tests

    python src/aes_tables.py
    python src/keyschedule.py

Both should print `OK`.

## Report

See `report/main.pdf`. Recompile with `pdflatex main.tex` twice from `report/`.