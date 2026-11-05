"""Switched-order probability tables of the article (Section 3.2), from the data deposit (export of src/08_export_zenodo.py).

Usage: python3 switched_probability.py <export_dir>
Same simulation as computeSwitchedProbabilityT (src/utils/compute_switched_probability.py): for
N = 1..5 runs per model, Monte Carlo draws from a normal fitted with the mean and std (ddof=0) of all the runs, and
Bootstrap resamples the runs with replacement, 4e4 simulations each. A simulation is switched when the order of the
models differs from the reference order. Two ways of summarising the N runs of each model: their best result (the one of
the article so far; reference = order of the maxima of all the runs) and their mean (reference = order of the means of
all the runs). Both statistics use the same draws, so N = 1 gives the same value in both. Seed 0 per set and method.
Prints the LaTeX rows of each table (in %) and checks that the reference orders by best, mean and P90 agree.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

SIMULATIONS = 40000
SAMPLE_SIZES = range(1, 6)

export_dir = sys.argv[1]
mnist = pd.read_csv(os.path.join(export_dir, "mnist_trials.csv"))
yolo = pd.read_csv(os.path.join(export_dir, "yolo_runs.csv"))
mnist = mnist[mnist["experiment"] == "main"]


def mnistData(models):
    return {m: mnist[mnist["model"] == m]["accuracy"].to_numpy() for m in models}


def yoloData(groups):
    return {g: yolo[yolo["group"] == g]["mAP50"].to_numpy() for g in groups}


SETS = [  # table, data
    ("HiddenLayerPerceptron vs DNN_6L (MNIST, accuracy)", mnistData(["HiddenLayerPerceptron", "DNN_6L"])),
    ("CNN_3L vs CNN_4L vs CNN_5L vs CNN_14L (MNIST, accuracy)", mnistData(["CNN_3L", "CNN_4L", "CNN_5L", "CNN_14L"])),
    ("VT vs VTHS (LLVIP, no equalization, mAP50)", yoloData(["LLVIP_VT_no_equalization", "LLVIP_VTHS_no_equalization"])),
]


def p90(v):
    return norm.ppf(0.9, loc=v.mean(), scale=v.std(ddof=1))


for title, data in SETS:
    names = list(data)
    values = [data[n] for n in names]
    if min(len(v) for v in values) < 2:
        raise SystemExit(f"{title}: no data")
    reference = {"best": np.argsort([v.max() for v in values]), "mean": np.argsort([v.mean() for v in values]),
                 "p90": np.argsort([p90(v) for v in values])}
    agree = all((reference[k] == reference["best"]).all() for k in reference)
    print(f"% {title}: n = {[len(v) for v in values]}; reference orders by best, mean and P90 "
          f"{'agree' if agree else 'DIFFER'}: " + " < ".join(names[i] for i in reference["best"]))
    if not agree:
        for k, order in reference.items():
            print(f"%   {k}: " + " < ".join(names[i] for i in order))
    rows = {n: {} for n in SAMPLE_SIZES}
    for method in ("mc", "bs"):
        rng = np.random.default_rng(0)
        for n in SAMPLE_SIZES:
            if method == "mc":
                draws = [rng.normal(v.mean(), v.std(), size=(SIMULATIONS, n)) for v in values]
            else:
                draws = [rng.choice(v, size=(SIMULATIONS, n), replace=True) for v in values]
            for stat, fn in (("best", np.max), ("mean", np.mean)):
                summary = np.column_stack([fn(d, axis=1) for d in draws])
                switched = np.any(np.argsort(summary, axis=1) != reference[stat], axis=1)
                rows[n][(stat, method)] = switched.mean()
    for n in SAMPLE_SIZES:  # in %, one decimal (Monte Carlo noise at 4e4 is about 0.2 points)
        r = rows[n]
        print(f"  {n} & " + " & ".join(f"{100 * r[k]:.1f}" for k in
                                       (("best", "mc"), ("best", "bs"), ("mean", "mc"), ("mean", "bs"))) + r" \\")
    print()
