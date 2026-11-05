"""Numbers of the cross-machine study of the article, from the data deposit (export of src/08_export_zenodo.py).

Usage: python3 cross_machine.py <export_dir> [n_boot]
Same statistics as scripts/gpu_check/analyze_cells.py, computed on gpu_trials.csv and
gpu_cells.csv. For each model, only the seeds trained in every cell that has that model are used.
- p90 etc.: empirical percentiles (np.percentile), in accuracy percentage points.
- seed sd: mean over cells of the standard deviation across seeds (ddof=1).
- transfer p: probability that the range between cells of a statistic, when every cell is resampled from the pooled
  runs of that model, is at least the observed range (bootstrap null, n_boot replicates, seed 20260904).
- Spearman and inversions: per pair of cells, over the common seeds; median over pairs.
- machine share: share of the variance that is not the seed main effect (cell effect plus seed x cell interaction).
- identical final weights: per pair of cells, trials of the same model and seed with equal final_weights_sha256.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

export_dir = sys.argv[1]
n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
trials = pd.read_csv(os.path.join(export_dir, "gpu_trials.csv"))
cells = pd.read_csv(os.path.join(export_dir, "gpu_cells.csv")).set_index("cell")
MODELS = ["SimplePerceptron", "HiddenLayerPerceptron", "CNN_3L", "DNN_6L", "CNN_14L", "CNN_14L_B10"]
STATS = {"mean": np.mean, "median": np.median, "P90": lambda x: np.percentile(x, 90),
         "P95": lambda x: np.percentile(x, 95), "P99": lambda x: np.percentile(x, 99), "max": np.max}
rng = np.random.default_rng(20260904)

print("cells:")
print(cells.to_string())
print(f"\ntrials: {len(trials)}; models: {trials.model.nunique()}")

rows = []
for model in MODELS:
    d = trials[trials.model == model]
    model_cells = sorted(d.cell.unique())
    seeds = set.intersection(*(set(d[d.cell == c].seed) for c in model_cells))
    M = np.array([[d[(d.cell == c) & (d.seed == s)].accuracy.iloc[0] for s in sorted(seeds)] for c in model_cells])
    M = 100 * M  # cells x seeds, in %
    k, n = M.shape
    row = {"model": model, "cells": k, "seeds": n, "seed_sd": M.std(axis=1, ddof=1).mean()}
    pooled = M.ravel()
    for name, f in STATS.items():
        per_cell = np.array([f(x) for x in M])
        obs = per_cell.max() - per_cell.min()
        idx = rng.integers(0, pooled.size, size=(n_boot, k, n))
        sample = pooled[idx]
        if name == "mean":
            null = sample.mean(axis=2)
        elif name == "median":
            null = np.median(sample, axis=2)
        elif name == "max":
            null = sample.max(axis=2)
        else:
            null = np.percentile(sample, int(name[1:]), axis=2)
        row[f"range_{name}"] = obs
        row[f"p_{name}"] = float(((null.max(axis=1) - null.min(axis=1)) >= obs - 1e-12).mean())
    rhos, invs = [], []
    for a, b in itertools.combinations(range(k), 2):
        u, w = M[a], M[b]
        rhos.append(stats.spearmanr(u, w).statistic if u.std() > 0 and w.std() > 0 else 1.0)
        du = np.sign(u[:, None] - u[None, :]); dw = np.sign(w[:, None] - w[None, :])
        valid = (du != 0) & (dw != 0)
        invs.append(100 * ((du != dw) & valid).sum() / valid.sum() if valid.sum() else 0.0)
    row["spearman_median"] = float(np.median(rhos))
    row["spearman_min"], row["spearman_max"] = float(np.min(rhos)), float(np.max(rhos))
    row["inversions_median"] = float(np.median(invs))
    gm = M.mean()
    v_seed = M.mean(axis=0).var(ddof=1)
    v_cell = M.mean(axis=1).var(ddof=1)
    v_res = ((M - M.mean(axis=0) - M.mean(axis=1)[:, None] + gm) ** 2).sum() / ((k - 1) * (n - 1))
    row["machine_share"] = 100 * (v_cell + v_res) / (v_seed + v_cell + v_res) if (v_seed + v_cell + v_res) else 0.0
    rows.append(row)

r = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250)
print("\nP90 range between cells (accuracy points) and in units of the seed sd:")
print(r[["cells", "seeds", "seed_sd", "range_P90"]].assign(range_in_sd=r.range_P90 / r.seed_sd).to_string(
    float_format="{:.4f}".format))
print(f"  largest P90 range: {r.range_P90.max():.4f} points ({r.range_P90.idxmax()})")
print("\nRange between cells, all statistics (points):")
print(r[[f"range_{s}" for s in STATS]].to_string(float_format="{:.4f}".format))
print(f"\nTransfer p (n_boot={n_boot}); high p = indistinguishable from seed sampling:")
print(r[[f"p_{s}" for s in STATS]].to_string(float_format="{:.4f}".format))
print("  median over models: " + ", ".join(f"{s} {r[f'p_{s}'].median():.4f}" for s in STATS))
print("\nSeed order between cells: Spearman (median, min, max over cell pairs), inversions (% of seed pairs):")
print(r[["spearman_median", "spearman_min", "spearman_max", "inversions_median"]].to_string(
    float_format="{:.2f}".format))
print("\nShare of the variance that is machine (cell + seed x cell), %:")
print(r[["machine_share"]].to_string(float_format="{:.1f}".format))

print("\nIdentical final weights per pair of cells (all models with common seeds):")
for a, b in itertools.combinations(cells.index, 2):
    x = trials[trials.cell == a].set_index(["model", "seed"]).final_weights_sha256
    y = trials[trials.cell == b].set_index(["model", "seed"]).final_weights_sha256
    common = x.index.intersection(y.index)
    same = (x.loc[common] == y.loc[common])
    by_model = same.groupby(level=0).agg(["sum", "size"])
    detail = "; ".join(f"{m} {int(s)}/{int(t)}" for m, (s, t) in by_model.iterrows() if s > 0)
    print(f"  {a}-{b} ({cells.loc[a,'gpu'][-9:]} {cells.loc[a,'torch']} / {cells.loc[b,'gpu'][-9:]} "
          f"{cells.loc[b,'torch']}): {int(same.sum())}/{len(common)}" + (f"  [{detail}]" if detail else ""))
init = trials.groupby(["model", "seed"]).init_weights_sha256.nunique()
print(f"\nTrials whose initial weights differ between cells: {int((init > 1).sum())} of {len(init)}")
