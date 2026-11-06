"""Numbers quoted in the text of the article that no table prints, from the data deposit (export of src/08_export_zenodo.py).

Usage: python3 text_numbers.py <export_dir>
Prints one labelled value per line, grouped by section. Comparisons are strict: a run equal to a threshold does not exceed
it. The Monte Carlo values come from the fitted normal in closed form, so nothing here depends on a random draw.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

export_dir = sys.argv[1]
mnist = pd.read_csv(os.path.join(export_dir, "mnist_trials.csv"))
yolo = pd.read_csv(os.path.join(export_dir, "yolo_runs.csv"))
gpu = pd.read_csv(os.path.join(export_dir, "gpu_trials.csv"))
main = mnist[mnist["experiment"] == "main"]
Z90 = norm.ppf(0.9)


def fittedP90(values):
    return values.mean() + Z90 * values.std(ddof=1)


print("Introduction")
perceptron = main[main["model"] == "SimplePerceptron"]
print(f"  Simple Perceptron, range of the accuracy: {100 * (perceptron.accuracy.max() - perceptron.accuracy.min()):.2f} points")
print(f"  Simple Perceptron, slowest over fastest run: {perceptron.total_epochs.max() / perceptron.total_epochs.min():.1f} "
      f"in epochs, {perceptron.train_duration_s.max() / perceptron.train_duration_s.min():.1f} in time")

print("\nInformed training")
cnn14 = 100 * main[main["model"] == "CNN_14L"]["accuracy"].to_numpy()
threshold = np.percentile(cnn14, 90)
p_mc = 1 - norm.cdf(threshold, cnn14.mean(), cnn14.std())
p_bs = (cnn14 > threshold).mean()
print(f"  CNN_14L, empirical P90: {threshold:.2f}%; runs above it: {100 * p_bs:.2f}%")
print(f"  Runs before the first one above it, Monte Carlo: {(1 - p_mc) / p_mc:.1f} +- {np.sqrt(1 - p_mc) / p_mc:.1f}")
print(f"  Runs before the first one above it, Bootstrap: {(1 - p_bs) / p_bs:.1f} +- {np.sqrt(1 - p_bs) / p_bs:.1f}")

print("\nReplication and comparison: a single run of VTHS better than one of VT, LLVIP without equalization")
for metric in ["mAP50", "mAP50_95"]:
    vt = yolo[yolo["group"] == "LLVIP_VT_no_equalization"][metric].to_numpy()
    vths = yolo[yolo["group"] == "LLVIP_VTHS_no_equalization"][metric].to_numpy()
    mc = norm.cdf((vths.mean() - vt.mean()) / np.hypot(vths.std(), vt.std()))
    bootstrap = (vths[:, None] > vt[None, :]).mean()
    print(f"  {metric}: Monte Carlo {100 * mc:.0f}%, Bootstrap {100 * bootstrap:.0f}%")

print("\nSuggestions")
groups = {condition: data["accuracy"].to_numpy() for condition, data in mnist.groupby("condition")}
above = [100 * (v > fittedP90(v)).mean() for v in groups.values()]
print(f"  MNIST configurations, runs above the fitted P90: {min(above):.1f}% to {max(above):.1f}%")
print(f"  CNN_14L, empirical and fitted P90: {np.percentile(cnn14, 90):.2f}% and {fittedP90(cnn14):.2f}%")


def averageReach(percentile, n_runs):
    return 100 * np.mean([1 - (1 - (v > np.percentile(v, percentile)).mean()) ** n_runs for v in groups.values()])


for percentile, n_runs in [(90, 7), (95, 13), (98, 13)]:
    print(f"  MNIST average, at least one of {n_runs} runs above the P{percentile}: {averageReach(percentile, n_runs):.1f}%")
print(f"  Sampling error of the mean with five runs instead of one: {100 * (1 - 1 / np.sqrt(5)):.0f}% lower")
cells = {group: 100 * data["mAP50"] for group, data in yolo.groupby("group")}
a, b = cells["LLVIP_VTHS_no_equalization"], cells["LLVIP_VT_rgb_th_equalization"]
print(f"  VTHS without equalization against VT with both: mean {a.mean():.2f}% and {b.mean():.2f}%, sd {a.std():.2f} and "
      f"{b.std():.2f}, P90 {fittedP90(a):.2f}% and {fittedP90(b):.2f}%, best {a.max():.2f}% and {b.max():.2f}%")
vt, vths = cells["LLVIP_VT_no_equalization"], cells["LLVIP_VTHS_no_equalization"]
print(f"  VT over VTHS without equalization: {vt.mean() - vths.mean():.2f} points in the mean, "
      f"{fittedP90(vt) - fittedP90(vths):.2f} in the P90")

print("\nReplication across machines (appendix)")
print("  Model, range of the fitted P90 between cells (points), and share of the variance of a run (%): seed, cell, seed x cell")
for model, data in gpu.groupby("model"):
    table = data.pivot_table(index="seed", columns="cell", values="accuracy", aggfunc="first").dropna() * 100
    runs = table.to_numpy()
    n_seeds, n_cells = runs.shape
    seed_part = runs.mean(axis=1).var(ddof=1)
    cell_part = runs.mean(axis=0).var(ddof=1)
    interaction = runs - runs.mean(axis=1, keepdims=True) - runs.mean(axis=0, keepdims=True) + runs.mean()
    interaction_part = (interaction ** 2).sum() / ((n_seeds - 1) * (n_cells - 1))
    total = (seed_part + cell_part + interaction_part) or 1.0
    fitted = [fittedP90(table[cell].to_numpy()) for cell in table.columns]
    print(f"  {model}: {max(fitted) - min(fitted):.2f}; {100 * seed_part / total:.1f}, {100 * cell_part / total:.1f}, "
          f"{100 * interaction_part / total:.1f}")
