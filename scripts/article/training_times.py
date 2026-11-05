"""Training time per run for the article, computed from the data deposit (export of src/08_export_zenodo.py).

Usage: python3 training_times.py <export_dir>
MNIST: train_duration_s of mnist_trials.csv (main, batch_size and ablation experiments).
YOLOv8: train_duration_h of yolo_runs.csv (the deposit holds the valid runs only).
Prints median, interquartile range, number of runs and total GPU hours per model or group.
"""
import os
import sys

import pandas as pd

export_dir = sys.argv[1]
mnist = pd.read_csv(os.path.join(export_dir, "mnist_trials.csv"))
yolo = pd.read_csv(os.path.join(export_dir, "yolo_runs.csv"))


def summary(values, unit_scale):
    v = values.dropna() * unit_scale
    return {"n": len(v), "median": v.median(), "q1": v.quantile(0.25), "q3": v.quantile(0.75),
            "min": v.min(), "max": v.max(), "total_h": v.sum() / 3600 if unit_scale == 1 else v.sum() / 60}


rows = []
for (experiment, model), d in mnist.groupby(["experiment", "model"], sort=False):
    rows.append({"set": f"MNIST {experiment}", "item": model, **summary(d["train_duration_s"], 1)})
mnist_tab = pd.DataFrame(rows)
print("== MNIST, seconds per run (total in GPU hours)")
print(mnist_tab.to_string(index=False, float_format="{:.1f}".format))
for experiment, d in mnist.groupby("experiment", sort=False):
    print(f"   {experiment}: {len(d)} runs, {d['train_duration_s'].sum() / 3600:.1f} h in total")

rows = []
valid = yolo  # the deposit holds the valid runs only
for group, d in valid.groupby("group", sort=False):
    rows.append({"set": "YOLOv8", "item": group, **summary(d["train_duration_h"], 60)})
yolo_tab = pd.DataFrame(rows)
print("\n== YOLOv8 valid runs, minutes per run (total in GPU hours)")
print(yolo_tab.to_string(index=False, float_format="{:.1f}".format))
print(f"   {len(valid)} valid runs, {valid['train_duration_h'].sum():.1f} h in total")
