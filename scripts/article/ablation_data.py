"""Ablation tables of the article, built from the data deposit (export of src/08_export_zenodo.py).

llvip_ablation: the valid YOLOv8 runs of the LLVIP equalization ablation, one row per run, with the factors fusion
(VT, VTHS), rgb_eq and th_eq (clahe, none) and the metrics R, P, mAP50 and mAP50-95. In these groups the evaluation
holds the person class only, so R and P are R_person and P_person.
mnist_ablation: the MNIST ablation trials (batch size x learning rate). With complete_only, only the iterations
trained in all nine conditions are kept, as in the analyses of the article.
"""
import os

import numpy as np
import pandas as pd


def llvip_ablation(export_dir):
    runs = pd.read_csv(os.path.join(export_dir, "yolo_runs.csv"))
    d = runs[runs["group"].str.startswith("LLVIP_")].copy()
    equalization = d["group"].str.split("_", n=2).str[2]
    d["fusion"] = d["group"].str.split("_").str[1]
    d["rgb_eq"] = np.where(equalization.isin(["rgb_equalization", "rgb_th_equalization"]), "clahe", "none")
    d["th_eq"] = np.where(equalization.isin(["th_equalization", "rgb_th_equalization"]), "clahe", "none")
    d = d.rename(columns={"R_person": "R", "P_person": "P", "mAP50_95": "mAP50-95"})
    return d[["run_id", "seed", "fusion", "rgb_eq", "th_eq", "R", "P", "mAP50", "mAP50-95"]].reset_index(drop=True)


def mnist_ablation(export_dir, complete_only=True):
    trials = pd.read_csv(os.path.join(export_dir, "mnist_trials.csv"))
    d = trials[trials["experiment"] == "ablation"]
    if complete_only:
        d = d[d["iteration"].map(d.groupby("iteration").size()) == 9]
    return d[["trial_id", "iteration", "seed", "batch_size", "learning_rate", "accuracy"]].reset_index(drop=True)
