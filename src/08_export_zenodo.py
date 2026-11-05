#!/usr/bin/env python3
# encoding: utf-8

"""
Build the public data deposit (Zenodo) of the training-variance manuscript
on MNIST classification and YOLOv8 object detection.

Every record is rebuilt from a whitelist of fields; no data file is copied
as is. Local paths, run folder names, training timestamps (the keys of the
MNIST training YAMLs), process ids and dataset configuration files never
reach the output. Training settings are read from the training scripts and
from the arguments of each run. Before finishing, every output file is
checked: the curve files must parse as numbers only, and the text files are
scanned for the local user and host, home paths, run folder names, e-mail
and IP addresses and timestamps. Any hit aborts the export.

Output (ignored by git), by default `<repo>/zenodo_export/`:

  mnist_trials.csv                one row per MNIST training trial
  mnist_curves.jsonl.gz           per-epoch test accuracy of the MNIST trials that logged it
  yolo_groups.csv                 YOLOv8 experiment groups and their training settings
  yolo_runs.csv                   one row per YOLOv8 training run
  yolo_curves.jsonl.gz            per-epoch losses and validation metrics of the YOLOv8 runs
  gpu_cells.csv                   cells (GPU and software stack) of the cross-GPU study
  gpu_trials.csv                  one row per training trial of the cross-GPU study
  gpu_curves.jsonl.gz             per-epoch curves of the cross-GPU trials
  README.md                       contents, field dictionary, licence

Usage:
    python src/08_export_zenodo.py [--out PATH] [--article CITATION] [--code-doi DOI]

`--article` replaces the citation of the article (by default, the manuscript
as submitted) and `--code-doi` adds the DOI of the archived code release, so
that the archived code can build the final deposit.
"""

import argparse
import ast
import csv
import getpass
import glob
import gzip
import json
import os
import re
import shutil
import socket

import yaml

from utils import output_path
from utils.log_utils import log, bcolors
from utils.plot_distribution import display_name


_SRC = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC)
_YOLO_PATH = os.path.join(output_path, "YOLO")
# Validity of every YOLOv8 run, written by check_yolo_results.py (utils/yolo_run_validity.py)
_YOLO_INVENTORY = os.path.join(_YOLO_PATH, "yolo_runs_inventory.csv")
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# Default citation of the manuscript and address of the code. The final
# citation and the DOI of the archived code are given on the command line
# (--article, --code-doi); the export warns while any TODO remains.
ARTICLE_CITATION = ("Heredia-Aguado, E., Rujano, A., Ballesta, M., Valiente, D., "
                    "Gil, A. The Impact of Performance Variability on Neural "
                    "Network Evaluation and its Implications for Model Design. "
                    "Manuscript submitted for publication.")
CODE_URL = "https://github.com/enheragu/mnist-learning-playground"

# MNIST experiments, in the order of the manuscript.
_MAIN_MODELS = ["SimplePerceptron", "HiddenLayerPerceptron", "DNN_6L", "CNN_3L",
                "CNN_4L", "CNN_5L", "CNN_14L", "BatchNormMaxoutNetInNet"]
_BATCH_MODELS = ["CNN_14L_B10", "CNN_14L_B25", "CNN_14L_B50", "CNN_14L_B80"]
_MAIN_FILE = "randomseed_training_metrics.yaml"
_ABLATION_FILE = os.path.join("CNN_14L_Ablation", "training_metrics.yaml")
# Seed and curve are logged only by the later trials of the main experiments.
_MAIN_KEYS = {"accuracy", "best_epoch", "total_epochs", "train_duration",
              "seed", "accuracy_plot"}
_ABLATION_KEYS = _MAIN_KEYS | {"batch_size", "learning_rate", "finish_timetag"}

# YOLOv8 groups used in the manuscript, in its order: tag of the analysis
# code (`check_yolo_results.py`, also the label of the sampling figures),
# folder under output_data/YOLO, dataset, input images and number of classes.
_YOLO_GROUPS = [
    ("kaist_day_vths_v2", "variance_day_vths_v2", "KAIST day", "VTHS fusion", 3),
    ("kaist_day_hsvt",    "variance_day_hsvt",    "KAIST day", "HSVT fusion", 3),
] + [
    (f"LLVIP_{fusion}_{eq}_equalization",
     f"variance_{fusion.lower()}_llvip/variance_{fusion.lower()}_llvip_{eq}_equalization",
     "LLVIP", f"{fusion} fusion", 1)
    for fusion in ("VTHS", "VT") for eq in ("no", "rgb_th", "th", "rgb")
]
_EQUALIZATION = {"no": ("none", "none"), "rgb": ("clahe", "none"),
                 "th": ("none", "clahe"), "rgb_th": ("clahe", "clahe")}
_YOLO_ARGS = [("model", "model_config"), ("pretrained", "pretrained"),
              ("epochs", "max_epochs"), ("patience", "patience"),
              ("batch", "batch_size"), ("imgsz", "image_size"),
              ("optimizer", "optimizer"), ("lr0", "lr0"),
              ("deterministic", "deterministic")]
_CSV_COLUMNS = {
    "epoch": "epoch",
    "train/box_loss": "train_box_loss", "train/cls_loss": "train_cls_loss",
    "train/dfl_loss": "train_dfl_loss",
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50_95",
    "val/box_loss": "val_box_loss", "val/cls_loss": "val_cls_loss",
    "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr_pg0", "lr/pg1": "lr_pg1", "lr/pg2": "lr_pg2",
}

MNIST_FIELDS = ["trial_id", "experiment", "model", "condition", "batch_size",
                "learning_rate", "iteration", "seed", "accuracy", "best_epoch",
                "total_epochs", "train_duration_s"]
YOLO_GROUP_FIELDS = ["group", "label", "dataset", "input", "rgb_equalization",
                     "thermal_equalization", "n_classes"] + \
                    [name for _, name in _YOLO_ARGS] + \
                    ["n_runs"]
YOLO_RUN_FIELDS = ["run_id", "group", "seed", "cache", "best_epoch",
                   "last_epoch", "train_duration_h", "mAP50", "mAP50_95",
                   "P_person", "R_person", "mAP50_person", "mAP50_95_person"]

# Cross-GPU study (scripts/gpu_check): one output folder per machine or session, named
# output_data_gpucheck_<tag>; the cal-* folders are timing calibrations and stay out.
# A cell is a GPU architecture with a software stack, identified from each record.
_GPU_GLOB = os.path.join(_REPO_ROOT, "output_data_gpucheck_*")
_GPU_CALIBRATION = "output_data_gpucheck_cal-"
_GPU_CELLS = [("C01", "sm_89", "2.9.1+cu128"), ("C02", "sm_120", "2.9.1+cu128"),
              ("C03", "sm_75", "2.9.1+cu128"), ("C04", "sm_89", "2.7.1+cu118"),
              ("C05", "sm_75", "2.7.1+cu118"), ("C06", "sm_60", "2.7.1+cu118"),
              ("C07", "sm_120", "2.7.1+cu128")]
_GPU_MODELS = ["SimplePerceptron", "HiddenLayerPerceptron", "CNN_3L", "DNN_6L",
               "CNN_14L", "CNN_14L_B10"]
_GPU_CURVES = [("accuracy_plot", "test_accuracy"), ("train_accuracy_plot", "train_accuracy"),
               ("train_loss_plot", "train_loss"), ("test_loss_plot", "test_loss")]
GPU_CELL_FIELDS = ["cell", "gpu", "capability", "sm_count", "torch", "n_trials"]
GPU_TRIAL_FIELDS = ["trial_id", "cell", "model", "seed_index", "seed", "accuracy",
                    "test_loss_final", "best_epoch", "total_epochs", "train_duration_s",
                    "init_weights_sha256", "final_weights_sha256"]


def _loadYaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=_YAML_LOADER) or {}


def _naturalKey(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


# ---------------------------------------------------------------------
# Training settings, read from the training scripts
# ---------------------------------------------------------------------

def _moduleConstants(path, names):
    """ Top-level `name = literal` assignments of a script. """
    found = {}
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            found[node.targets[0].id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise SystemExit(f"{os.path.basename(path)}: constants not found {sorted(missing)}")
    return found


def _classBatchSizes(path):
    """ `batch_size` class attributes defined in a model file. """
    sizes = {}
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for st in node.body:
                if isinstance(st, ast.Assign) and \
                        getattr(st.targets[0], "id", None) == "batch_size":
                    sizes[node.name] = ast.literal_eval(st.value)
    return sizes


def readSettings():
    main = _moduleConstants(os.path.join(_SRC, "00_train_models.py"),
                            ["learning_rate", "patience", "num_epochs"])
    ablation = _moduleConstants(os.path.join(_SRC, "01_train_ablation_test.py"),
                                ["patience", "num_epochs", "total_iterations",
                                 "batch_conditions", "learning_rates"])
    default_batch = _classBatchSizes(os.path.join(_SRC, "models", "BaseModel.py"))["BaseModelTrainer"]
    batch = {m: default_batch for m in _MAIN_MODELS}
    batch.update(_classBatchSizes(os.path.join(_SRC, "models", "BatchSizeStudy.py")))
    return main, ablation, batch


# ---------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------

def _mnistCurve(e, stats):
    if "accuracy_plot" not in e:
        return None
    curve = [float(v) for v in e["accuracy_plot"]]
    if len(curve) != e["total_epochs"] + 1 or abs(max(curve) - e["accuracy"]) > 1e-12:
        stats["mnist_curve_mismatch"] += 1
    return curve


def _count(stats, experiment, condition, e):
    c = stats["mnist_counts"].setdefault((experiment, condition), [0, 0, 0])
    c[0] += 1
    c[1] += "seed" in e
    c[2] += "accuracy_plot" in e


def exportMnist(out_dir, main_cfg, batch, stats):
    rows = []
    for experiment, models in (("main", _MAIN_MODELS), ("batch_size", _BATCH_MODELS)):
        for model in models:
            data = _loadYaml(os.path.join(output_path, model, _MAIN_FILE))
            seeds = [e["seed"] for e in data.values() if "seed" in e]
            if len(set(seeds)) != len(seeds):
                raise SystemExit(f"{model}: repeated seeds")
            for e in data.values():
                extra = set(e) - _MAIN_KEYS
                if extra:
                    raise SystemExit(f"{model}: unexpected fields {sorted(extra)}")
                rows.append(({
                    "experiment": experiment,
                    "model": "CNN_14L" if experiment == "batch_size" else model,
                    "condition": model, "batch_size": batch[model],
                    "learning_rate": main_cfg["learning_rate"], "seed": e.get("seed"),
                    "accuracy": e["accuracy"], "best_epoch": e["best_epoch"],
                    "total_epochs": e["total_epochs"],
                    "train_duration_s": e["train_duration"],
                }, _mnistCurve(e, stats)))
                _count(stats, experiment, model, e)

    ablation = _loadYaml(os.path.join(output_path, _ABLATION_FILE))
    abl_rows = []
    for iteration, (key, conditions) in enumerate(ablation.items(), 1):
        for e in conditions.values():
            extra = set(e) - _ABLATION_KEYS
            if extra:
                raise SystemExit(f"ablation: unexpected fields {sorted(extra)}")
            if int(str(key).split("_")[-1]) != e["seed"]:
                raise SystemExit("ablation: iteration key does not match its seed")
            condition = f"CNN_14L_B{e['batch_size']}_L{e['learning_rate']}"
            abl_rows.append(((e["batch_size"], e["learning_rate"], iteration), {
                "experiment": "ablation", "model": "CNN_14L", "condition": condition,
                "batch_size": e["batch_size"], "learning_rate": e["learning_rate"],
                "iteration": iteration, "seed": e["seed"],
                "accuracy": e["accuracy"], "best_epoch": e["best_epoch"],
                "total_epochs": e["total_epochs"],
                "train_duration_s": e["train_duration"],
            }, _mnistCurve(e, stats)))
            _count(stats, "ablation", condition, e)
    abl_rows.sort(key=lambda t: t[0])
    rows += [(row, curve) for _, row, curve in abl_rows]
    stats["ablation_iterations"] = len(ablation)
    stats["ablation_complete"] = sum(1 for c in ablation.values() if len(c) == 9)

    n_curves = 0
    with open(os.path.join(out_dir, "mnist_trials.csv"), "w", newline="") as f, \
         gzip.open(os.path.join(out_dir, "mnist_curves.jsonl.gz"), "wt") as fc:
        w = csv.DictWriter(f, fieldnames=MNIST_FIELDS)
        w.writeheader()
        for trial_id, (row, curve) in enumerate(rows, 1):
            w.writerow({"trial_id": trial_id, **row})
            if curve:
                n_curves += 1
                fc.write(json.dumps({"trial_id": trial_id, "test_accuracy": curve},
                                    separators=(",", ":")) + "\n")
    log(f"  -> mnist_trials.csv ({len(rows)} trials), "
        f"mnist_curves.jsonl.gz ({n_curves} curves)", color=bcolors.OKGREEN)
    return len(rows), n_curves


# ---------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------

def _readCurve(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader)]
        if set(header) != set(_CSV_COLUMNS):
            raise SystemExit(f"unexpected columns in a results.csv: {header}")
        cols = {h: [] for h in header}
        for line in reader:
            for h, v in zip(header, line):
                cols[h].append(float(v))
    rec = {_CSV_COLUMNS[h]: cols[h] for h in _CSV_COLUMNS}
    rec["epoch"] = [int(v) for v in rec["epoch"]]
    return rec


def _settingValue(values):
    """ The common value of a setting, or each value with its count. """
    counts = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    if len(counts) == 1:
        return next(iter(counts))
    return "; ".join(f"{v} ({n})" for v, n in sorted(counts.items(), key=lambda t: -t[1]))


def exportYolo(out_dir, stats):
    if not os.path.exists(_YOLO_INVENTORY):
        raise SystemExit("inventory of the YOLOv8 runs not found; run check_yolo_results.py first")
    with open(_YOLO_INVENTORY, newline="") as f:
        inventory = {row["run"]: row for row in csv.DictReader(f)}
    group_rows, run_rows = [], []
    run_id = 0
    curves = gzip.open(os.path.join(out_dir, "yolo_curves.jsonl.gz"), "wt")
    for tag, folder, dataset, inputs, n_classes in _YOLO_GROUPS:
        base = os.path.join(_YOLO_PATH, folder)
        if not os.path.isdir(base):
            raise SystemExit(f"{tag}: folder not found")
        eq = re.search(r"_(no|rgb_th|th|rgb)_equalization$", tag)
        rgb_eq, th_eq = _EQUALIZATION[eq.group(1)] if eq else ("none", "none")

        # Only the valid runs, the ones used in the article (utils/yolo_run_validity.py)
        found = []
        for dirpath, dirnames, filenames in os.walk(base):
            if "args.yaml" not in filenames:
                continue
            inv = inventory.get(os.path.relpath(dirpath, _YOLO_PATH))
            if inv is None or inv["tag"] != tag:
                raise SystemExit(f"{tag}: a run is missing from the inventory; run check_yolo_results.py first")
            if inv["valid"] != "True":
                continue
            if "results.yaml.cached" not in filenames or "results.csv" not in filenames:
                raise SystemExit(f"{tag}: a valid run has no final evaluation summary or no curve; "
                                 "run check_yolo_results.py first")
            if os.path.getmtime(os.path.join(dirpath, "results.yaml")) > \
                    os.path.getmtime(os.path.join(dirpath, "results.yaml.cached")):
                raise SystemExit(f"{tag}: summary of a run is older than its results; "
                                 "run check_yolo_results.py first")
            found.append((os.path.relpath(dirpath, base), dirpath))
        found.sort(key=lambda t: _naturalKey(t[0]))
        if not found:
            raise SystemExit(f"{tag}: no valid runs")

        args_all, seeds = [], []
        for _, dirpath in found:
            run_id += 1
            args = _loadYaml(os.path.join(dirpath, "args.yaml"))
            args_all.append(args)
            seeds.append(args["seed"])
            curve = _readCurve(os.path.join(dirpath, "results.csv"))
            c = _loadYaml(os.path.join(dirpath, "results.yaml.cached"))
            if len(c["dataset_info"]["names"]) != n_classes:
                raise SystemExit(f"{tag}: {len(c['dataset_info']['names'])} classes, "
                                 f"expected {n_classes}")
            if (c["rgb_equalization"], c["thermal_equalization"]) != (rgb_eq, th_eq):
                raise SystemExit(f"{tag}: equalization does not match the group")
            td, val = c["train_data"], c["validation_0"]["data"]
            run_rows.append({
                "run_id": run_id, "group": tag, "seed": args["seed"], "cache": args.get("cache"),
                "best_epoch": td["epoch_best_fit_index"],
                "last_epoch": td["epoch_executed_index"],
                "train_duration_h": td["train_duration_h"],
                "mAP50": val["all"]["mAP50"], "mAP50_95": val["all"]["mAP50-95"],
                "P_person": val["person"]["P"], "R_person": val["person"]["R"],
                "mAP50_person": val["person"]["mAP50"],
                "mAP50_95_person": val["person"]["mAP50-95"],
            })
            # Same epoch indexing in the summary and in the curve file
            i = curve["epoch"].index(td["epoch_best_fit_index"]) \
                if td["epoch_best_fit_index"] in curve["epoch"] else None
            fit = None if i is None else \
                0.1 * curve["mAP50"][i] + 0.9 * curve["mAP50_95"][i]
            if curve["epoch"][-1] != td["epoch_executed_index"] or fit is None or \
                    abs(fit - td["best_fitness"]) > 1e-4:
                stats["yolo_epoch_mismatch"] += 1
            curves.write(json.dumps({"run_id": run_id, **curve},
                                    separators=(",", ":")) + "\n")
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"{tag}: repeated seeds among the valid runs")

        settings = {}
        for key, name in _YOLO_ARGS:
            values = [os.path.basename(str(a[key])) if key == "model" else a[key]
                      for a in args_all]
            settings[name] = _settingValue(values)
        group_rows.append({
            "group": tag, "label": display_name(tag), "dataset": dataset,
            "input": inputs, "rgb_equalization": rgb_eq,
            "thermal_equalization": th_eq, "n_classes": n_classes, **settings,
            "n_runs": len(found),
        })
        log(f"  {tag:32s} runs {len(found):3d}")
    curves.close()

    with open(os.path.join(out_dir, "yolo_groups.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=YOLO_GROUP_FIELDS)
        w.writeheader()
        w.writerows(group_rows)
    with open(os.path.join(out_dir, "yolo_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=YOLO_RUN_FIELDS)
        w.writeheader()
        w.writerows(run_rows)
    log(f"  -> yolo_groups.csv ({len(group_rows)} groups), yolo_runs.csv ({run_id} runs), "
        f"yolo_curves.jsonl.gz", color=bcolors.OKGREEN)
    return group_rows, run_id, run_id


# ---------------------------------------------------------------------
# Cross-GPU study
# ---------------------------------------------------------------------

def exportGpu(out_dir, stats):
    """
        Merges the records of every machine or session into their cell. A trial
        found twice in the same cell must have the same final weights and
        accuracy; it is kept once. Host names and machine tags are collected so
        that the final check can make sure none of them reaches the output.
    """
    cell_of = {(cap, torch): cell for cell, cap, torch in _GPU_CELLS}
    records = {cell: {} for cell, _, _ in _GPU_CELLS}
    meta, checksums = {}, set()
    folders = sorted(d for d in glob.glob(_GPU_GLOB)
                     if os.path.isdir(d) and not os.path.basename(d).startswith(_GPU_CALIBRATION))
    if not folders:
        raise SystemExit("cross-GPU study: no output folders found")
    for folder in folders:
        for path in sorted(glob.glob(os.path.join(folder, "gpupool_training_metrics_*.yaml"))):
            for key, rec in (_loadYaml(path) or {}).items():
                cell = cell_of.get((rec["capability"], rec["torch"]))
                if cell is None:
                    raise SystemExit(f"cross-GPU study: record outside the cells "
                                     f"({rec['capability']}, {rec['torch']})")
                if rec["model"] not in _GPU_MODELS or key != f"{rec['model']}__{rec['seed']}":
                    raise SystemExit("cross-GPU study: record key does not match its model and seed")
                if any(name not in rec for name, _ in _GPU_CURVES):
                    raise SystemExit("cross-GPU study: record without its curves")
                checksums.add(rec["pool_checksum"])
                stats["gpu_private_terms"].update(
                    str(rec[k]) for k in ("hostname", "tag") if rec.get(k))
                info = (rec["gpu"], rec["sm_count"])
                if meta.setdefault(cell, info) != info:
                    raise SystemExit(f"cross-GPU study: {cell} mixes GPU models")
                old = records[cell].get(key)
                if old is not None:
                    if (old["final_weights_sha256"], old["accuracy"]) != \
                            (rec["final_weights_sha256"], rec["accuracy"]):
                        raise SystemExit(f"cross-GPU study: two copies of a trial in {cell} differ")
                    stats["gpu_duplicates"] += 1
                    continue
                records[cell][key] = rec
    if len(checksums) != 1:
        raise SystemExit("cross-GPU study: records from different seed pools")
    stats["gpu_pool_checksum"] = checksums.pop()
    stats["gpu_settings"] = _moduleConstants(
        os.path.join(_REPO_ROOT, "scripts", "gpu_check", "run_study.py"),
        ["LEARNING_RATE", "PATIENCE", "NUM_EPOCHS"])
    # The README states that a trial starts from the same weights in every cell
    init = {}
    for cell_records in records.values():
        for key, rec in cell_records.items():
            init.setdefault(key, set()).add(rec["init_weights_sha256"])
    if any(len(h) != 1 for h in init.values()):
        raise SystemExit("cross-GPU study: a trial starts from different weights in two cells")

    order = {m: i for i, m in enumerate(_GPU_MODELS)}
    cell_rows, trial_rows = [], []
    curves = gzip.open(os.path.join(out_dir, "gpu_curves.jsonl.gz"), "wt")
    for cell, cap, torch in _GPU_CELLS:
        recs = sorted(records[cell].values(), key=lambda r: (order[r["model"]], r["seed_index"]))
        if not recs:
            raise SystemExit(f"cross-GPU study: {cell} has no trials")
        for r in recs:
            trial_id = len(trial_rows) + 1
            trial_rows.append({
                "trial_id": trial_id, "cell": cell, "model": r["model"],
                "seed_index": r["seed_index"], "seed": r["seed"], "accuracy": r["accuracy"],
                "test_loss_final": r["test_loss_final"], "best_epoch": r["best_epoch"],
                "total_epochs": r["total_epochs"], "train_duration_s": r["train_duration"],
                "init_weights_sha256": r["init_weights_sha256"],
                "final_weights_sha256": r["final_weights_sha256"],
            })
            stats["gpu_counts"][(cell, r["model"])] = stats["gpu_counts"].get((cell, r["model"]), 0) + 1
            curves.write(json.dumps({"trial_id": trial_id, **{new: r[name] for name, new in _GPU_CURVES}},
                                    separators=(",", ":")) + "\n")
        gpu, sm_count = meta[cell]
        cell_rows.append({"cell": cell, "gpu": gpu, "capability": cap, "sm_count": sm_count,
                          "torch": torch, "n_trials": len(recs)})
        log(f"  {cell} {gpu:28s} {cap:7s} torch {torch:12s} trials {len(recs):4d}")
    curves.close()

    with open(os.path.join(out_dir, "gpu_cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GPU_CELL_FIELDS)
        w.writeheader()
        w.writerows(cell_rows)
    with open(os.path.join(out_dir, "gpu_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GPU_TRIAL_FIELDS)
        w.writeheader()
        w.writerows(trial_rows)
    log(f"  -> gpu_cells.csv ({len(cell_rows)} cells), gpu_trials.csv ({len(trial_rows)} trials), "
        f"gpu_curves.jsonl.gz; repeated trials merged: {stats['gpu_duplicates']}",
        color=bcolors.OKGREEN)
    return cell_rows, len(trial_rows)


# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------

_README = """# Per-trial results: performance variability of neural network training on MNIST and YOLOv8 object detection

This deposit holds the per-trial results behind the article:

> {article}

Each MNIST training trial run for the article is one row of
`mnist_trials.csv`, and each YOLOv8 training run one row of
`yolo_runs.csv`. The trials of the cross-GPU study are the rows of
`gpu_trials.csv`. The other files hold the per-epoch learning curves, the
training settings of the YOLOv8 groups and the cells of the cross-GPU study. The code that trains the MNIST
models and recomputes the tables and figures of the article is archived
separately: {code}. The YOLOv8 runs were trained with the companion
repository https://github.com/enheragu/yolo_test_utils.

## Files

| File | Content | Records |
|---|---|---|
| `mnist_trials.csv` | One row per MNIST training trial | {n_mnist} |
| `mnist_curves.jsonl.gz` | Per-epoch test accuracy of the MNIST trials that logged it | {n_mnist_curves} |
| `yolo_groups.csv` | YOLOv8 experiment groups and their training settings | {n_groups} |
| `yolo_runs.csv` | One row per YOLOv8 training run | {n_runs} |
| `yolo_curves.jsonl.gz` | Per-epoch losses and validation metrics of the YOLOv8 runs | {n_yolo_curves} |
| `gpu_cells.csv` | Cells (GPU and software stack) of the cross-GPU study | {n_gpu_cells} |
| `gpu_trials.csv` | One row per training trial of the cross-GPU study | {n_gpu} |
| `gpu_curves.jsonl.gz` | Per-epoch curves of the cross-GPU trials | {n_gpu} |

## MNIST experiments

The `experiment` column of `mnist_trials.csv` takes these values. All
models are trained with Adam and early stopping on the test accuracy.

| `experiment` | Design | Training settings |
|---|---|---|
| `main` | Eight architectures, {n_main} trials each | Batch size {batch_main}, learning rate {lr_main}, patience {pat_main} epochs, at most {ep_main} epochs, shuffled data |
| `batch_size` | `CNN_14L` with batch sizes {batch_sizes}, {n_batch} trials each (batch size {batch_main} is the `CNN_14L` row of `main`) | As `main` |
| `ablation` | `CNN_14L`, batch size {{{abl_batches}}} x learning rate {{{abl_lrs}}}, {n_iterations} iterations ({n_complete} with all nine trials); the trials of an iteration share the seed | Patience {pat_abl} epochs, at most {ep_abl} epochs, data order fixed (no shuffling) |

Trials per condition, and how many of them logged their seed and their
per-epoch curve:

{count_table}

## Fields of `mnist_trials.csv`

An empty cell means that the field does not apply to the row or was not
logged for it.

| Field | Meaning |
|---|---|
| `trial_id` | Integer identifier, also the key of the curve file |
| `experiment` | See above |
| `model` | Architecture, as named in the code and in the article |
| `condition` | Model or ablation cell as labelled in the article figures, e.g. `CNN_14L_B10_L0.001` |
| `batch_size`, `learning_rate` | Training hyperparameters of the trial |
| `iteration` | `ablation` only: iteration index; the trials with the same `iteration` share `seed`. The per-seed comparisons of the article use the iterations with all nine trials |
| `seed` | Random 32-bit seed of the trial, which drives the weight initialization (and the data order in `main` and `batch_size`); empty where it was not logged. In `ablation` it is the seed of the iteration |
| `accuracy` | Test accuracy (fraction) of the selected epoch |
| `best_epoch` | 0-based epoch with the highest test accuracy, whose weights are kept |
| `total_epochs` | `best_epoch` plus the patience, as logged by the trainer; the curves hold `total_epochs` + 1 values (epochs 0 to `total_epochs`) |
| `train_duration_s` | Wall-clock training time in seconds; depends on the load of the machine |

`mnist_curves.jsonl.gz` holds one JSON object per line, for every trial
that logged its curve: `trial_id` and `test_accuracy`, the test accuracy
after each epoch. Its maximum equals `accuracy`.

## YOLOv8 experiments

`yolo_groups.csv` lists the groups of YOLOv8 runs used in the article, with
the dataset, the input images (the static early-fusion methods VTHS, HSVT
and VT are those of Heredia-Aguado et al., Remote Sensing 17(6), 1060,
2025, https://doi.org/10.3390/rs17061060), the histogram equalization
(CLAHE) applied to the visible and thermal images, and the training
settings read from the arguments of the runs (when a setting differs
between runs, each value is given with its number of runs). `group` is the
name used in the analysis code and in the sampling figures of the article,
`label` the one used in the distribution and survival figures.

{group_table}

`yolo_runs.csv` holds one row per run used in the article: runs that
finished training with a final evaluation, each with a different seed
within its group. Their final evaluation feeds the distribution, sampling
and ablation figures, and their per-epoch curve the survival plots.

## Fields of `yolo_runs.csv`

| Field | Meaning |
|---|---|
| `run_id` | Integer identifier, also the key of the curve file |
| `group` | See `yolo_groups.csv` |
| `seed` | Seed passed to the YOLOv8 trainer, read from the arguments of the run |
| `cache` | Image cache of the trainer, read from the arguments of the run |
| `best_epoch`, `last_epoch` | 0-based epoch of the selected weights and of the last epoch run |
| `train_duration_h` | Wall-clock training time in hours |
| `mAP50`, `mAP50_95` | Final evaluation, all classes |
| `P_person`, `R_person`, `mAP50_person`, `mAP50_95_person` | Final evaluation, class `person` |

The final-evaluation metrics are those stored by the training code at the
end of each run. The metrics are fractions, not percentages.

## Seeds and repeatability of the YOLOv8 runs

The runs were trained with `deterministic` enabled and the images cached
in memory (`cache` = `ram`). With this cache, a rerun with the same seed
does not reproduce a run exactly: the order in which the cached images are
loaded changes the images drawn for the mosaic augmentation.

## Curve file of the YOLOv8 runs

`yolo_curves.jsonl.gz` holds one JSON object per line: `run_id` and, for
each epoch, `epoch`, the training losses (`train_box_loss`,
`train_cls_loss`, `train_dfl_loss`), the validation metrics (`precision`,
`recall`, `mAP50`, `mAP50_95`), the validation losses (`val_box_loss`,
`val_cls_loss`, `val_dfl_loss`) and the learning rates of the three
parameter groups (`lr_pg0`, `lr_pg1`, `lr_pg2`), as logged by YOLOv8.

## Cross-GPU study

`gpu_trials.csv` holds the trials of a complementary study: six MNIST
conditions re-trained with a common pool of seeds on several GPU
architectures and software stacks. The pool is generated from a fixed master
seed by `scripts/gpu_check/make_seed_pool.py` (checksum `{gpu_checksum}`),
so a given `model` and `seed` is the same trial in every cell. The trials use
learning rate {gpu_lr}, patience {gpu_pat} epochs, at most {gpu_ep} epochs and
the batch size of each model. The weights are initialized on the CPU before
the model is moved to the GPU, so `init_weights_sha256` is the same in every
cell for a given trial, and training runs with deterministic algorithms,
TF32 disabled and cuDNN benchmarking off. A cell is one GPU architecture with
one software stack (`gpu_cells.csv`). Trials per cell and model:

{gpu_count_table}

A cell may have been trained on more than one machine or session of the
same GPU model. Trials found more than once in a cell had the same final
weights and accuracy in every copy ({gpu_duplicates} repeated copies) and are
kept once.

## Fields of `gpu_trials.csv`

| Field | Meaning |
|---|---|
| `trial_id` | Integer identifier, also the key of the curve file |
| `cell` | See `gpu_cells.csv` (`gpu`, compute `capability`, `sm_count` and `torch` build of each cell) |
| `model` | Architecture, as named in `mnist_trials.csv` |
| `seed_index`, `seed` | Position of the seed in the pool of the model, and the seed |
| `accuracy`, `test_loss_final` | Test accuracy (fraction) and test loss of the selected epoch |
| `best_epoch`, `total_epochs` | As in `mnist_trials.csv` |
| `train_duration_s` | Wall-clock training time in seconds; depends on the load of the machine |
| `init_weights_sha256`, `final_weights_sha256` | First 16 hexadecimal digits of the SHA-256 of the initial and final weights; equal values mean bit-identical weights |

`gpu_curves.jsonl.gz` holds one JSON object per line: `trial_id` and, for
each epoch, `test_accuracy`, `train_accuracy`, `train_loss` and `test_loss`.

## Reproducing

The MNIST experiments are produced by `00_train_models.py` (`main`,
`batch_size`) and `01_train_ablation_test.py` (`ablation`) of the code
repository, and analysed by `02_compute_analysis.py`,
`03_plot_distribution.py` and `05_check_ablation_test.py`. The YOLOv8 runs
are analysed by `check_yolo_results.py`. The cross-GPU study is produced by
`scripts/gpu_check/run_study.py` and analysed by
`scripts/gpu_check/analyze_cells.py`.

## Licence and attribution

This deposit is released under the Creative Commons Attribution 4.0
International licence (CC BY 4.0). Please cite the article when using these
files, as well as the datasets: MNIST (LeCun and Cortes, 1998), KAIST
Multispectral Pedestrian (Hwang et al., CVPR 2015) and LLVIP (Jia et al.,
ICCV Workshops 2021).
"""


def writeReadme(out_dir, main_cfg, abl_cfg, batch, stats, groups, n, gpu_cells, article, code):
    clines = ["| `cell` | GPU | Capability | `torch` | " + " | ".join(f"`{m}`" for m in _GPU_MODELS) + " |",
              "|---|---|---|---|" + "---|" * len(_GPU_MODELS)]
    for c in gpu_cells:
        clines.append(f"| `{c['cell']}` | {c['gpu']} | `{c['capability']}` | `{c['torch']}` | " +
                      " | ".join(str(stats["gpu_counts"].get((c["cell"], m), 0)) for m in _GPU_MODELS) + " |")
    lines = ["| `experiment` | `condition` | Trials | Seed logged | Curve logged |",
             "|---|---|---|---|---|"]
    for (experiment, condition), (n_all, n_seed, n_curve) in stats["mnist_counts"].items():
        lines.append(f"| `{experiment}` | `{condition}` | {n_all} | {n_seed} | {n_curve} |")
    glines = ["| `group` | Dataset | Input | Runs |",
              "|---|---|---|---|"]
    for g in groups:
        glines.append(f"| `{g['group']}` | {g['dataset']} | {g['input']} | {g['n_runs']} |")
    n_main = {stats["mnist_counts"][("main", m)][0] for m in _MAIN_MODELS}
    n_batch = {stats["mnist_counts"][("batch_size", m)][0] for m in _BATCH_MODELS}
    text = _README.format(
        article=article, code=code,
        n_mnist=n["mnist"], n_mnist_curves=n["mnist_curves"], n_groups=len(groups),
        n_runs=n["yolo"], n_yolo_curves=n["yolo_curves"],
        n_main="/".join(str(v) for v in sorted(n_main)),
        n_batch="/".join(str(v) for v in sorted(n_batch)),
        batch_main=batch["CNN_14L"], lr_main=main_cfg["learning_rate"],
        pat_main=main_cfg["patience"], ep_main=main_cfg["num_epochs"],
        batch_sizes=", ".join(str(batch[m]) for m in _BATCH_MODELS),
        abl_batches=", ".join(str(b) for b in sorted(abl_cfg["batch_conditions"])),
        abl_lrs=", ".join(str(lr) for lr in sorted(abl_cfg["learning_rates"])),
        n_iterations=stats["ablation_iterations"], n_complete=stats["ablation_complete"],
        pat_abl=abl_cfg["patience"],
        ep_abl=abl_cfg["num_epochs"],
        count_table="\n".join(lines), group_table="\n".join(glines),
        n_gpu=n["gpu"], n_gpu_cells=len(gpu_cells), gpu_count_table="\n".join(clines),
        gpu_checksum=stats["gpu_pool_checksum"], gpu_duplicates=stats["gpu_duplicates"],
        gpu_lr=stats["gpu_settings"]["LEARNING_RATE"], gpu_pat=stats["gpu_settings"]["PATIENCE"],
        gpu_ep=stats["gpu_settings"]["NUM_EPOCHS"])
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)
    log("  -> README.md", color=bcolors.OKGREEN)


# ---------------------------------------------------------------------
# Final checks
# ---------------------------------------------------------------------

def checkOutput(out_dir, private_terms=()):
    """
        The curve files must contain only ids and lists of numbers; every
        other file is scanned as text for anything that identifies a
        machine, a person, a local path, a run folder or a training time,
        including the host names and machine tags of the cross-GPU records
        (`private_terms`). Any hit aborts the export.
    """
    for name, key in (("mnist_curves.jsonl.gz", "trial_id"),
                      ("yolo_curves.jsonl.gz", "run_id"),
                      ("gpu_curves.jsonl.gz", "trial_id")):
        with gzip.open(os.path.join(out_dir, name), "rt") as f:
            for line in f:
                rec = json.loads(line)
                if not isinstance(rec.pop(key), int) or any(
                        not all(isinstance(v, (int, float)) for v in vals)
                        for vals in rec.values()):
                    raise SystemExit(f"{name}: non-numeric content")

    terms = {t for t in (getpass.getuser(), socket.gethostname(), *private_terms) if t}
    patterns = [re.escape(t) if len(t) >= 6 else rf"\b{re.escape(t)}\b" for t in terms]
    patterns += [r"/home/", r"~/", r"[\w.+-]+@[\w-]+\.[\w.-]+",
                 r"\b\d{1,3}(?:\.\d{1,3}){3}\b", r"\d{4}-\d{2}-\d{2}[_T ]\d{2}",
                 r"\d{8}_\d{6}", r"_4090", r"dataset_", r"tmp_cfg", r"(?i:arvc)"]
    forbidden = re.compile("|".join(patterns))
    placeholders = []
    for dirpath, _, files in os.walk(out_dir):
        for fn in files:
            if fn.endswith(".gz"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    m = forbidden.search(line)
                    if m:
                        raise SystemExit(f"forbidden content in {os.path.relpath(path, out_dir)}"
                                         f":{i}: {m.group(0)!r}")
                    if "TODO" in line:
                        placeholders.append(f"{os.path.relpath(path, out_dir)}:{i}")
    log("  output check passed", color=bcolors.OKGREEN)
    for t in placeholders:
        log(f"  [WARN] placeholder to fill before upload: {t}", color=bcolors.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "zenodo_export"))
    parser.add_argument("--article", default=ARTICLE_CITATION,
                        help="citation of the article (default: the manuscript as submitted)")
    parser.add_argument("--code-doi", default=None,
                        help="DOI of the archived code release, e.g. 10.5281/zenodo.NNNNNNN")
    args = parser.parse_args()
    code = (f"{CODE_URL} (archived release: https://doi.org/{args.code_doi})" if args.code_doi
            else f"{CODE_URL} [TODO: archived release DOI]")
    out_dir = os.path.abspath(args.out)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    stats = {"mnist_counts": {}, "mnist_curve_mismatch": 0, "yolo_epoch_mismatch": 0,
             "gpu_counts": {}, "gpu_duplicates": 0, "gpu_private_terms": set()}
    main_cfg, abl_cfg, batch = readSettings()
    log("Exporting MNIST trials")
    n_mnist, n_mnist_curves = exportMnist(out_dir, main_cfg, batch, stats)
    log(f"  curves whose length or maximum disagree with the logged fields: "
        f"{stats['mnist_curve_mismatch']}")
    log("Exporting YOLOv8 runs")
    groups, n_yolo, n_yolo_curves = exportYolo(out_dir, stats)
    log(f"  runs whose best or last epoch disagree with their curve: "
        f"{stats['yolo_epoch_mismatch']}")
    log("Exporting the cross-GPU study")
    gpu_cells, n_gpu = exportGpu(out_dir, stats)
    writeReadme(out_dir, main_cfg, abl_cfg, batch, stats, groups,
                {"mnist": n_mnist, "mnist_curves": n_mnist_curves,
                 "yolo": n_yolo, "yolo_curves": n_yolo_curves, "gpu": n_gpu}, gpu_cells,
                args.article, code)
    log("Checking the output")
    checkOutput(out_dir, stats["gpu_private_terms"])
    log(f"Done: {out_dir}", color=bcolors.OKGREEN)


if __name__ == "__main__":
    main()
