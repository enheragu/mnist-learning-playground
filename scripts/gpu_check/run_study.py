#!/usr/bin/env python3
# encoding: utf-8
"""
Runner of the cross-GPU variance study.

Trains the seed pool from `seed_pools.yaml` and leaves the results in
`output_data_gpucheck_<tag>/`, ready to be compared across machines.

DECISIONS THAT COME FROM THINGS THAT WENT WRONG
-----------------------------------------------
1. INITIALIZATION ON CPU. The models call `.to(device)` in the constructor, so
   `_initialize_weights()` consumed the CUDA RNG, whose distribution across
   threads depends on the NUMBER OF SMs. Measured result: 4 of 5 models started
   with different weights on the 4090 (128 SMs) and the 5080 (84 SMs), before
   training. Without this the study would compare different starting points,
   which is seed variance, not hardware variance.

2. INDEXED BY SEED, not by timestamp. The original pipeline used
   `datetime.now()` as the key: with several processes and several machines
   there is a risk of collision and the order means nothing. The timestamp is
   stored as a FIELD.

3. ORDER seed-outside / model-inside. If a session is cut short, ALL machines
   are missing the same trials and the intersection is still a clean prefix of
   the same pool.

4. ABORTS if the GPU does not support the stack. On Kaggle the GPU cannot be
   chosen through the API and a P100 (sm_60) may be assigned, where cu128 does
   not run.

5. DOES NOT WRITE TO output_data/. The 2024 results are frozen; there is an
   explicit assertion, not just the naming convention.

Usage:
    python run_study.py --tag gpu-a
    python run_study.py --tag X --shard 0/4 --deadline-min 660
    python run_study.py --tag X --dry-run
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)

import torch
import yaml

from models import (SimplePerceptron, HiddenLayerPerceptron, DNN_6L,
                    CNN_3L, CNN_14L)
from models.BatchSizeStudy import CNN_14L_B10
from utils.set_seed import set_seed
from preloaded_loader import make_loaders

MODELS = {m.__name__: m for m in
          (SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L, CNN_14L_B10)}

POOL_FILE = os.path.join(REPO, "seed_pools.yaml")
DATA_ROOT = os.path.join(REPO, "data")

# Hyperparameters of the original study; not to be changed.
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10
LEARNING_RATE = 0.001
PATIENCE = 10
NUM_EPOCHS = 500


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def hash_state(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def set_clean_arm():
    """Clean-arm configuration: comparable with Pascal/Turing and deterministic."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def check_gpu():
    """Aborts if the GPU cannot run this stack (the P100 case on Kaggle)."""
    if not torch.cuda.is_available():
        sys.exit("ABORT: no GPU available.")
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    p = torch.cuda.get_device_properties(0)
    try:
        import torch.nn as nn
        x = torch.randn(256, 256, device="cuda")
        float((x @ x).sum())
        nn.Conv2d(8, 16, 3).cuda()(torch.randn(2, 8, 16, 16, device="cuda"))
        torch.cuda.synchronize()
    except Exception as e:
        sys.exit(f"ABORT: GPU {p.name} ({sm}) cannot run this stack: "
                 f"{type(e).__name__}: {e}\n"
                 f"        torch archs: {torch.cuda.get_arch_list()}")
    return {"name": p.name, "capability": sm, "sm_count": p.multi_processor_count,
            "vram_MiB": p.total_memory // 2**20}


def build_model(ModelClass, seed, output_path):
    """
    Builds the model with the initialization on CPU.

    CUDA is hidden during the constructor so that the layers are created on CPU
    and `_initialize_weights()` consumes the CPU RNG (hardware independent),
    instead of the CUDA one (which depends on the number of SMs). The model is
    then moved to the GPU. Reuses each model's own init logic, without
    duplicating it.
    """
    orig = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        m = ModelClass(input_size=INPUT_SIZE, num_classes=NUM_CLASSES,
                       learning_rate=LEARNING_RATE, patience=PATIENCE,
                       seed=seed, output_path=output_path)
    finally:
        torch.cuda.is_available = orig
    init_hash = hash_state(m)
    m = m.to("cuda")
    m.device = torch.device("cuda")
    assert hash_state(m) == init_hash, "moving to the GPU altered the weights"
    return m, init_hash


def load_shard(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def append_result(path, key, record):
    """Incremental write: if the session dies, the work already done is kept."""
    data = load_shard(path)
    data[key] = record
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, path)   # atomic: never leaves the yaml half-written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="identifier of this machine")
    ap.add_argument("--out-base", default=None,
                    help="parent directory of output_data_gpucheck_<tag> (default: the repo)")
    ap.add_argument("--shard", default="0/1", help="k/N: this process runs the seeds with index %%N==k")
    ap.add_argument("--deadline-min", type=float, default=None,
                    help="wall-clock minutes; stops cleanly before using them up (Kaggle sessions)")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of models")
    ap.add_argument("--limit", type=int, default=None,
                    help="runs at most N trials; useful in calibration, so that "
                         "all machines run EXACTLY the same ones and are comparable")
    ap.add_argument("--dry-run", action="store_true", help="shows the plan and exits")
    args = ap.parse_args()

    k, n_shards = (int(v) for v in args.shard.split("/"))
    assert 0 <= k < n_shards, "--shard k/N with 0 <= k < N"

    # ---- pool
    if not os.path.exists(POOL_FILE):
        sys.exit(f"ABORT: {POOL_FILE} does not exist. Generate it with make_seed_pool.py")
    pool_doc = yaml.safe_load(open(POOL_FILE))
    pools = pool_doc["pools"]
    order = [c["model"] for c in pool_doc["conditions"]]
    if args.conditions:
        bad = [c for c in args.conditions if c not in order]
        if bad:
            sys.exit(f"ABORT: unknown conditions {bad}")
        order = [c for c in order if c in args.conditions]

    # ---- output
    base = args.out_base or REPO
    out_dir = os.path.join(base, f"output_data_gpucheck_{args.tag}")
    # The 2024 results are frozen. Explicit assertion, not convention.
    norm = os.path.normpath(out_dir)
    assert f"{os.sep}output_data{os.sep}" not in norm + os.sep and \
           os.path.basename(norm) != "output_data", \
           f"ABORT: the output path would fall inside the frozen output_data/: {norm}"
    os.makedirs(out_dir, exist_ok=True)

    shard_path = os.path.join(out_dir, f"gpupool_training_metrics_p{k}.yaml")
    done = set(load_shard(shard_path))

    # ---- plan: seed outside, model inside
    plan = []
    max_n = max(len(pools[m]) for m in order)
    for i in range(max_n):
        if i % n_shards != k:
            continue
        for m in order:
            if i < len(pools[m]):
                seed = pools[m][i]
                key = f"{m}__{seed}"
                if key not in done:
                    plan.append((i, m, seed, key))

    if args.limit:
        plan = plan[:args.limit]

    log(f"tag={args.tag}  shard={k}/{n_shards}  pool_checksum={pool_doc['checksum']}")
    log(f"output: {shard_path}")
    log(f"pending: {len(plan)}   already done in this shard: {len(done)}")
    if args.dry_run:
        for i, m, seed, _ in plan[:10]:
            log(f"  idx={i:4} {m:24} seed={seed}")
        log(f"  ... ({len(plan)} in total)")
        return

    # ---- environment
    set_clean_arm()
    gpu = check_gpu()
    log(f"GPU: {gpu['name']}  {gpu['capability']}  {gpu['sm_count']} SMs")

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=False)
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=False)

    # One set of tensors per batch_size: the models differ in batch size.
    loader_cache = {}
    t_start = time.time()
    deadline = t_start + args.deadline_min * 60 if args.deadline_min else None

    ckpt_dir = os.path.join(out_dir, "_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    for nth, (i, mname, seed, key) in enumerate(plan, 1):
        if deadline and time.time() > deadline:
            log(f"deadline reached; stopping cleanly after {nth-1}/{len(plan)}")
            break

        ModelClass = MODELS[mname]
        set_seed(seed)
        model, init_hash = build_model(ModelClass, seed, ckpt_dir)

        bs = model.batch_size
        if bs not in loader_cache:
            # Neutral with respect to the RNG: preload_dataset saves and restores the state.
            loader_cache[bs] = make_loaders(train_ds, test_ds, bs, model.device)
        train_loader, test_loader = loader_cache[bs]

        t0 = time.time()
        metrics = model.spinTrainEval(train_loader, test_loader, num_epochs=NUM_EPOCHS)
        data = next(iter(metrics.values()))
        dur = time.time() - t0

        record = {
            "seed": int(seed),
            "seed_index": int(i),
            "model": mname,
            "accuracy": float(data["accuracy"]),
            "test_loss_final": float(model.test_loss_each_epoch[-1]) if model.test_loss_each_epoch else None,
            "best_epoch": int(data["best_epoch"]),
            "total_epochs": int(data["total_epochs"]),
            "train_duration": float(data["train_duration"]),
            "accuracy_plot": [float(v) for v in model.accuracy_each_epoch],
            "train_accuracy_plot": [float(v) for v in model.train_accuracy_each_epoch],
            "train_loss_plot": [float(v) for v in model.train_loss_each_epoch],
            "test_loss_plot": [float(v) for v in model.test_loss_each_epoch],
            "init_weights_sha256": init_hash,
            "final_weights_sha256": hash_state(model),
            # Context: not part of the key, but it is what makes it possible
            # to reconstruct what happened if something does not add up.
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "tag": args.tag,
            "shard": f"{k}/{n_shards}",
            "gpu": gpu["name"],
            "capability": gpu["capability"],
            "sm_count": gpu["sm_count"],
            # torch.__version__ is a TorchVersion (a subclass of str) and
            # yaml.safe_dump cannot represent it: it has to be forced to a plain str.
            "torch": str(torch.__version__),
            "pool_checksum": pool_doc["checksum"],
        }
        append_result(shard_path, key, record)

        # The .pth files are not kept: they take up GBs and the study compares hashes.
        for f in os.listdir(ckpt_dir):
            try:
                os.remove(os.path.join(ckpt_dir, f))
            except OSError:
                pass

        eta = ""
        if deadline:
            eta = f"  ({(deadline - time.time())/60:.0f} min of session left)"
        log(f"{nth}/{len(plan)}  {mname:24} seed={seed:<11} "
            f"acc={record['accuracy']:.4f} ep={record['total_epochs']:<3} "
            f"{dur:5.1f}s  init={init_hash}{eta}")

    log(f"done. total in this shard: {len(load_shard(shard_path))}")


if __name__ == "__main__":
    main()
