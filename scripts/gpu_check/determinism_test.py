#!/usr/bin/env python3
# encoding: utf-8
"""
Intra-machine determinism test for the cross-GPU variance study.

Question it answers: on THIS same GPU, with the SAME seed, do two identical
runs of the real pipeline give bit-for-bit identical results?

It is the entry gate of the whole study. On CPU determinism came for free;
on GPU it does not. If two runs here do not match, the `bit_exact` metric is no
longer usable across machines and the analysis has to be redesigned around
distributional metrics.

Uses the REAL pipeline (BaseModelTrainer.train_model, same optimizer, same
DataLoader with shuffle=True) but with few epochs: converging is not needed to
detect divergence.

Usage:
    python determinism_test.py --tag 4090-defaults
    python determinism_test.py --tag 4090-notf32 --no-tf32 --strict-determinism

Output: JSON in scripts/gpu_check/probe_results/determinism_<tag>.json
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

# Allows importing the real pipeline from src/
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.set_seed import set_seed
from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L

OUT_DIR = os.path.join(HERE, "probe_results")
DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "data"))

SEED = 20260729
EPOCHS = 3          # enough to detect divergence; we are not after convergence
REPEATS = 2
INPUT_SIZE = 28 * 28
NUM_CLASSES = 10

MODELS = [SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L]


def hash_state(model):
    """Hash of the full state_dict, in a stable key order."""
    h = hashlib.sha256()
    for k in sorted(model.state_dict().keys()):
        v = model.state_dict()[k]
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def driver_version():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def environment():
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "gpu_name": props.name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "sm_count": props.multi_processor_count,
        "driver": driver_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "flags": {
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        },
    }


def one_run(ModelClass, train_dataset, test_dataset, scratch, epochs):
    """One full run of the real pipeline with a fixed seed."""
    set_seed(SEED)   # exactly as 00_train_models.py does

    model = ModelClass(input_size=INPUT_SIZE, num_classes=NUM_CLASSES,
                       learning_rate=0.001, patience=10_000,  # no early stopping
                       seed=SEED, output_path=scratch)
    init_hash = hash_state(model)

    # Same loaders as 00_train_models.py (note: shuffle=True for train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)

    t0 = time.time()
    model.train_model(train_loader, test_loader, num_epochs=epochs)
    dur = time.time() - t0

    return {
        "init_weights": init_hash,
        "final_weights": hash_state(model),
        "accuracy_per_epoch": [float(a) for a in model.accuracy_each_epoch],
        "accuracy_hex": [float(a).hex() for a in model.accuracy_each_epoch],
        "duration_s": round(dur, 2),
        "s_per_epoch": round(dur / epochs, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-tf32", action="store_true",
                    help="Disables TF32 in cuDNN and matmul")
    ap.add_argument("--strict-determinism", action="store_true",
                    help="use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of models by name (default: all)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU available.")

    epochs = args.epochs

    if args.strict_determinism:
        # Must be set BEFORE any cuBLAS handle is created.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    selected = MODELS
    if args.models:
        by_name = {m.__name__: m for m in MODELS}
        missing = [n for n in args.models if n not in by_name]
        if missing:
            sys.exit(f"ERROR: unknown models: {missing}. Available: {list(by_name)}")
        selected = [by_name[n] for n in args.models]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    env = environment()
    print(f"== Determinism test :: tag={args.tag} ==")
    print(f"   {env['gpu_name']}  {env['compute_capability']}  {env['sm_count']} SMs  driver {env['driver']}")
    print(f"   torch {env['torch']} (cuda {env['cuda_build']}, cudnn {env['cudnn']})")
    print(f"   TF32 cudnn={env['flags']['cudnn.allow_tf32']} matmul={env['flags']['matmul.allow_tf32']}"
          f" | strict={args.strict_determinism}")
    print(f"   seed={SEED}  epochs={epochs}  repeats={REPEATS}\n")

    results, verdicts = {}, {}
    with tempfile.TemporaryDirectory(prefix="determinism_") as scratch:
        for ModelClass in selected:
            name = ModelClass.__name__
            runs = []
            for rep in range(REPEATS):
                runs.append(one_run(ModelClass, train_dataset, test_dataset, scratch, epochs))
            results[name] = runs

            same_init = len({r["init_weights"] for r in runs}) == 1
            same_final = len({r["final_weights"] for r in runs}) == 1
            same_acc = len({tuple(r["accuracy_hex"]) for r in runs}) == 1
            verdicts[name] = {"init_identical": same_init,
                              "final_weights_identical": same_final,
                              "accuracy_identical": same_acc}

            mark = "OK  " if (same_init and same_final and same_acc) else "FAIL"
            print(f"  [{mark}] {name:24} init={runs[0]['init_weights']} "
                  f"final={'=' if same_final else '!='} acc={'=' if same_acc else '!='} "
                  f"({runs[0]['s_per_epoch']} s/epoch)")
            if not same_final:
                for i, r in enumerate(runs):
                    print(f"          rep{i}: final={r['final_weights']} acc={r['accuracy_per_epoch']}")

    all_ok = all(all(v.values()) for v in verdicts.values())
    print(f"\n  OVERALL VERDICT: {'deterministic' if all_ok else 'NOT deterministic'}")

    payload = {"tag": args.tag, "seed": SEED, "epochs": epochs, "repeats": REPEATS,
               "environment": env, "verdicts": verdicts, "runs": results,
               "all_deterministic": all_ok}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"determinism_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
