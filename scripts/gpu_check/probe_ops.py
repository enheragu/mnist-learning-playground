#!/usr/bin/env python3
# encoding: utf-8
"""
Operator probe for the cross-GPU variance study.

Runs each suspect operator IN ISOLATION on inputs and weights generated on CPU
with a fixed seed (therefore bit-for-bit identical on any machine) and
publishes a SHA-256 hash of each output and of each gradient.

Comparing the hashes of two GPUs answers, without training anything and in
seconds:
  - do the convolutions (cuDNN) or the GEMMs (cuBLAS) diverge?
  - is BatchNorm a source of divergence on its own?  <-- confounder to isolate
  - does the backward diverge before the forward?

Usage:
    python probe_ops.py --tag <name>            # torch defaults
    python probe_ops.py --tag <name> --no-tf32  # TF32 disabled (comparable with Pascal/Turing)

Output: JSON in scripts/gpu_check/probe_results/probe_<tag>.json
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import torch
import torch.nn as nn

BATCH = 64
INPUT_SEED = 20260729  # seed of the input/weight pool; DO NOT change between machines

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "probe_results")


def sha(t):
    """Stable hash of the binary content of a tensor."""
    a = t.detach().to(torch.float32).contiguous().cpu().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def cpu_tensor(shape, gen):
    """Deterministic tensor ALWAYS created on CPU (the CUDA RNG is not comparable)."""
    return torch.randn(*shape, generator=gen)


def driver_version():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def environment(device):
    props = torch.cuda.get_device_properties(device)
    cap = torch.cuda.get_device_capability(device)
    return {
        "gpu_name": props.name,
        "compute_capability": f"sm_{cap[0]}{cap[1]}",
        "sm_count": props.multi_processor_count,
        "vram_MiB": props.total_memory // (1024 ** 2),
        "driver": driver_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "flags": {
            "cudnn.enabled": torch.backends.cudnn.enabled,
            "cudnn.deterministic": torch.backends.cudnn.deterministic,
            "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<unset>"),
        },
    }


# (name, module constructor, input shape, axis it covers)
OPS = [
    ("linear_784x10",       lambda: nn.Linear(784, 10),                 (BATCH, 784),        "minimal GEMM"),
    ("linear_784x800",      lambda: nn.Linear(784, 800),                (BATCH, 784),        "medium-width GEMM"),
    ("linear_784x2500",     lambda: nn.Linear(784, 2500),               (BATCH, 784),        "very wide GEMM"),
    ("linear_2500x2000",    lambda: nn.Linear(2500, 2000),              (BATCH, 2500),       "wide x wide GEMM"),
    ("conv3x3_1to64_28",    lambda: nn.Conv2d(1, 64, 3, padding=1),     (BATCH, 1, 28, 28),  "conv 3x3 first layer"),
    ("conv3x3_128to128_14", lambda: nn.Conv2d(128, 128, 3, padding=1),  (BATCH, 128, 14, 14), "conv 3x3 (Winograd candidate)"),
    ("conv5x5_64to128_14",  lambda: nn.Conv2d(64, 128, 5, padding=1),   (BATCH, 64, 14, 14), "conv 5x5 (CNN_3L/5L)"),
    ("conv1x1_512to2048_7", lambda: nn.Conv2d(512, 2048, 1),            (BATCH, 512, 7, 7),  "conv 1x1 (= GEMM)"),
    ("bn2d_64_28",          lambda: nn.BatchNorm2d(64),                 (BATCH, 64, 28, 28), "BatchNorm 2D  <-- confounder"),
    ("bn2d_128_14",         lambda: nn.BatchNorm2d(128),                (BATCH, 128, 14, 14), "BatchNorm 2D  <-- confounder"),
    ("bn1d_800",            lambda: nn.BatchNorm1d(800),                (BATCH, 800),        "BatchNorm 1D  <-- confounder"),
    ("maxpool2x2",          lambda: nn.MaxPool2d(2, 2),                 (BATCH, 64, 28, 28), "no arithmetic reduction (control)"),
    ("relu",                lambda: nn.ReLU(),                          (BATCH, 2500),       "elementwise (control)"),
]


def run_op(name, build, shape, device):
    """Runs fwd+bwd of an operator with deterministic weights and input."""
    gen = torch.Generator().manual_seed(INPUT_SEED)

    mod = build()
    # Deterministic weights created on CPU: identical on any machine.
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(cpu_tensor(tuple(p.shape), gen) * 0.05)
    mod = mod.to(device).train()

    x = cpu_tensor(shape, gen).to(device).requires_grad_(True)

    out = mod(x)
    g = cpu_tensor(tuple(out.shape), torch.Generator().manual_seed(INPUT_SEED + 1)).to(device)
    out.backward(g)

    res = {
        "forward": sha(out),
        "out_shape": list(out.shape),
        "grad_input": sha(x.grad),
        "out_sum": float(out.detach().double().sum().item()),
        "out_absmax": float(out.detach().abs().max().item()),
    }
    for pname, p in mod.named_parameters():
        if p.grad is not None:
            res[f"grad_{pname}"] = sha(p.grad)
    return res


def run_loss(device):
    """CrossEntropyLoss reduction on its own (sum over the batch)."""
    gen = torch.Generator().manual_seed(INPUT_SEED)
    logits = cpu_tensor((BATCH, 10), gen).to(device).requires_grad_(True)
    labels = torch.randint(0, 10, (BATCH,), generator=gen).to(device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    return {
        "forward": sha(loss.reshape(1)),
        "loss_value_hex": float(loss.item()).hex(),
        "grad_input": sha(logits.grad),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="Identifier of this machine/configuration")
    ap.add_argument("--no-tf32", action="store_true",
                    help="Disables TF32 in cuDNN and matmul (required to compare with Pascal/Turing)")
    ap.add_argument("--cudnn-off", action="store_true",
                    help="Disables cuDNN completely (falls back to native ATen convolutions)")
    ap.add_argument("--benchmark", action="store_true",
                    help="Enables cuDNN autotuning (not deterministic across runs)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no GPU available. This probe only makes sense on CUDA.")

    # Base configuration: the pipeline one (set_seed) plus the flags that were missing.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = not args.cudnn_off
    if args.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    env = environment(device)

    print(f"== Operator probe :: tag={args.tag} ==")
    print(f"   {env['gpu_name']}  {env['compute_capability']}  {env['sm_count']} SMs  "
          f"driver {env['driver']}")
    print(f"   torch {env['torch']} (cuda {env['cuda_build']}, cudnn {env['cudnn']})")
    print(f"   TF32 cudnn={env['flags']['cudnn.allow_tf32']} matmul={env['flags']['matmul.allow_tf32']}"
          f" | cudnn.enabled={env['flags']['cudnn.enabled']} benchmark={env['flags']['cudnn.benchmark']}")
    print()

    results = {}
    for name, build, shape, axis in OPS:
        try:
            results[name] = run_op(name, build, shape, device)
            results[name]["axis"] = axis
            print(f"  {name:22} {results[name]['forward']}  fwd | "
                  f"{results[name]['grad_input']}  dX   ({axis})")
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {name:22} ERROR: {e}")

    results["cross_entropy"] = run_loss(device)
    results["cross_entropy"]["axis"] = "reduction over the batch"
    print(f"  {'cross_entropy':22} {results['cross_entropy']['forward']}  fwd | "
          f"{results['cross_entropy']['grad_input']}  dX")

    payload = {"tag": args.tag, "input_seed": INPUT_SEED, "batch": BATCH,
               "environment": env, "ops": results}

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"probe_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
