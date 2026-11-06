#!/usr/bin/env python3
# encoding: utf-8
"""
Checks whether initializing on CPU makes the initial weights match across
GPUs of different architectures.

Problem detected: the models build their layers with .to(self.device) in the
constructor, so _initialize_weights() operates on tensors ALREADY on the GPU
and consumes the CUDA RNG. That RNG distributes the numbers across threads
according to the launch configuration, which depends on the number of SMs of
the device (4090: 128, 5080: 84). Result: same seed, different initial weights
as soon as the tensor is large.

Without fixing it, comparing "same seed on another GPU" would not be "same
starting point with different arithmetic", but simply a different starting
point -- which is the seed variance that the original study already measured.

Proposed fix: build the model with CUDA hidden (so the constructor picks CPU
and the initialization consumes the CPU RNG, which is hardware independent)
and move the model to the GPU afterwards.

Usage:  python init_on_cpu_test.py --tag <name>
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))

import torch

from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L
from utils.set_seed import set_seed

SEED = 20260729
MODELS = [SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L]
OUT_DIR = os.path.join(HERE, "probe_results")


def hash_state(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build(ModelClass, on_cpu, scratch):
    """Builds the model; if on_cpu, hides CUDA so that the init uses the CPU RNG."""
    set_seed(SEED)
    if on_cpu:
        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            m = ModelClass(input_size=784, num_classes=10, learning_rate=0.001,
                           patience=10, seed=SEED, output_path=scratch)
        finally:
            torch.cuda.is_available = orig
        h = hash_state(m)          # hash BEFORE moving: values generated on CPU
        m = m.to("cuda")
        assert h == hash_state(m), "moving to the GPU altered the weights (it should not)"
        return h
    else:
        m = ModelClass(input_size=784, num_classes=10, learning_rate=0.001,
                       patience=10, seed=SEED, output_path=scratch)
        return hash_state(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. This check compares the initial weights built on the GPU and on the CPU.")

    import tempfile
    p = torch.cuda.get_device_properties(0)
    c = torch.cuda.get_device_capability(0)
    print(f"== {args.tag} :: {p.name}  sm_{c[0]}{c[1]}  {p.multi_processor_count} SMs ==")
    print(f"   torch {torch.__version__}\n")
    print(f"   {'model':24} {'init on GPU (current)':22} {'init on CPU (proposed)':22}")

    res = {"tag": args.tag, "gpu": p.name, "sm_count": p.multi_processor_count,
           "capability": f"sm_{c[0]}{c[1]}", "torch": torch.__version__, "models": {}}

    with tempfile.TemporaryDirectory() as scratch:
        for M in MODELS:
            gpu_h = build(M, on_cpu=False, scratch=scratch)
            cpu_h = build(M, on_cpu=True, scratch=scratch)
            res["models"][M.__name__] = {"init_on_gpu": gpu_h, "init_on_cpu": cpu_h}
            print(f"   {M.__name__:24} {gpu_h:22} {cpu_h:22}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"initcpu_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n   -> {path}")


if __name__ == "__main__":
    main()
