#!/usr/bin/env python3
# encoding: utf-8
"""
Validation gate for the preloaded loader.

Trains the same seeds through the ORIGINAL path (torch DataLoader) and through
the NEW one (PreloadedLoader) and compares the hashes of the final weights. If
they are not bit-for-bit identical, the new loader is NOT used: it would be
changing results.

It is not a formality. `00_train_models.py` uses shuffle=True, so the order of
the data depends on the global RNG and any deviation in how it is consumed
shifts it silently.

Usage:
    python validate_preload.py --seeds 5 --epochs 2
    python validate_preload.py --models CNN_14L --seeds 3
"""

import argparse
import hashlib
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
sys.path.insert(0, HERE)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L
from utils.set_seed import set_seed
from preloaded_loader import make_loaders

DATA_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "data"))
MODELS = {m.__name__: m for m in
          (SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_3L, CNN_14L)}


def hash_state(model):
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def run(ModelClass, seed, epochs, scratch, train_ds, test_ds, preloaded, cache):
    set_seed(seed)
    model = ModelClass(input_size=784, num_classes=10, learning_rate=0.001,
                       patience=10_000, seed=seed, output_path=scratch)

    if preloaded:
        # The tensors are cached: preloading in every trial would be absurd,
        # and the runner will not do it either.
        if "loaders" not in cache:
            cache["loaders"] = make_loaders(train_ds, test_ds, model.batch_size, model.device)
        train_loader, test_loader = cache["loaders"]
    else:
        train_loader = DataLoader(dataset=train_ds, batch_size=model.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_ds, batch_size=model.batch_size, shuffle=False)

    t0 = time.time()
    model.train_model(train_loader, test_loader, num_epochs=epochs)
    return hash_state(model), time.time() - t0, list(model.accuracy_each_epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    names = args.models or list(MODELS)
    bad = [n for n in names if n not in MODELS]
    if bad:
        sys.exit(f"Unknown models: {bad}. Available: {list(MODELS)}")

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA GPU available. This check validates the loader that keeps the dataset on the GPU.")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_ds = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    p = torch.cuda.get_device_properties(0)
    print(f"== Preloaded loader validation ==")
    print(f"   {p.name}  torch {torch.__version__}")
    print(f"   {args.seeds} seeds x {args.epochs} epochs per model\n")

    seeds = [1000 + i for i in range(args.seeds)]
    all_ok = True
    speedups = []

    with tempfile.TemporaryDirectory() as scratch:
        for name in names:
            M = MODELS[name]
            cache = {}
            ok = True
            t_old_tot = t_new_tot = 0.0
            for s in seeds:
                h_old, t_old, acc_old = run(M, s, args.epochs, scratch, train_ds, test_ds, False, cache)
                h_new, t_new, acc_new = run(M, s, args.epochs, scratch, train_ds, test_ds, True, cache)
                t_old_tot += t_old
                t_new_tot += t_new
                if h_old != h_new:
                    ok = False
                    print(f"   [FAIL] {name} seed {s}: {h_old} != {h_new}")
                    print(f"           acc original : {acc_old}")
                    print(f"           acc preloaded: {acc_new}")
            sp = t_old_tot / t_new_tot if t_new_tot else 0
            speedups.append(sp)
            mark = "OK  " if ok else "FAIL"
            print(f"   [{mark}] {name:24} {args.seeds}/{args.seeds} identical"
                  f"   {t_old_tot/args.seeds:5.1f}s -> {t_new_tot/args.seeds:5.1f}s  ({sp:.2f}x)")
            all_ok &= ok

    print()
    if all_ok:
        print(f"   VERDICT: bit-for-bit identical. Mean speedup {sum(speedups)/len(speedups):.2f}x")
        print("   -> the preloaded loader can be used.")
    else:
        print("   VERDICT: NOT identical. The preloaded loader CANNOT be used as is.")
        sys.exit(1)


if __name__ == "__main__":
    main()
