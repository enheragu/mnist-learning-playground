#!/usr/bin/env python3
# encoding: utf-8
"""
Checks kernel for the cross-GPU variance study.

Runs on Kaggle exactly the same tests that have already been passed on the lab
machines, with the SAME code and the SAME dataset (they arrive through the
Kaggle Dataset `<kaggle-user>/<dataset>`, not by download):

  1. Operator probe, `defaults` and `notf32` arms.
  2. Initialization-on-CPU test  <-- the most informative one here: the T4 has
     40 SMs, versus 84 on the 5080 and 128 on the 4090. If the divergence of
     initial weights is caused by the number of SMs, the T4 should give a THIRD
     different set with the current init, and match the other two with init on
     CPU.
  3. Determinism test, clean arm.

Before anything else it verifies the code and dataset fingerprints against
FINGERPRINTS.txt and aborts if they do not match: it would be a sign of running
something different from the rest of the study, and the results would not be
comparable.

Outputs in /kaggle/working/results/.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

TORCH_VERSION = "2.9.1"
CUDA_TAG = "cu128"
WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
RESULTS = f"{WORK}/results"


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


def run(cmd, **kw):
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, text=True, **kw)


def fingerprint(paths_cmd, cwd):
    r = subprocess.run(paths_cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


# ------------------------------------------------------------------ 1. code
section("1. Locating the code and the dataset")

if os.path.exists(REPO):
    shutil.rmtree(REPO)

# Kaggle respects neither the depth nor the name of the directory: it mounts at
# /kaggle/input/datasets/<user>/<slug>/ and also EXTRACTS the zip flattening its
# contents, so no directory called `repo` is left.
#
# That is why the root is identified by what it CONTAINS (src/ +
# requirements.lock), not by its name or by where it is.
candidates, zips = [], []
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isdir(os.path.join(root, "src")) and "requirements.lock" in files:
        candidates.append(root)
        dirs[:] = []          # do not descend further into the tree found
        continue
    if "repo.zip" in files:
        zips.append(os.path.join(root, "repo.zip"))

if candidates:
    SRC = candidates[0]
    print(f"  source: {SRC} (already extracted)")
    # /kaggle/input is read-only; the scripts write results.
    shutil.copytree(SRC, REPO)
elif zips:
    # Depending on how Kaggle processes the upload, the .zip may arrive unextracted.
    import zipfile
    print(f"  source: {zips[0]} (zip not extracted)")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
    if not os.path.isdir(REPO):
        inner = [d for d in glob.glob(f"{WORK}/*") if os.path.isdir(os.path.join(d, "src"))]
        if not inner:
            sys.exit(f"ERROR: the zip does not contain a tree with src/. Contents: {os.listdir(WORK)}")
        os.rename(inner[0], REPO)
else:
    print("  Full tree of /kaggle/input (up to 4 levels):")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth > 4:
            dirs[:] = []
            continue
        print(f"    {'  ' * depth}{root}")
        for f in files[:6]:
            print(f"    {'  ' * (depth + 1)}{f}")
    sys.exit("ERROR: cannot find repo/ or repo.zip under /kaggle/input. "
             "Is the dataset attached to the kernel?")
os.makedirs(RESULTS, exist_ok=True)
print(f"  copied to: {REPO}")

# ------------------------------------------------------------- 2. fingerprints
section("2. Fingerprint verification")

TFP_CMD = ("{ find src/models -type f -name '*.py' 2>/dev/null; "
           "ls src/utils/set_seed.py requirements.lock "
           "scripts/gpu_check/run_study.py scripts/gpu_check/preloaded_loader.py 2>/dev/null; } "
           "| LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")
# Over the *-ubyte files, not the .gz: they are what torchvision reads, and the
# .gz files are not sent to Kaggle because they collide when decompressed.
DFP_CMD = ("cd data/MNIST/raw && ls *-ubyte | LC_ALL=C sort | xargs sha256sum "
           "| sha256sum | cut -c1-16")

expected = {}
fp_file = os.path.join(REPO, "FINGERPRINTS.txt")
if os.path.exists(fp_file):
    for line in open(fp_file):
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            expected[k] = v

got_t = fingerprint(TFP_CMD, REPO)
got_d = fingerprint(DFP_CMD, REPO)

print(f"  training fingerprint : {got_t}   (expected: {expected.get('training_fingerprint','?')})")
print(f"  dataset fingerprint  : {got_d}   (expected: {expected.get('dataset_fingerprint','?')})")

bad = []
if expected.get("training_fingerprint") and got_t != expected["training_fingerprint"]:
    bad.append("code")
if expected.get("dataset_fingerprint") and got_d != expected["dataset_fingerprint"]:
    bad.append("dataset")
if bad:
    sys.exit(f"ERROR: {' and '.join(bad)} do not match the reference. "
             "The results would not be comparable with the rest of the study.")
print("  -> they match the control machine")

# --------------------------------------------------------------- 3. environment
section("3. Assigned GPU")

run("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader")

# --------------------------------------------------------------- 4. stack
section(f"4. Installing torch {TORCH_VERSION}+{CUDA_TAG}")

t0 = time.time()
r = run(f"{sys.executable} -m pip install -q torch=={TORCH_VERSION} torchvision "
        f"--index-url https://download.pytorch.org/whl/{CUDA_TAG}")
print(f"  rc={r.returncode}  time={time.time()-t0:.0f} s")
if r.returncode != 0:
    sys.exit("ERROR: could not install the pinned stack")

# The GPU has to be usable: on Kaggle there is no guarantee which one is
# assigned, and the P100 (sm_60) is not supported by cu128.
check = subprocess.run([sys.executable, "-c", r'''
import sys, torch, torch.nn as nn
if not torch.cuda.is_available(): sys.exit("no CUDA")
major, minor = torch.cuda.get_device_capability(0)
sm = f"sm_{major}{minor}"
arch = torch.cuda.get_arch_list()
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  {sm}  {p.multi_processor_count} SMs  torch {torch.__version__}")
try:
    x = torch.randn(256, 256, device="cuda"); float((x @ x).sum())
    nn.Conv2d(8, 16, 3).cuda()(torch.randn(2, 8, 16, 16, device="cuda"))
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"GPU {sm} cannot run: {type(e).__name__}: {e}")
'''], capture_output=True, text=True)
print(check.stdout, end="")
if check.returncode != 0:
    print(check.stderr)
    sys.exit("ERROR: the assigned GPU cannot be used for this stack. Relaunch requesting T4.")

# ------------------------------------------------------------- 5. tests
GPU_CHECK = os.path.join(REPO, "scripts", "gpu_check")
TAG = os.environ.get("EEHA_TAG", "kaggle-T4")
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")

section("5. Operator probe")
for arm, flag in (("defaults", ""), ("notf32", "--no-tf32")):
    run(f"{sys.executable} probe_ops.py --tag {TAG}-{arm} {flag}", cwd=GPU_CHECK, env=env)

section("6. Initialization: GPU (current) vs CPU (proposed)")
run(f"{sys.executable} init_on_cpu_test.py --tag {TAG}", cwd=GPU_CHECK, env=env)

section("7. Determinism, clean arm")
run(f"{sys.executable} determinism_test.py --tag {TAG}-clean "
    f"--no-tf32 --strict-determinism --epochs 3", cwd=GPU_CHECK, env=env)

# ------------------------------------------------------------ 8. collect
section("8. Collecting results")
n = 0
for f in glob.glob(os.path.join(GPU_CHECK, "probe_results", "*.json")):
    shutil.copy(f, RESULTS)
    print(f"  {os.path.basename(f)}")
    n += 1

meta = {"tag": TAG, "training_fingerprint": got_t, "dataset_fingerprint": got_d,
        "torch": TORCH_VERSION + "+" + CUDA_TAG, "n_results": n}
with open(os.path.join(RESULTS, "run_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"\n  {n} files in {RESULTS}")
