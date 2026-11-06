#!/usr/bin/env python3
# encoding: utf-8
"""
Operator probe with BOTH STACKS in the SAME session and on the SAME GPU.

WHAT QUESTION IT ANSWERS
------------------------
In the full trainings, changing the torch version preserves bit-exactness in
dense networks and breaks it as soon as there are convolutions (500/632, with
the split following exactly whether the model has Conv2d or not).

From this we INFER that the cuBLAS GEMMs are stable across versions and that
cuDNN changes its convolution implementation or heuristic. But it is an
inference from full trainings: it has never been measured operator by operator.

This probe measures it directly. `probe_ops.py` covers 4 convolutions and 4
dense GEMMs with inputs and weights seeded from CPU, so the comparison is exact:

    conv1x1_512to2048_7   conv3x3_128to128_14   conv3x3_1to64_28   conv5x5_64to128_14
    linear_784x10   linear_784x800   linear_784x2500   linear_2500x2000

If the pattern is confirmed, the `linear_*` will come out identical across
stacks and the `conv*` different. If it is NOT confirmed, the explanation is a
different one and has to be reconsidered.

WHY IN A SINGLE SESSION
-----------------------
Installing the two stacks one after the other in the same kernel, the
comparison is on the SAME physical GPU and the same session. It even removes
the "another session" variable, which in the rest of the study had to be
verified separately (84/84 and 48/48).

It costs ~15 min of quota. It has to be launched by hand with `GPU T4 x2`: the
P100 does not support cu128 and the goal is precisely to compare the two stacks.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"
RES = f"{WORK}/mechanism"

# (label, torch version, CUDA tag)
STACKS = [
    ("cu128", "2.9.1", "cu128"),   # the one of the main study
    ("cu118", "2.7.1", "cu118"),   # the one of the stack experiment
]
EXPECTED_GPU = "T4"


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)


# --------------------------------------------------------------- 1. code
section("1. Code")
if os.path.exists(REPO):
    shutil.rmtree(REPO)
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if os.path.isdir(os.path.join(root, "src")) and "requirements.lock" in files:
        src = root
        break
if src is None:
    sys.exit("ERROR: cannot find the code under /kaggle/input")
shutil.copytree(src, REPO)
os.makedirs(RES, exist_ok=True)
GPU_CHECK = os.path.join(REPO, "scripts", "gpu_check")
print(f"  {src} -> {REPO}")

# ----------------------------------------------------------- 2. check the GPU
section("2. Assigned GPU")
n = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
                   shell=True, capture_output=True, text=True).stdout.strip()
print(f"  {n}")
if EXPECTED_GPU and EXPECTED_GPU not in n:
    sys.exit(f"ABORT: this probe needs {EXPECTED_GPU} (the P100 does not support cu128).\n"
             f"  If this is the automatic run of the push, this is correct: it stops in seconds.\n"
             f"  To launch it, open the editor and choose GPU {EXPECTED_GPU} x2.")

# ------------------------------------------------------ 3. probe each stack
env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
for label, version, cuda in STACKS:
    section(f"3.{label}  torch {version}+{cuda}")
    t0 = time.time()
    r = subprocess.run(f"{sys.executable} -m pip install -q torch=={version} torchvision "
                       f"--index-url https://download.pytorch.org/whl/{cuda}",
                       shell=True, text=True)
    print(f"  installation rc={r.returncode}  {time.time()-t0:.0f} s")
    if r.returncode != 0:
        sys.exit(f"ERROR: could not install {version}+{cuda}")

    # Clean arm, the same as in the study: no TF32 and deterministic algorithms.
    p = subprocess.run([sys.executable, "probe_ops.py", "--tag", f"t4-{label}",
                        "--no-tf32"], cwd=GPU_CHECK, env=env,
                       capture_output=True, text=True)
    print(p.stdout[-1500:] if p.stdout else "")
    if p.returncode != 0:
        print(p.stderr[-2000:])
        sys.exit(f"ERROR: the probe failed with {version}+{cuda}")
    for f in glob.glob(os.path.join(GPU_CHECK, "probe_results", "*.json")):
        shutil.copy(f, RES)

# ------------------------------------------------------------- 4. compare
section("4. Operator-by-operator comparison")
probe_files = {os.path.basename(f): f for f in glob.glob(f"{RES}/probe_t4-*.json")}
print(f"  files: {sorted(probe_files)}")
data = {}
for fname, f in probe_files.items():
    d = json.load(open(f))
    label = "cu128" if "cu128" in fname else "cu118"
    data[label] = d

if len(data) == 2:
    a, b = data["cu128"], data["cu118"]
    for k in ("torch", "cuda_build", "cudnn", "driver", "gpu_name"):
        print(f"  {k:12} cu128={a['environment'].get(k,'?')}   cu118={b['environment'].get(k,'?')}")
    ra = a["ops"]
    rb = b["ops"]
    ops = sorted(set(ra) & set(rb))
    print(f"\n  {'operator':26}{'type':8}{'forward':>10}{'grad':>10}")
    print("  " + "-" * 56)
    same_lin = same_conv = n_lin = n_conv = 0
    for op in ops:
        if not isinstance(ra[op], dict) or "forward" not in ra[op]:
            continue
        kind = "conv" if op.startswith("conv") else ("linear" if op.startswith("linear") else "other")
        fw = "SAME" if ra[op]["forward"] == rb[op]["forward"] else "different"
        gr = "SAME" if ra[op].get("grad_input") == rb[op].get("grad_input") else "different"
        if kind == "linear":
            n_lin += 1; same_lin += (fw == "SAME")
        elif kind == "conv":
            n_conv += 1; same_conv += (fw == "SAME")
        print(f"  {op:26}{kind:8}{fw:>10}{gr:>10}")
    print("  " + "-" * 56)
    print(f"  identical dense GEMMs : {same_lin}/{n_lin}")
    print(f"  identical convolutions: {same_conv}/{n_conv}")
    print("\n  Prediction: linear ALL identical, conv ALL different.")
else:
    print("  ERROR: there are not two probes to compare")
